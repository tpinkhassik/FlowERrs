"""
Diverse evaluation script using DPP-coupled ODE integration (DiverseFlow).

Drop-in replacement for eval_multiGPU.py that couples K samples per reaction
via a DPP repulsive term during ODE integration, promoting diverse predictions.
"""

import os
import glob
import datetime
import torch
import numpy as np
from rdkit import Chem
from utils.data_utils import ReactionDataset, BEmatrix_to_mol, chiral_vec_to_mol, ps
from utils.rounding import saferound_tensor
import torch.distributed as dist
from train import init_model, init_loader
from utils.train_utils import log_rank_0, setup_logger, log_args
from settings import Args
from collections import defaultdict
import time

from model.diverse_flow import diverse_ode_integrate
from eval_multiGPU import (
    is_sym, redist_fix, redistribute_round, custom_round,
    chiral_delta_round, chiral_round,
    standardize_smiles, standardize_smiles_chiral, split_number,
)

ps = Chem.SmilesParserParams()
ps.removeHs = False
ps.sanitize = True

start = time.time()


def predict_batch_diverse(args, batch_idx, data_batch, model, flow, split):
    """Predict using DPP-coupled diverse ODE integration.

    Unlike standard eval, all K samples per reaction MUST be in the same batch
    for DPP coupling to work. When split > 1, we split across reactions (not
    across samples), processing fewer reactions at a time but keeping all K
    samples per reaction together.
    """
    src_data_indices = data_batch.src_data_indices
    y = data_batch.src_token_ids
    y_len = data_batch.src_lens
    x0 = data_batch.src_matrices
    cv0 = data_batch.src_chiral_vecs
    matrix_masks = data_batch.matrix_masks
    node_masks = data_batch.node_masks
    use_chirality = getattr(model, 'use_chirality', True)

    if use_chirality:
        cv0_delta = torch.zeros_like(cv0)
        cv0_delta = torch.where(node_masks.bool(), cv0_delta, cv0)
    else:
        cv0_delta = None

    batch_size, n, n = x0.shape
    sample_size = args.sample_size

    log_rank_0(f"[Diverse] Batch idx: {batch_idx}, batch_shape {batch_size, n, n} {(time.time() - start): .2f}s")

    # Split across reactions, not across samples, to keep K samples coupled
    if split > 1:
        chunk_size = max(1, batch_size // split)
        reaction_chunks = list(range(0, batch_size, chunk_size))
    else:
        reaction_chunks = [0]
        chunk_size = batch_size

    all_traj_be = []
    all_traj_cv = []

    for chunk_start in reaction_chunks:
        chunk_end = min(chunk_start + chunk_size, batch_size)
        cs = chunk_end - chunk_start

        x0_chunk = x0[chunk_start:chunk_end]
        y_chunk = y[chunk_start:chunk_end]
        y_len_chunk = y_len[chunk_start:chunk_end]
        mm_chunk = matrix_masks[chunk_start:chunk_end]
        nm_chunk = node_masks[chunk_start:chunk_end]

        x0_repeated = x0_chunk.repeat_interleave(sample_size, dim=0)

        x0_sample_repeated = flow.sample_be_matrix(x0_repeated)

        matrix_masks_repeated = mm_chunk.repeat_interleave(sample_size, dim=0)
        node_masks_repeated = nm_chunk.repeat_interleave(sample_size, dim=0)

        x0_sample_repeated = x0_sample_repeated.masked_fill(~(matrix_masks_repeated.bool()), 0)

        if use_chirality:
            cv0_chunk = cv0_delta[chunk_start:chunk_end]
            cv0_repeated = cv0_chunk.repeat_interleave(sample_size, dim=0)
            cv0_sample_repeated = flow.sample_chiral_vec(cv0_repeated)
            cv0_sample_repeated = cv0_sample_repeated.masked_fill(~(node_masks_repeated.bool()), 0)
        else:
            cv0_sample_repeated = torch.zeros(x0_repeated.shape[0], x0_repeated.shape[1], device=x0.device)

        torch.cuda.empty_cache()

        y_repeated = y_chunk.repeat_interleave(sample_size, dim=0)
        y_emb_repeated = model.id2emb(y_repeated)
        y_len_repeated = y_len_chunk.repeat_interleave(sample_size, dim=0)

        traj_be, traj_cv = diverse_ode_integrate(
            model, y_emb_repeated, y_len_repeated,
            x0_sample_repeated, cv0_sample_repeated,
            matrix_masks_repeated, node_masks_repeated,
            K=sample_size,
            num_steps=args.diverse_num_steps,
            gamma=args.diverse_gamma,
            diverse_be=True,
            diverse_cv=False,
        )

        del matrix_masks_repeated, node_masks_repeated

        # traj shape: (2, cs*sample_size, n, n) -> collect per reaction
        traj_be_cpu = traj_be.detach().cpu()
        traj_cv_cpu = traj_cv.detach().cpu() if use_chirality else None
        for r in range(cs):
            start_idx = r * sample_size
            end_idx = (r + 1) * sample_size
            all_traj_be.append(traj_be_cpu[:, start_idx:end_idx])
            if traj_cv_cpu is not None:
                all_traj_cv.append(traj_cv_cpu[:, start_idx:end_idx])

    # Concat all: (2, batch_size * sample_size, n, n)
    traj_be = torch.cat(all_traj_be, dim=1)
    traj_cv = torch.cat(all_traj_cv, dim=1) if all_traj_cv else None
    return traj_be, traj_cv


def get_predictions(args, model, flow, data_loader, iter_count=np.inf, write_o=None):
    accuracy = []
    model.eval()
    with torch.no_grad():
        log_rank_0('Start Diverse ODE Prediction...')
        if dist.get_rank() == 0:
            inferenced_indexes = set()

        for batch_idx, data_batch in enumerate(data_loader):
            if batch_idx >= iter_count: break
            data_batch.to(args.device)

            src_data_indices = data_batch.src_data_indices
            x0 = data_batch.src_matrices
            cv0 = data_batch.src_chiral_vecs
            y_len = data_batch.src_lens
            batch_size, n, n = x0.shape
            src_smiles_list = data_batch.src_smiles_list
            tgt_smiles_list = data_batch.tgt_smiles_list
            tgt_chiral_vec_list = data_batch.tgt_chiral_vecs

            if (batch_size*n*n) <= 15*130*130:
                traj_be, traj_cv = predict_batch_diverse(args, batch_idx, data_batch, model, flow, 1)
            else:
                traj_be, traj_cv = predict_batch_diverse(args, batch_idx, data_batch, model, flow, 2)

            if torch.distributed.is_initialized() and dist.get_world_size() > 1:
                gathered_results = [None for _ in range(dist.get_world_size())]
                dist.gather_object(
                    (src_data_indices,
                     traj_be, traj_cv,
                     x0,
                     cv0,
                     y_len,
                     src_smiles_list,
                     tgt_smiles_list,
                     tgt_chiral_vec_list,
                     ),
                    gathered_results if dist.get_rank() == 0 else None,
                    dst=0
                )
            else:
                gathered_results = [
                    (src_data_indices,
                     traj_be,
                     traj_cv,
                     x0,
                     cv0,
                     y_len,
                     src_smiles_list,
                     tgt_smiles_list,
                     tgt_chiral_vec_list,
                     )
                    ]

            if dist.get_rank() > 0:
                continue

            for result in gathered_results:
                src_data_indices, traj_be, traj_cv, x0, cv0, y_len, src_smiles_list, tgt_smiles_list, tgt_chiral_vec_list = result
                batch_size, n, n = x0.shape
                has_cv = traj_cv is not None

                last_be_step = traj_be[-1]

                product_BE_matrices = custom_round(last_be_step)

                if has_cv:
                    last_cv_step = traj_cv[-1]
                    pred_delta_chiral_vecs = chiral_delta_round(last_cv_step)
                    src_chiral_vecs = cv0.repeat_interleave(args.sample_size, dim=0).cpu()
                    product_chiral_vecs = chiral_round(src_chiral_vecs + pred_delta_chiral_vecs)
                    product_chiral_vecs_batch = torch.split(product_chiral_vecs, args.sample_size)

                product_BE_matrices_batch = torch.split(product_BE_matrices, args.sample_size)

                for idx in range(batch_size):
                    reac_smi = src_smiles_list[idx]
                    product_smi = tgt_smiles_list[idx]
                    tgt_chiral_vecs = tgt_chiral_vec_list[idx]
                    product_BE_matrices = product_BE_matrices_batch[idx]
                    product_chiral_vecs = product_chiral_vecs_batch[idx] if has_cv else None

                    data_idx = int(src_data_indices[idx].detach().cpu())
                    if data_idx in inferenced_indexes: continue
                    else: inferenced_indexes.add(data_idx)

                    reac_mol = Chem.MolFromSmiles(reac_smi, ps)
                    prod_mol = Chem.MolFromSmiles(product_smi, ps)

                    tgt_smiles = standardize_smiles(prod_mol)
                    tgt_chiral_smi = Chem.MolToSmiles(prod_mol, isomericSmiles=True, allHsExplicit=True)

                    num_nodes = int(y_len[idx])
                    reac_be_matrix = x0[idx][:num_nodes, :num_nodes].detach().cpu().numpy()

                    if has_cv:
                        reac_chiral_vec = cv0[idx][:num_nodes].detach().cpu().numpy()
                        tgt_delta_chiral_vec = tgt_chiral_vecs[:num_nodes].cpu().numpy()
                        tgt_chiral_vec = np.clip(reac_chiral_vec + tgt_delta_chiral_vec, -1, 1)
                        tgt_is_chiral = (tgt_chiral_vec != 0)
                        tgt_changed_is_chiral = (tgt_delta_chiral_vec != 0) & tgt_is_chiral

                    n_pad = product_BE_matrices.shape[1]
                    if has_cv:
                        flat_pairs = torch.cat([
                            product_BE_matrices.view(args.sample_size, -1).float(),
                            product_chiral_vecs.float(),
                        ], dim=1)
                    else:
                        flat_pairs = product_BE_matrices.view(args.sample_size, -1).float()
                    pairs, pair_counts = torch.unique(flat_pairs, dim=0, return_counts=True)
                    pairs, pair_counts = pairs.cpu().numpy(), pair_counts.cpu().numpy()

                    not_sym = 0
                    correct = wrong_smi_conserved = wrong_smi_non_conserved = 0
                    no_smi_conserved = no_smi_non_conserved = 0
                    correct_cv = 0
                    correct_centers = false_positives = false_negatives = 0
                    correct_chiral_centers = 0
                    total_chiral_centers = 0
                    wrong_sign_chiral = 0
                    correct_changed_chiral_centers = 0
                    total_changed_chiral_centers = 0
                    pred_chiral_smi_dict = defaultdict(int)

                    for i in range(pairs.shape[0]):
                        pair, count = pairs[i], pair_counts[i]
                        pred_be = pair[:n_pad * n_pad].reshape(n_pad, n_pad)[:num_nodes, :num_nodes]
                        pred_cv = pair[n_pad * n_pad:][:num_nodes] if has_cv else None

                        pred_be = redist_fix(pred_be, reac_smi, reac_be_matrix)

                        assert pred_be.shape == reac_be_matrix.shape, "pred and reac not the same shape"

                        if not is_sym(pred_be):
                            not_sym += 1

                        try:
                            pred_mol = BEmatrix_to_mol(reac_mol, pred_be)

                            if has_cv:
                                try:
                                    pred_mol_chiral = chiral_vec_to_mol(pred_mol, pred_cv)
                                    pred_smi_mapped = Chem.MolToSmiles(pred_mol_chiral, isomericSmiles=True, allHsExplicit=True)
                                    pred_mol_chiral = Chem.MolFromSmiles(pred_smi_mapped, ps)
                                    pred_chiral_smi_dict[standardize_smiles_chiral(pred_mol_chiral)] += count
                                except Exception:
                                    pass

                            pred_smi = standardize_smiles(pred_mol)
                            pred_mol = Chem.MolFromSmiles(pred_smi, ps)
                            pred_smi = standardize_smiles(pred_mol)
                            tgt_mol = Chem.MolFromSmiles(tgt_smiles, ps)
                            tgt_smiles = standardize_smiles(tgt_mol)

                            if pred_smi == tgt_smiles and pred_be.sum() == reac_be_matrix.sum():
                                correct += count
                            elif pred_be.sum() == reac_be_matrix.sum():
                                wrong_smi_conserved += count
                            else:
                                wrong_smi_non_conserved += count
                        except:
                            if pred_be.sum() == reac_be_matrix.sum():
                                no_smi_conserved += count
                            else:
                                no_smi_non_conserved += count

                        if has_cv:
                            correct_centers += count * (pred_cv == tgt_chiral_vec).sum()
                            total_chiral_centers += count * tgt_is_chiral.sum()
                            correct_chiral_centers += count * (pred_cv[tgt_is_chiral] == tgt_chiral_vec[tgt_is_chiral]).sum()

                            total_changed_chiral_centers += count * tgt_changed_is_chiral.sum()
                            correct_changed_chiral_centers += count * (pred_cv[tgt_changed_is_chiral] == tgt_chiral_vec[tgt_changed_is_chiral]).sum()

                            if np.array_equal(pred_cv, tgt_chiral_vec):
                                correct_cv += count
                            else:
                                false_positives += count * (pred_cv[tgt_chiral_vec == 0] != 0).sum()
                                false_negatives += count * (pred_cv[tgt_chiral_vec != 0] == 0).sum()
                                wrong_sign_chiral += count * (
                                    (pred_cv[tgt_is_chiral] != 0)
                                    & (pred_cv[tgt_is_chiral] != tgt_chiral_vec[tgt_is_chiral])
                                ).sum()

                    chiral_center_acc = (
                        float(correct_chiral_centers) / float(total_chiral_centers)
                        if total_chiral_centers > 0 else float("nan")
                    )

                    correct_chiral_smi = pred_chiral_smi_dict.get(tgt_chiral_smi, 0)

                    metric = [
                        correct,
                        wrong_smi_conserved,
                        wrong_smi_non_conserved,
                        no_smi_conserved,
                        no_smi_non_conserved,
                        correct_cv,
                        correct_centers,
                        false_positives,
                        false_negatives,
                        correct_chiral_centers,
                        total_chiral_centers,
                        wrong_sign_chiral,
                        chiral_center_acc,
                        correct_changed_chiral_centers,
                        total_changed_chiral_centers,
                        0,
                        correct_chiral_smi,
                    ]

                    predictions = [(smi, pred_chiral_smi_dict[smi]) for smi in pred_chiral_smi_dict]
                    n_unique = pairs.shape[0]
                    if write_o is not None:
                        write_o.write(f"{metric}|{not_sym}|{predictions}|unique={n_unique}/{args.sample_size}\n")
                        write_o.flush()
                    accuracy.append(metric)

    return accuracy


def main(args):
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = args.device
    if args.local_rank != -1:
        dist.init_process_group(backend=args.backend, init_method='env://', timeout=datetime.timedelta(0, 7200))
        torch.cuda.set_device(args.local_rank)
        torch.backends.cudnn.benchmark = True

    if args.do_validate:
        phase = "valid-diverse"
        checkpoints = glob.glob(os.path.join(args.model_path, "*.pt"))
        checkpoints = sorted(
            checkpoints,
            key=lambda ckpt: int(ckpt.split(".")[-2].split("_")[-1]),
            reverse=True
        )
        assert len(args.steps2validate) > 1, "Nothing to validate on"
        checkpoints = [ckpt for ckpt in checkpoints
            if ckpt.split(".")[-2].split("_")[0] in args.steps2validate]
    else:
        phase = "test-diverse"
        checkpoints = [os.path.join(args.model_path, args.model_name)]

    for ckpt_i, checkpoint in enumerate(checkpoints):
        state = torch.load(checkpoint, weights_only=False, map_location=device)
        pretrain_args = state["args"]
        pretrain_args.load_from = checkpoint
        pretrain_args.device = device

        pretrain_state_dict = state["state_dict"]
        pretrain_args.local_rank = args.local_rank

        attn_model, flow, state = init_model(pretrain_args)
        if hasattr(attn_model, "module"):
            attn_model = attn_model.module

        pretrain_state_dict = {k.replace("module.", ""): v for k, v in pretrain_state_dict.items()}
        attn_model.load_state_dict(pretrain_state_dict, strict=False)
        log_rank_0(f"Loaded pretrained state_dict from {checkpoint} (use_chirality={getattr(attn_model, 'use_chirality', True)})")
        log_rank_0(f"DiverseFlow params: gamma={args.diverse_gamma}, steps={args.diverse_num_steps}")

        os.makedirs(args.result_path, exist_ok=True)
        results_path = os.path.join(
            args.result_path,
            f'{phase}-{args.sample_size}-g{args.diverse_gamma}-s{args.diverse_num_steps}-{checkpoint.split(".")[-2]}.txt'
        )
        if os.path.isfile(results_path):
            with open(results_path, 'r') as fp:
                n_lines = len(fp.readlines())
                file_mod = 'a'
                start = n_lines
            log_rank_0(f"Continuing previous runs at reaction {start}...")
        else:
            log_rank_0("Starting new run...")
            file_mod = 'w'
            start = 0

        if args.do_validate:
            with open(args.val_path, 'r') as test_o:
                test_smiles_list = test_o.readlines()[start:]
        else:
            with open(args.test_path, 'r') as test_o:
                test_smiles_list = test_o.readlines()[start:]

        assert len(test_smiles_list) > 0, "Nothing to do inference"

        test_dataset = ReactionDataset(args, test_smiles_list)
        test_loader = init_loader(args, test_dataset,
                                batch_size=args.test_batch_size,
                                shuffle=False, epoch=None, use_sort=False)

        with open(results_path, file_mod) as result_o:
            metrics = get_predictions(args, attn_model, flow, test_loader, write_o=result_o)
        if dist.get_rank() == 0:
            metrics = np.array(metrics)
            topk_accuracies = np.mean(metrics[:, 0].astype(bool))
            n_unique_avg = "N/A"  # logged per-reaction in file
            log_rank_0(f"Topk accuracies: {(topk_accuracies * 100): .2f}")


if __name__ == "__main__":
    args = Args
    args.local_rank = int(os.environ["LOCAL_RANK"]) if os.environ.get("LOCAL_RANK") else -1
    logger = setup_logger(args, "eval_diverse")
    log_args(args, 'diverse_evaluation')
    main(args)
