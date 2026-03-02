
import os
import glob
import datetime
import torch
import numpy as np
from rdkit import Chem
import torchdiffeq
from utils.data_utils import ReactionDataset, BEmatrix_to_mol, chiral_vec_to_mol, ps
from utils.rounding import saferound_tensor
import torch.distributed as dist
from train import init_model, init_loader
from utils.train_utils import log_rank_0, setup_logger, log_args
from settings import Args
from collections import defaultdict
import time
import iteround

ps = Chem.SmilesParserParams()
ps.removeHs = False
ps.sanitize = True

def is_sym(a):
    return (a.transpose(1, 0) == a).all()

def redist_fix(pred_matrix, reac_smi, reac_be_matrix):
    pred_electron_sum = np.zeros([len(pred_matrix)])
    for i in range(len(pred_matrix)):
        pred_electron_sum[i] = \
        np.sum(pred_matrix[i, :]) + np.sum(pred_matrix[:, i]) - pred_matrix[i, i]

    reac_electron_sum = np.zeros([len(reac_be_matrix)])
    for i in range(len(reac_be_matrix)):
        reac_electron_sum[i] = \
        np.sum(reac_be_matrix[i, :]) + np.sum(reac_be_matrix[:, i]) - reac_be_matrix[i, i]

    diff = reac_electron_sum - pred_electron_sum

    if np.sum(diff) == 0:
        pred_matrix[np.diag_indices_from(pred_matrix)] += diff

    return pred_matrix

# # old implementation uses CPU
# def redistribute_round(x):
#     rounded_diff = iteround.saferound(x.flatten().cpu().numpy().tolist(), 0)
#     rounded_diff = torch.as_tensor(rounded_diff, dtype=torch.float).view(*x.shape)
#     return rounded_diff.to(x)

# new implementation uses GPU
def redistribute_round(x):
    rounded = saferound_tensor(x, places=0, strategy="difference")
    return rounded

def custom_round(x):
    output = []
    for i in range(x.shape[0]):
        try: output.append(redistribute_round(x[i]))
        except: output.append(torch.round(x[i]))
    return torch.stack(output)

def chiral_delta_round(x: torch.Tensor) -> torch.Tensor:
    """
    Round a predicted chirality delta, bounded to be within -2 and +2.
    Args: x (torch.Tensor): input delta-chirality output
    Returns: torch.Tensor: rounded tensor with elements in {-2, -1, 0, +1, +2}
    """
    return torch.clamp(torch.round(x), min=-2, max=2)

def chiral_round(x: torch.Tensor) -> torch.Tensor:
    """
    Round a chiral vector, bounded to be within -1 and +1.
    Args: x (torch.Tensor): input absolute chiral vector output
    Returns: torch.Tensor: rounded tensor with elements in {-1, 0, +1}
    """
    return torch.clamp(torch.round(x), min=-1, max=1)





def standardize_smiles(mol):
    [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
    return Chem.MolToSmiles(mol, isomericSmiles=False, allHsExplicit=True)

def standardize_smiles_chiral(mol):
    [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
    return Chem.MolToSmiles(mol, isomericSmiles=True, allHsExplicit=True)

def split_number(number, num_parts):
    if number % num_parts != 0:
        raise ValueError("The number cannot be evenly divided into the specified number of parts.")
    return [number // num_parts] * num_parts

start = time.time()
def predict_batch(args, batch_idx, data_batch, model, flow, split, rand_matrix=None):
    src_data_indices = data_batch.src_data_indices
    y = data_batch.src_token_ids
    y_len = data_batch.src_lens
    x0 = data_batch.src_matrices
    cv0 = data_batch.src_chiral_vecs
    # x1 = data_batch.tgt_matrices
    # cv1 = data_batch.tgt_chiral_vecs
    matrix_masks = data_batch.matrix_masks
    node_masks = data_batch.node_masks
    cv0_delta = torch.zeros_like(cv0)
    cv0_delta = torch.where(node_masks.bool(), cv0_delta, cv0)

    batch_size, n, n = x0.shape
    
    log_rank_0(f"Batch idx: {batch_idx}, batch_shape {batch_size, n, n} {(time.time() - start): .2f}s")
    # --------ODE inference--------------#
    SAMPLE_BATCH = args.sample_size
    # split_sample_batches = split_number(SAMPLE_BATCH, 2) if n >= 400 else split_number(SAMPLE_BATCH, 1)
    # split_sample_batches = split_number(SAMPLE_BATCH, 1)
    split_sample_batches = split_number(SAMPLE_BATCH, split)
    
    big_traj_list = []
    for sample_size in split_sample_batches:
        src_data_indices = src_data_indices.repeat_interleave(sample_size, dim=0)
        x0_repeated = x0.repeat_interleave(sample_size, dim=0)
        cv0_repeated = cv0_delta.repeat_interleave(sample_size, dim=0)

        x0_sample_repeated = flow.sample_be_matrix(x0_repeated)
        cv0_sample_repeated = flow.sample_chiral_vec(cv0_repeated)


        matrix_masks_repeated = matrix_masks.repeat_interleave(sample_size, dim=0)
        node_masks_repeated = node_masks.repeat_interleave(sample_size, dim=0)

        x0_sample_repeated = x0_sample_repeated.masked_fill(~(matrix_masks_repeated.bool()), 0) # ode initial step has RMS norm thus padding nan has to be swap to 0
        cv0_sample_repeated = cv0_sample_repeated.masked_fill(~(node_masks_repeated.bool()), 0)

        del matrix_masks_repeated
        del node_masks_repeated

        torch.cuda.empty_cache()

        y_repeated = y.repeat_interleave(sample_size, dim=0)
        y_emb_repeated = model.id2emb(y_repeated)
        y_len_batch_repeated = y_len.repeat_interleave(sample_size, dim=0)
        
        def velocity(t, state):
            x, cv = state
            v_be, v_cv = model.forward(y_emb_repeated, y_len_batch_repeated, x, t, cv)
            return (v_be, v_cv)

        traj_be, traj_cv = torchdiffeq.odeint(
            velocity,
            (x0_sample_repeated, cv0_sample_repeated),
            torch.linspace(0, 1, 2).to(args.device),
            atol=1e-4,
            rtol=1e-4,
            method="dopri5",
        )
        big_traj_list.append((
            traj_be.transpose(0, 1).detach().cpu(),
            traj_cv.transpose(0, 1).detach().cpu(),
            sample_size,
            ))

    # merging
    all_traj_be = []
    all_traj_cv = []
    for bs in range(batch_size):
        for traj_be, traj_cv, sample_size in big_traj_list:
            all_traj_be.append(traj_be[bs*sample_size:(bs+1)*sample_size].transpose(0, 1))
            all_traj_cv.append(traj_cv[bs*sample_size:(bs+1)*sample_size].transpose(0, 1))
    traj_be = torch.concat(all_traj_be, dim=1) # concat on sampling dimension
    traj_cv = torch.concat(all_traj_cv, dim=1)
    # ------------------------------------#
    return traj_be, traj_cv

def get_predictions(args, model, flow, data_loader, iter_count=np.inf, write_o=None):
    accuracy = []
    model.eval()
    with torch.no_grad():
        log_rank_0('Start ODE Prediction...')
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


            # if (batch_size*n*n) <= 5*360*360:
            if (batch_size*n*n) <= 15*130*130:
                traj_be, traj_cv = predict_batch(args, batch_idx, data_batch, model, flow, 1)
            else:
                traj_be, traj_cv = predict_batch(args, batch_idx, data_batch, model, flow, 2)

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

                last_be_step = traj_be[-1]
                last_cv_step = traj_cv[-1]


                product_BE_matrices = custom_round(last_be_step)
                pred_delta_chiral_vecs = chiral_delta_round(last_cv_step)
                src_chiral_vecs = cv0.repeat_interleave(args.sample_size, dim=0).cpu()
                product_chiral_vecs = chiral_round(src_chiral_vecs + pred_delta_chiral_vecs)

                product_BE_matrices_batch = torch.split(product_BE_matrices, args.sample_size)
                product_chiral_vecs_batch = torch.split(product_chiral_vecs, args.sample_size)

                for idx in range(batch_size):
                    reac_smi, product_smi, tgt_chiral_vecs, product_BE_matrices, product_chiral_vecs = \
                        src_smiles_list[idx], tgt_smiles_list[idx], tgt_chiral_vec_list[idx], product_BE_matrices_batch[idx], product_chiral_vecs_batch[idx]
                    
                    data_idx = int(src_data_indices[idx].detach().cpu())
                    if data_idx in inferenced_indexes: continue
                    else: inferenced_indexes.add(data_idx)

                    reac_mol = Chem.MolFromSmiles(reac_smi, ps)
                    prod_mol = Chem.MolFromSmiles(product_smi, ps)

                    tgt_smiles = standardize_smiles(prod_mol)
                    # atom maps cleared in-place by standardize_smiles; stereo preserved
                    tgt_chiral_smi = Chem.MolToSmiles(prod_mol, isomericSmiles=True, allHsExplicit=True)

                    num_nodes = int(y_len[idx])
                    reac_be_matrix = x0[idx][:num_nodes, :num_nodes].detach().cpu().numpy()
                    reac_chiral_vec = cv0[idx][:num_nodes].detach().cpu().numpy()
                    tgt_delta_chiral_vec = tgt_chiral_vecs[:num_nodes].cpu().numpy()
                    tgt_chiral_vec = np.clip(reac_chiral_vec + tgt_delta_chiral_vec, -1, 1)
                    tgt_is_chiral = (tgt_chiral_vec != 0)
                    # centers where chirality changed AND remain chiral in the product
                    tgt_changed_is_chiral = (tgt_delta_chiral_vec != 0) & tgt_is_chiral

                    # Unique on joint (BE matrix, chiral vector) pairs to preserve pairing
                    n_pad = product_BE_matrices.shape[1]
                    flat_pairs = torch.cat([
                        product_BE_matrices.view(args.sample_size, -1).float(),
                        product_chiral_vecs.float(),
                    ], dim=1)
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
                        pred_cv = pair[n_pad * n_pad:][:num_nodes]

                        pred_be = redist_fix(pred_be, reac_smi, reac_be_matrix)

                        assert pred_be.shape == reac_be_matrix.shape, "pred and reac not the same shape"

                        if not is_sym(pred_be):
                            not_sym += 1

                        # Bonding evaluation
                        try:
                            pred_mol = BEmatrix_to_mol(reac_mol, pred_be)

                            # Apply chirality before standardize_smiles clears atom maps in-place
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

                        # Chiral vector evaluation
                        correct_centers += count * (pred_cv == tgt_chiral_vec).sum()
                        total_chiral_centers += count * tgt_is_chiral.sum()
                        correct_chiral_centers += count * (pred_cv[tgt_is_chiral] == tgt_chiral_vec[tgt_is_chiral]).sum()

                        # Changed-center evaluation (inversion / creation)
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
                        correct,                          # 0
                        wrong_smi_conserved,              # 1
                        wrong_smi_non_conserved,          # 2
                        no_smi_conserved,                 # 3
                        no_smi_non_conserved,             # 4
                        correct_cv,                       # 5
                        correct_centers,                  # 6
                        false_positives,                  # 7
                        false_negatives,                  # 8
                        correct_chiral_centers,           # 9
                        total_chiral_centers,             # 10
                        wrong_sign_chiral,                # 11
                        chiral_center_acc,                # 12
                        correct_changed_chiral_centers,   # 13
                        total_changed_chiral_centers,     # 14
                        0,                                # 15 (reserved)
                        correct_chiral_smi,               # 16
                    ]

                    predictions = [(smi, pred_chiral_smi_dict[smi]) for smi in pred_chiral_smi_dict]
                    if write_o is not None:
                        write_o.write(f"{metric}|{not_sym}|{predictions}\n")
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
        phase = "valid"
        checkpoints = glob.glob(os.path.join(args.model_path, "*.pt"))
        checkpoints = sorted(
            checkpoints,
            key=lambda ckpt: int(ckpt.split(".")[-2].split("_")[-1]),
            reverse=True
        )
        assert len(args.steps2validate) > 1, "Nothing to validate on"
        checkpoints = [ckpt for ckpt in checkpoints 
            if ckpt.split(".")[-2].split("_")[0] in args.steps2validate] # lr0.001
    else:
        phase = "test"
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
            attn_model = attn_model.module        # unwrap DDP attn_model to enable accessing attn_model func directly

        pretrain_state_dict = {k.replace("module.", ""): v for k, v in pretrain_state_dict.items()}
        attn_model.load_state_dict(pretrain_state_dict)
        log_rank_0(f"Loaded pretrained state_dict from {checkpoint}")

        os.makedirs(args.result_path, exist_ok=True)
        results_path = os.path.join(args.result_path, f'{phase}-{args.sample_size}-{checkpoint.split(".")[-2]}.txt')
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
            topk_accuracies = np.mean(metrics[:, 0].astype(bool)) # correct smiles
            log_rank_0(f"Topk accuracies: {(topk_accuracies * 100): .2f}")


if __name__ == "__main__":
    args = Args
    args.local_rank = int(os.environ["LOCAL_RANK"]) if os.environ.get("LOCAL_RANK") else -1
    logger = setup_logger(args, "eval")
    log_args(args, 'evaluation') 
    main(args)
