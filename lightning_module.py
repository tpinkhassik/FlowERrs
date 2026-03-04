import os
import logging
import numpy as np
import torch
import pytorch_lightning as pl
import torchdiffeq
from rdkit import Chem

from model.attn_encoder import AttnEncoderXL
from model.flow_matching import ConditionalFlowMatcher
from utils.data_utils import BEmatrix_to_mol
from utils.train_utils import NoamLR, param_norm, grad_norm
from torch.nn.init import xavier_uniform_

ps = Chem.SmilesParserParams()
ps.removeHs = False
ps.sanitize = True


def standardize_smiles(mol):
    [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
    return Chem.MolToSmiles(mol, isomericSmiles=False, allHsExplicit=True)


class FlowERLightningModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = AttnEncoderXL(args)
        self.flow = ConditionalFlowMatcher(args)

    def init_xavier(self):
        for p in self.model.parameters():
            if p.dim() > 1 and p.requires_grad:
                xavier_uniform_(p)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch.to(device)
        return batch

    # ---- Training ----

    def training_step(self, train_batch, batch_idx):
        y = train_batch.src_token_ids
        y_len = train_batch.src_lens
        x0 = train_batch.src_matrices
        x1 = train_batch.tgt_matrices
        cv1 = train_batch.tgt_chiral_vecs
        matrix_masks = train_batch.matrix_masks
        node_masks = train_batch.node_masks
        step_weights = train_batch.step_weights.float()

        x0_sample = self.flow.sample_be_matrix(x0)
        cv0_delta = torch.zeros_like(cv1)
        cv0_delta = torch.where(node_masks.bool(), cv0_delta, cv1)
        cv0_sample = self.flow.sample_chiral_vec(cv0_delta)

        t = torch.rand(x0.shape[0]).type_as(x0)

        xt, cvt = self.flow.sample_conditional_pt(x0, x1, cv0_delta, cv1, t)
        ut = self.flow.compute_conditional_vector_field(x0_sample, x1)
        u_cvt = self.flow.compute_conditional_vector_field(cv0_sample, cv1)

        y_emb = self.model.id2emb(y)
        vt, v_cvt = self.model(y_emb, y_len, xt, t, cvt)

        be_err2 = ((vt - ut) * matrix_masks) ** 2
        cv_err2 = ((v_cvt - u_cvt) * node_masks) ** 2

        weight_denom = step_weights.sum().clamp_min(1e-12)
        be_loss = torch.sum(be_err2 * step_weights[:, None, None]) / weight_denom
        cv_loss = torch.sum(cv_err2 * step_weights[:, None]) / weight_denom

        loss = be_loss + cv_loss

        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/be_loss", be_loss, sync_dist=True)
        self.log("train/cv_loss", cv_loss, sync_dist=True)

        # Lightning handles accumulation: it calls .backward() on each micro-batch
        # then optimizer.step() after accumulate_grad_batches micro-batches.
        # To match original (loss / N).backward(), we divide here.
        return loss / self.args.accumulation_count

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.global_step > 0 and self.global_step % self.args.log_iter == 0:
            p_norm = param_norm(self.model)
            g_norm_val = grad_norm(self.model)
            self.log("train/param_norm", p_norm, rank_zero_only=True)
            self.log("train/grad_norm", g_norm_val, rank_zero_only=True)
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("train/lr", lr, rank_zero_only=True)

    # ---- Validation ----

    def on_validation_epoch_start(self):
        self._val_results = []

    def validation_step(self, val_batch, batch_idx):
        result = self._run_validation_batch(val_batch)
        self._val_results.append(result)
        return result

    def _run_validation_batch(self, val_batch):
        y = val_batch.src_token_ids
        y_len = val_batch.src_lens
        x0 = val_batch.src_matrices
        cv0 = val_batch.src_chiral_vecs
        matrix_masks = val_batch.matrix_masks
        node_masks = val_batch.node_masks
        src_smiles_list = val_batch.src_smiles_list
        tgt_smiles_list = val_batch.tgt_smiles_list

        batch_size, n, _ = x0.shape
        sample_size = self.args.sample_size

        cv0_delta = torch.zeros_like(cv0)
        cv0_delta = torch.where(node_masks.bool(), cv0_delta, cv0)

        x0_repeated = x0.repeat_interleave(sample_size, dim=0)
        cv0_repeated = cv0_delta.repeat_interleave(sample_size, dim=0)
        x0_sample_repeated = self.flow.sample_be_matrix(x0_repeated)
        cv0_sample_repeated = self.flow.sample_chiral_vec(cv0_repeated)

        matrix_masks_rep = matrix_masks.repeat_interleave(sample_size, dim=0)
        node_masks_rep = node_masks.repeat_interleave(sample_size, dim=0)
        x0_sample_repeated = x0_sample_repeated.masked_fill(~(matrix_masks_rep.bool()), 0)
        cv0_sample_repeated = cv0_sample_repeated.masked_fill(~(node_masks_rep.bool()), 0)

        y_repeated = y.repeat_interleave(sample_size, dim=0)
        y_emb_repeated = self.model.id2emb(y_repeated)
        y_len_repeated = y_len.repeat_interleave(sample_size, dim=0)

        def velocity(t, state):
            x, cv = state
            v_be, v_cv = self.model.forward(y_emb_repeated, y_len_repeated, x, t, cv)
            return (v_be, v_cv)

        traj_be, traj_cv = torchdiffeq.odeint(
            velocity,
            (x0_sample_repeated, cv0_sample_repeated),
            torch.linspace(0, 1, 2, device=self.device),
            atol=1e-4,
            rtol=1e-4,
            method="dopri5",
        )

        last_be = traj_be[-1].cpu()
        pred_matrices = torch.round(last_be)
        pred_matrices_batch = torch.split(pred_matrices, sample_size)

        correct_count = 0
        total_count = 0
        for idx in range(batch_size):
            reac_smi = src_smiles_list[idx]
            product_smi = tgt_smiles_list[idx]
            num_nodes = int(y_len[idx])

            try:
                prod_mol = Chem.MolFromSmiles(product_smi, ps)
                tgt_smi = standardize_smiles(prod_mol)
            except Exception:
                total_count += 1
                continue

            reac_mol = Chem.MolFromSmiles(reac_smi, ps)
            found = False
            for s in range(sample_size):
                pred_be = pred_matrices_batch[idx][s][:num_nodes, :num_nodes].numpy()
                try:
                    pred_mol = BEmatrix_to_mol(reac_mol, pred_be)
                    pred_smi = standardize_smiles(pred_mol)
                    pred_mol = Chem.MolFromSmiles(pred_smi, ps)
                    pred_smi = standardize_smiles(pred_mol)
                    if pred_smi == tgt_smi:
                        found = True
                        break
                except Exception:
                    continue
            if found:
                correct_count += 1
            total_count += 1

        return {"correct": correct_count, "total": total_count}

    def on_validation_epoch_end(self):
        if not hasattr(self, "_val_results") or not self._val_results:
            return
        total_correct = sum(r["correct"] for r in self._val_results)
        total_count = sum(r["total"] for r in self._val_results)
        acc = total_correct / max(total_count, 1)
        self.log("val/topk_accuracy", acc * 100, prog_bar=True, rank_zero_only=True)
        logging.info(f"Topk accuracies: {acc * 100:.2f}")
        self._val_results = []

    # ---- Optimizer ----

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            betas=(self.args.beta1, self.args.beta2),
            eps=self.args.eps,
            weight_decay=self.args.weight_decay,
        )
        scheduler = NoamLR(
            optimizer,
            model_size=self.args.emb_dim,
            warmup_steps=self.args.warmup_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    # ---- Legacy checkpoint loading ----

    @classmethod
    def load_from_legacy_checkpoint(cls, args, checkpoint_path):
        """Load a checkpoint saved by the original train.py."""
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        pretrain_args = state.get("args", args)
        for k, v in vars(args).items():
            if not hasattr(pretrain_args, k):
                setattr(pretrain_args, k, v)

        module = cls(pretrain_args)
        state_dict = state["state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        module.model.load_state_dict(state_dict)
        logging.info(f"Loaded legacy checkpoint from {checkpoint_path}")
        return module, state


class LossHistoryCallback(pl.Callback):
    """Writes per-checkpoint loss history CSV, matching original train.py behavior."""

    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self.loss_history_path = os.path.join(model_path, "loss_history.csv")
        self.ckpt_losses = {"total": [], "be": [], "cv": []}

    def on_fit_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            if not os.path.exists(self.loss_history_path):
                with open(self.loss_history_path, "w") as f:
                    f.write("step,total_loss,be_loss,cv_loss\n")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss_val = outputs["loss"].item() * pl_module.args.accumulation_count
        be_loss = trainer.callback_metrics.get("train/be_loss")
        cv_loss = trainer.callback_metrics.get("train/cv_loss")
        self.ckpt_losses["total"].append(loss_val)
        if be_loss is not None:
            self.ckpt_losses["be"].append(be_loss.item())
        if cv_loss is not None:
            self.ckpt_losses["cv"].append(cv_loss.item())

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if not trainer.is_global_zero:
            return
        step = trainer.global_step
        mean_total = float(np.mean(self.ckpt_losses["total"])) if self.ckpt_losses["total"] else float("nan")
        mean_be = float(np.mean(self.ckpt_losses["be"])) if self.ckpt_losses["be"] else float("nan")
        mean_cv = float(np.mean(self.ckpt_losses["cv"])) if self.ckpt_losses["cv"] else float("nan")
        with open(self.loss_history_path, "a") as f:
            f.write(f"{step},{mean_total:.8f},{mean_be:.8f},{mean_cv:.8f}\n")
        self.ckpt_losses = {"total": [], "be": [], "cv": []}
