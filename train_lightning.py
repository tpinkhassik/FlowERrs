import os
import glob
import logging
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from settings import Args
from lightning_module import FlowERLightningModule, LossHistoryCallback
from lightning_data import FlowERDataModule
from utils.train_utils import setup_logger, log_args, param_count


def main():
    args = Args
    # ReactionDataset.__init__ reads args.device (stores but never uses it).
    # Set a placeholder so it doesn't crash; Lightning handles real device placement.
    args.device = torch.device("cpu")
    logger = setup_logger(args, "train_lightning")
    log_args(args, "lightning_training")

    # --- Build or load model ---
    checkpoint_path = args.load_from
    if (checkpoint_path == "") or (not os.path.isfile(checkpoint_path)):
        ckpts = glob.glob(os.path.join(args.model_path, "model.*.pt"))
        checkpoint_path = max(ckpts, key=os.path.getmtime) if ckpts else ""
    checkpoint_exists = bool(checkpoint_path) and os.path.isfile(checkpoint_path)

    if checkpoint_exists:
        logging.info(f"Loading legacy checkpoint: {checkpoint_path}")
        model, state = FlowERLightningModule.load_from_legacy_checkpoint(args, checkpoint_path)
    else:
        logging.info("No checkpoint found; initializing from scratch with Xavier init.")
        model = FlowERLightningModule(args)
        model.init_xavier()
        state = None

    logging.info(f"Number of parameters: {param_count(model.model)}")

    # --- Data ---
    data_module = FlowERDataModule(args)

    # --- Callbacks ---
    os.makedirs(args.model_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.model_path,
        filename="lightning-{step}",
        every_n_train_steps=args.save_iter,
        save_top_k=-1,
    )

    loss_history_callback = LossHistoryCallback(args.model_path)

    # --- Trainer ---
    num_gpus = int(os.environ.get("NUM_GPUS_PER_NODE", 1))
    num_nodes = int(os.environ.get("NUM_NODES", 1))

    trainer = pl.Trainer(
        devices=num_gpus,
        num_nodes=num_nodes,
        strategy="ddp" if num_gpus > 1 or num_nodes > 1 else "auto",
        max_steps=args.max_steps,
        accumulate_grad_batches=args.accumulation_count,
        gradient_clip_val=args.clip_norm,
        val_check_interval=args.eval_iter,
        limit_val_batches=args.val_count,
        reload_dataloaders_every_n_epochs=1,
        callbacks=[checkpoint_callback, loss_history_callback],
        enable_progress_bar=True,
    )

    # --- Resume optimizer/scheduler state if available ---
    # Lightning's fit(ckpt_path=...) expects a Lightning checkpoint.
    # For legacy checkpoints, we already loaded model weights above.
    # Optimizer/scheduler state is NOT resumed from legacy checkpoints;
    # NoamLR will restart from step 0. For full resume, use a Lightning .ckpt file.
    lightning_ckpt = None
    lightning_ckpts = glob.glob(os.path.join(args.model_path, "lightning-*.ckpt"))
    if lightning_ckpts:
        lightning_ckpt = max(lightning_ckpts, key=os.path.getmtime)
        logging.info(f"Resuming from Lightning checkpoint: {lightning_ckpt}")

    trainer.fit(model, datamodule=data_module, ckpt_path=lightning_ckpt)


if __name__ == "__main__":
    main()
