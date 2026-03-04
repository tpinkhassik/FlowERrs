import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, SequentialSampler

from utils.data_utils import ReactionDataset
from settings import Args


class FlowERDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        if stage in ("fit", None):
            with open(self.args.train_path, "r") as f:
                train_smiles = f.readlines()
            with open(self.args.val_path, "r") as f:
                val_smiles = f.readlines()

            self.train_dataset = ReactionDataset(self.args, train_smiles)
            self.val_dataset = ReactionDataset(self.args, val_smiles)

    def train_dataloader(self):
        """Called every epoch when reload_dataloaders_every_n_epochs=1.

        Re-sorts, re-shuffles-in-bucket, and re-batches the dataset each time,
        matching the original train.py behavior.
        """
        ds = self.train_dataset
        ds.sort()
        ds.shuffle_in_bucket(bucket_size=1000)
        ds.batch(batch_type=self.args.batch_type, batch_size=self.args.train_batch_size)

        return DataLoader(
            dataset=ds,
            batch_size=1,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=lambda batch: batch[0],
            pin_memory=True,
        )

    def val_dataloader(self):
        ds = self.val_dataset
        ds.sort()
        ds.shuffle_in_bucket(bucket_size=1000)
        ds.batch(batch_type=self.args.batch_type, batch_size=self.args.val_batch_size)

        return DataLoader(
            dataset=ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=lambda batch: batch[0],
            pin_memory=True,
        )
