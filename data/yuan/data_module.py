from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset, random_split


@dataclass
class YuanDataset:
    train_dataset: TensorDataset
    val_dataset: TensorDataset


class YuanDataModule(pl.LightningDataModule):
    """
    Data module for Inspur Yuan dataset.
    """
    def __init__(
            self,
            batch_size: int = 32,
            num_workers: int = 8,
            processed_data_path: str = './processed_data.npz',
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.processed_data_path = processed_data_path
        self.dataset: Optional[YuanDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            npz_data = np.load(self.processed_data_path)
            ids = torch.from_numpy(npz_data['id'].astype(np.int64))
            print(torch.max(ids))
            attention_masks = torch.from_numpy(npz_data['attention_mask'].astype(np.int64))
            dataset = TensorDataset(ids, attention_masks)
            train_dataset, val_dataset = random_split(dataset,
                                                      [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
            self.dataset = YuanDataset(train_dataset, val_dataset)

    def train_dataloader(self):
        return DataLoader(self.dataset.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


if __name__ == '__main__':
    dm = YuanDataModule(batch_size=32, num_workers=8)
    dm.setup()
    ids, masks = next(iter(dm.train_dataloader()))
    print(ids.shape)
