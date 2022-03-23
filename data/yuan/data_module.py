from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.distributed import DistributedSampler


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
            pin_memory: bool = True,
            use_distributed_sampler: bool = False
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.processed_data_path = processed_data_path
        self.pin_memory = pin_memory
        self.use_distributed_sampler = use_distributed_sampler
        self.dataset: Optional[YuanDataset] = None

    def setup(self, stage: Optional[str] = None, has_labels: bool = False) -> None:
        if self.dataset is None:
            npz_data = np.load(self.processed_data_path)
            ids = (npz_data['id'])[0:(488282 + 100)].astype(np.int64)
            # print(f'max id = {torch.max(ids)}')
            # attention_masks = torch.from_numpy(npz_data['attention_mask'].astype(np.int64))
            if (has_labels):
                # labels = np.roll(ids, -1, axis=1)
                labels = ids
                dataset = TensorDataset(torch.from_numpy(ids), torch.from_numpy(labels))
            else:
                dataset = TensorDataset(torch.from_numpy(ids))
            train_dataset, val_dataset = random_split(dataset, [488282, 100])
            # train_dataset = train_dataset[0:488282]
            self.dataset = YuanDataset(train_dataset, val_dataset)

    def train_dataloader(self):
        if self.use_distributed_sampler:
            sampler = DistributedSampler(self.dataset.train_dataset)
        else:
            sampler = RandomSampler(self.dataset.train_dataset)
        return DataLoader(
            self.dataset.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            sampler=sampler
        )

    def val_dataloader(self):
        if self.use_distributed_sampler:
            sampler = DistributedSampler(self.dataset.val_dataset)
        else:
            sampler = RandomSampler(self.dataset.val_dataset)
        return DataLoader(
            self.dataset.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            sampler=sampler
        )


if __name__ == '__main__':
    dm = YuanDataModule(batch_size=32, num_workers=8)
    dm.setup()
    ids, masks = next(iter(dm.train_dataloader()))
    print(ids.shape)
