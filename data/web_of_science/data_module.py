from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split

from ..datamodule import DataModule

@dataclass
class WOSDataset:

    train_dataset: Union[TensorDataset, Subset]
    val_dataset: Union[TensorDataset, Subset]


class WOSDataModule(DataModule):
    """
    Data module for Web of Science dataset.
    """
    def __init__(
            self,
            batch_size: int = 32,
            num_workers: int = 8,
            processed_data_path: str = './processed_data.npz',
            drop_attention_mask: bool = True,
            pin_memory: bool = False
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.processed_data_path = processed_data_path
        self.pin_memory = pin_memory
        self.drop_attention_mask = drop_attention_mask
        self.dataset: WOSDataset

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            npz_data = np.load(self.processed_data_path)
            ids = torch.from_numpy(npz_data['id']).type('long')
            if not self.drop_attention_mask:
                attention_masks = torch.from_numpy(npz_data['attention_mask']).type(torch.long)
                dataset = TensorDataset(ids, attention_masks)
            else:
                dataset = TensorDataset(ids)
            train_dataset, val_dataset = random_split(
                dataset,
                [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))]
            )
            self.dataset = WOSDataset(train_dataset, val_dataset)

    def train_dataloader(self):
        return DataLoader(
            self.dataset.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )


if __name__ == '__main__':
    dm = WOSDataModule(batch_size=32, num_workers=8)
    dm.setup()
    input_ids, attention_mask = next(iter(dm.train_dataloader()))
    print(input_ids.shape)
    print(attention_mask.shape)
