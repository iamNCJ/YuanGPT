from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader


class MockDataModule(pl.LightningDataModule):
    def __init__(
            self,
            vocab_size: int,
            seq_length: int,
            batch_size: int = 32,
            mock_data_size: int = 100,
            num_workers: int = 8
    ):
        super().__init__()
        self.voca_size = vocab_size
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.data_size = mock_data_size
        self.num_workers = num_workers
        self.dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = torch.randint(self.voca_size, (self.data_size, self.seq_length))

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


if __name__ == '__main__':
    dm = MockDataModule(vocab_size=1000, seq_length=3072)
    dm.setup()
    d = next(iter(dm.train_dataloader()))
    print(d.shape)
