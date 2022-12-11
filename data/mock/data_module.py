from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.distributed import DistributedSampler

class MockDataModule(pl.LightningDataModule):
    """
    Mock data module for testing. The data in this module is generated randomly.
    """
    def __init__(
            self,
            vocab_size: int,
            seq_length: int,
            batch_size: int = 32,
            mock_data_size: int = 500000,
            num_workers: int = 8,
            use_distributed_sampler: bool = False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.data_size = mock_data_size
        self.num_workers = num_workers
        self.use_distributed_sampler = use_distributed_sampler
        self.dataset = None

    def setup(self, stage: Optional[str] = None, has_labels: bool = False) -> None:
        ids = torch.randint(self.vocab_size, (self.data_size, self.seq_length))
        if (has_labels):
            labels = ids
            self.dataset = TensorDataset(ids, labels)
        else:
            self.dataset = TensorDataset(ids)

    def train_dataloader(self):
        if self.use_distributed_sampler:
            sampler = DistributedSampler(self.dataset)
        else:
            sampler = RandomSampler(self.dataset)
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler)

    def val_dataloader(self):
        if self.use_distributed_sampler:
            sampler = DistributedSampler(self.dataset)
        else:
            sampler = RandomSampler(self.dataset)
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler)


if __name__ == '__main__':
    dm = MockDataModule(vocab_size=1000, seq_length=3072)
    dm.setup()
    [d] = next(iter(dm.train_dataloader()))
    print(d.shape)
