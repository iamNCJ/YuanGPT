
from typing import Optional, Union

class DataModule:
	def setup(self, stage: Optional[str] = None, has_labels: bool = False) -> None:
		raise NotImplementedError

	def train_dataloader(self):
		raise NotImplementedError

	def val_dataloader(self):
		raise NotImplementedError

