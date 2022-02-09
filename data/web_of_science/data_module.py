# import pytorch_lightning as pl


# class WebTextDataModule(pl.LightningDataModule):
#     """
#     This class is used to load the data from the WebText dataset.
#     """
#
#     def __init__(self, hparams):
#         """
#         Initializes the WebTextDataModule class.
#         """
#         super().__init__()
#         self.hparams = hparams
#
#     def setup(self, stage=None):
#         """
#         Loads the data from the WebText dataset.
#         """

if __name__ == '__main__':
    from datasets import load_dataset

    dataset = load_dataset("web_of_science", 'WOS5736')
    print(dataset)
