import datetime

from pytorch_lightning import loggers as pl_loggers


class NamedLogger(pl_loggers.TensorBoardLogger):
    def __init__(self, params: dict):
        super().__init__('logs', default_hp_metric=False)
        self.params = params

    @property
    def version(self):
        dt = datetime.datetime.now()
        return f'{dt.year}_{dt.month:02d}_{dt.day:02d}_{dt.hour:02d}_{dt.minute:02d}_' +\
               '_'.join([str(v) for v in self.params.values()])
