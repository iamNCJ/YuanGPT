import datetime

from pytorch_lightning import loggers as pl_loggers


class NamedLogger(pl_loggers.TensorBoardLogger):
    def __init__(self, params: dict, log_name: str = None):
        super().__init__('tensorboard', default_hp_metric=False)
        self.params = params
        self.log_name = log_name

    @property
    def version(self):
        dt = datetime.datetime.now()
        if self.log_name == None:
            return f'{dt.year}_{dt.month:02d}_{dt.day:02d}_{dt.hour:02d}_{dt.minute:02d}_' +\
               '_'.join([str(v) for v in self.params.values()])
        else:
            return self.log_name
