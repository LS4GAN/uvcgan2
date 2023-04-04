import os
import pandas as pd

HISTORY_NAME = 'history.csv'

class TrainingHistory:

    def __init__(self, savedir):
        self._history = None
        self._savedir = savedir

    def end_epoch(self, epoch, metrics):
        values = metrics.values
        values['epoch'] = epoch
        values['time']  = pd.Timestamp.utcnow()

        if self._history is None:
            self._history = pd.DataFrame([ values, ])
        else:
            self._history = self._history.append([ values, ])

        self.save()

    def save(self):
        self._history.to_csv(
            os.path.join(self._savedir, HISTORY_NAME), index = False
        )

    def load(self):
        path = os.path.join(self._savedir, HISTORY_NAME)

        if os.path.exists(path):
            self._history = pd.read_csv(path, parse_dates = [ 'time', ])

    @property
    def history(self):
        return self._history


