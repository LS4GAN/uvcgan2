import copy

class LossMetrics:

    def __init__(self):
        self._values = None
        self._n      = 0

    @property
    def values(self):
        if self._values is None:
            return None

        return { k : v / self._n for (k,v) in self._values.items() }

    def update(self, values):
        if self._values is None:
            self._values = copy.deepcopy(values)
        else:
            for k,v in values.items():
                self._values[k] += v

        self._n += 1

