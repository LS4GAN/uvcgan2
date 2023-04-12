from torch.utils.data import Dataset

class DatasetZipper(Dataset):

    def __init__(self, datasets, **kwargs):
        super().__init__(**kwargs)

        assert len(datasets) > 0, \
            "DatasetZipper does not know how to zip empty list of datasets"

        self._datasets = datasets
        self._len      = len(datasets[0])

        lengths = [ len(dset) for dset in datasets ]

        assert all(x == self._len for x in lengths), \
            f"DatasetZipper cannot zip datasets of unequal lengths: {lengths}"

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        return tuple(d[index] for d in self._datasets)

