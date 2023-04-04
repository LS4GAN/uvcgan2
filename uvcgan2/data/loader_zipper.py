
class DataLoaderZipper:

    def __init__(self, loaders):
        self._loaders = loaders

    def __len__(self):
        return min(len(d) for d in self._loaders)

    def __iter__(self):
        return zip(*self._loaders)

