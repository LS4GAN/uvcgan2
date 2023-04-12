from collections.abc import Mapping

class NamedDict(Mapping):
    # pylint: disable=too-many-instance-attributes

    _fields = None

    def __init__(self, *args, **kwargs):
        self._fields = {}

        for arg in args:
            self._fields[arg] = None

        self._fields.update(**kwargs)

    def __contains__(self, key):
        return (key in self._fields)

    def __getitem__(self, key):
        return self._fields[key]

    def __setitem__(self, key, value):
        self._fields[key] = value

    def __getattr__(self, key):
        return self._fields[key]

    def __setattr__(self, key, value):
        if (self._fields is not None) and (key in self._fields):
            self._fields[key] = value
        else:
            super().__setattr__(key, value)

    def __iter__(self):
        return iter(self._fields)

    def __len__(self):
        return len(self._fields)

    def items(self):
        return self._fields.items()

    def keys(self):
        return self._fields.keys()

    def values(self):
        return self._fields.values()

