import json
import hashlib

class ConfigBase:

    __slots__ = []

    def to_dict(self):
        return { x : getattr(self, x) for x in self.__slots__ }

    def to_json(self, **kwargs):
        return json.dumps(self, default = lambda x : x.to_dict(), **kwargs)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def get_hash(self):
        s = self.to_json(sort_keys = True)

        md5 = hashlib.md5()
        md5.update(s.encode())

        return md5.hexdigest()

