import os
from collections import OrderedDict


class CommandCache:
    def __init__(self, max_size=128):
        self.max_size = max_size
        self.data = OrderedDict()

    def get(self, key):
        cached = self.data.get(key)
        if cached is None:
            return None
        if not self.is_valid(cached):
            del self.data[key]
            return None
        self.data.move_to_end(key)
        return cached

    def set(self, key, value):
        self.data[key] = value
        self.data.move_to_end(key)
        if len(self.data) > self.max_size:
            self.data.popitem(last=False)

    @staticmethod
    def is_valid(cached):
        result, _ = cached
        if isinstance(result, dict) and "image_path" in result:
            return os.path.exists(result["image_path"])
        return True
