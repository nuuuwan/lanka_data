import hashlib
import json
import os
import tempfile


class TriangleDataCacheMixin:
    CACHE_PREFIX = "triangle"

    @classmethod
    def _cache_path(cls, region_to_weight):
        key = str(sorted(region_to_weight.items()))
        hash_value = hashlib.md5(key.encode()).hexdigest()
        dir_temp = os.path.join(
            tempfile.gettempdir(), "lanka_data", "triangle_cache"
        )
        os.makedirs(dir_temp, exist_ok=True)
        return os.path.join(dir_temp, f"{cls.CACHE_PREFIX}_{hash_value}.json")

    @staticmethod
    def _load(path):
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _save(path, layout):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(layout, f)
