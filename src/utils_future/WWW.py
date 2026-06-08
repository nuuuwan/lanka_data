import csv
import hashlib
import os
import tempfile
from functools import cached_property

import requests

from utils_future.BinaryFile import BinaryFile
from utils_future.JSONFile import JSONFile
from utils_future.Log import Log

log = Log("WWW")


class WWW:
    T_TIMEOUT = 10
    DIR_WWW_CACHE = os.path.join(tempfile.gettempdir(), "lanka_data", "www")
    HASH_LEN = 16

    def __init__(self, url: str):
        self.url = url

    @cached_property
    def cache_file_base(self):
        h = hashlib.md5(self.url.encode("utf-8")).hexdigest()[: self.HASH_LEN]
        os.makedirs(self.DIR_WWW_CACHE, exist_ok=True)
        return os.path.join(self.DIR_WWW_CACHE, h)

    def read_json(self, do_use_cache=True):
        cache_json_file = JSONFile(self.cache_file_base + ".json")
        cache_hit = cache_json_file.exists() and do_use_cache

        if cache_hit:
            log.debug(f"💾 Getting {self.url} from cache - {cache_json_file}.")
            return cache_json_file.read()

        log.info(f"🌐 Getting {self.url} from web.")
        response = requests.get(self.url, timeout=self.T_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        cache_json_file.write(data)
        log.debug(f"Wrote {cache_json_file}")
        return data

    def _read_tsv_from_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            return list(reader)

    def read_tsv(self, do_use_cache=True):
        cache_tsv_file = BinaryFile(self.cache_file_base + ".tsv")
        cache_hit = cache_tsv_file.exists() and do_use_cache

        if cache_hit:
            log.debug(f"💾 Getting {self.url} from cache - {cache_tsv_file}.")
            return self._read_tsv_from_file(cache_tsv_file.path)

        log.info(f"🌐 Getting {self.url} from web.")
        response = requests.get(self.url, timeout=self.T_TIMEOUT)
        response.raise_for_status()
        cache_tsv_file.write(response.content)
        log.debug(f"Wrote {cache_tsv_file}")
        return self._read_tsv_from_file(cache_tsv_file.path)

    def download(self, do_use_cache=True) -> str:
        cache_file_path = self.cache_file_base + "." + self.url.split("/")[-1]
        cache_file = BinaryFile(cache_file_path)

        if cache_file.exists() and do_use_cache:
            log.debug(f"💾 Getting {self.url} from cache - {cache_file}.")
            return cache_file.path

        log.info(f"🌐 Getting {self.url} from web.")
        response = requests.get(self.url, timeout=self.T_TIMEOUT)
        response.raise_for_status()
        cache_file.write(response.content)
        log.debug(f"Wrote {cache_file}")
        return cache_file.path
