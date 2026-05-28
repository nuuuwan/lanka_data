import hashlib
import json
import os
import shutil
import tempfile
import time
from functools import cached_property

from lanka_data.what import WhatFactory
from lanka_data.where import Regions
from lanka_data.where.RegionsMapUtils import RegionsMapUtils
from utils_future import Log

log = Log("Db")


class Db:
    DIR_TEMP_DATA = os.path.join(tempfile.gettempdir(), "lanka_data")
    DIR_CACHE = os.path.join(DIR_TEMP_DATA, "cache")

    def __init__(self, cmd: str):
        self.cmd = cmd

    @cached_property
    def cache_file_base(self) -> str:
        os.makedirs(self.DIR_CACHE, exist_ok=True)

        cmd_id = self.cmd.lower()
        h = hashlib.md5(cmd_id.encode("utf-8")).hexdigest()[:8]
        return os.path.join(self.DIR_CACHE, h)

    @classmethod
    def cache_clear(cls):
        shutil.rmtree(cls.DIR_CACHE, ignore_errors=True)
        os.makedirs(cls.DIR_CACHE, exist_ok=True)
        log.warning("Cache cleared.")

    def _run_normalized(
        self, where_cmd: str, what_cmd: str, when_cmd: str, how_cmd: str
    ):
        regions = Regions.from_token(where_cmd)
        what = WhatFactory.from_what_and_when(what_cmd, when_cmd)

        results = what.get_result(regions)

        if how_cmd == "JSON":
            return results

        if how_cmd == "Map":
            return RegionsMapUtils.draw_map(
                regions.regions, self.cache_file_base
            )

        raise ValueError(f"Unknown how_cmd: {how_cmd}")

    @staticmethod
    def _parse_cmd(cmd: str):
        tokens = cmd.split("/")
        n_tokens = len(tokens)

        if n_tokens == 2:
            tokens = [tokens[0], "Basic", "2024", tokens[1]]
            n_tokens = 4

        where_cmd = "LK" if n_tokens < 1 else tokens[0]
        what_cmd = "Basic" if n_tokens < 2 else tokens[1]
        when_cmd = "2024" if n_tokens < 3 else tokens[2]
        how_cmd = "JSON" if n_tokens < 4 else tokens[3]
        log.debug(f"{where_cmd=}, {what_cmd=}, {when_cmd=}, {how_cmd=}")
        return where_cmd, what_cmd, when_cmd, how_cmd

    def _run(self):
        where_cmd, what_cmd, when_cmd, how_cmd = self._parse_cmd(self.cmd)
        return self._run_normalized(where_cmd, what_cmd, when_cmd, how_cmd)

    def run_unsafe(self, do_open_images, do_use_cache):
        t_start = time.perf_counter()
        cache_json_file = os.path.join(self.cache_file_base + ".json")

        cache_hit = os.path.exists(cache_json_file) and do_use_cache
        if cache_hit:
            result = json.load(open(cache_json_file, "r", encoding="utf-8"))
        else:
            result = self._run()
            with open(cache_json_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        if do_open_images:
            if "image_path" in result:
                image_path = result["image_path"]
                os.system(f"open {image_path}")

        query_time_ms = int((time.perf_counter() - t_start) * 1_000)
        return {
            "result": result,
            "query_time_ms": query_time_ms,
            "cache_hit": cache_hit,
        }

    def run(self, do_open_images, do_use_cache):
        try:
            return self.run_unsafe(do_open_images, do_use_cache)

        except Exception as e:
            log.error(f"Error running command '{self.cmd}': {e}")
            return {"error": str(e)}
