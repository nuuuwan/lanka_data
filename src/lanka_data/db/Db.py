import json
import logging
import os
import shutil
import tempfile
import time
from functools import cached_property

from lanka_data.where.Regions import Regions

log = logging.getLogger(__name__)


class Db:
    DIR_TEMP_DATA = os.path.join(tempfile.gettempdir(), "lanka_data")
    DIR_CACHE = os.path.join(DIR_TEMP_DATA, "cache")

    def __init__(self, cmd: str):
        self.cmd = cmd

    @cached_property
    def cache_file_base(self) -> str:
        os.makedirs(self.DIR_CACHE, exist_ok=True)

        tokens = self.cmd.lower().split("/")
        cmd_id = "-".join(tokens)
        return os.path.join(self.DIR_CACHE, cmd_id)

    @classmethod
    def cache_clear(cls):
        shutil.rmtree(cls.DIR_CACHE, ignore_errors=True)
        os.makedirs(cls.DIR_CACHE, exist_ok=True)
        log.warning("Cache cleared.")

    def _run(self):
        tokens = self.cmd.split("/")
        token0_tokens = tokens[0].split(":")

        regions = None
        if len(token0_tokens) == 1:
            region_id = token0_tokens[0]
            regions = Regions.from_region_id(region_id)
        elif len(token0_tokens) == 2:
            parent_region_id, region_type = token0_tokens
            regions = Regions.from_parent_region_id_and_region_type(
                region_type, parent_region_id
            )

        n_tokens = len(tokens)
        if n_tokens == 1:
            return regions.regions

        if n_tokens == 2:
            if tokens[1] == "JSON":
                return regions.regions
            if tokens[1] == "Map":
                return regions.draw_map(self.cache_file_base)

        raise ValueError(f"Invalid command: {self.cmd}")

    def run_unsafe(self, do_open_images, do_use_cache):
        t_start = time.perf_counter()
        cache_json_file = os.path.join(self.cache_file_base + ".json")

        cache_hit = os.path.exists(cache_json_file) and do_use_cache
        if cache_hit:
            results = json.load(open(cache_json_file, "r", encoding="utf-8"))
        else:
            results = self._run()
            with open(cache_json_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        if do_open_images:
            if "image_path" in results:
                image_path = results["image_path"]
                os.system(f"open {image_path}")

        query_time_ms = int((time.perf_counter() - t_start) * 1_000)
        return {
            "results": results,
            "query_time_ms": query_time_ms,
            "cache_hit": cache_hit,
        }

    def run(self, do_open_images, do_use_cache):
        try:
            return self.run_unsafe(do_open_images, do_use_cache)

        except Exception as e:
            log.error(f"Error running command '{self.cmd}': {e}")
            return {"error": str(e)}
