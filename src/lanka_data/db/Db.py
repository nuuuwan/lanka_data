import hashlib
import json
import os
import shutil
import tempfile
import time
from functools import cached_property

from lanka_data.how import HowFactory
from lanka_data.what import WhatFactory
from lanka_data.where import Regions
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
        where = Regions.from_token(where_cmd)
        what = WhatFactory.from_what_and_when(what_cmd, when_cmd)
        how = HowFactory.from_how_cmd(how_cmd)
        return how.get_result(where, what, when_cmd)

    @staticmethod
    def _parse_cmd(cmd: str):
        tokens = cmd.split("/")
        n_tokens = len(tokens)

        if n_tokens != 4:
            raise ValueError(
                "Invalid command format:"
                + " expected <what>/<when>/<where>/<how>,"
                + f" got '{cmd}'"
            )

        what_cmd, when_cmd, where_cmd, how_cmd = tokens

        if when_cmd == "2012":
            if "-pre" not in where_cmd:
                tokens = where_cmd.split(":")
                if len(tokens) == 2 and tokens[1] == "dsd":
                    where_cmd = tokens[0] + "-pre2019:" + ":".join(tokens[1:])

        log.debug(f"{what_cmd=}, {when_cmd=}, {where_cmd=}, {how_cmd=}")
        return where_cmd, what_cmd, when_cmd, how_cmd

    def _run(self):
        if self.cmd == "*":
            return dict(
                what_to_whens=WhatFactory.get_what_to_whens(),
                where=["LK*", "EC-*", "LG-*"],
                how=["JSON", "Map"],
            )

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
