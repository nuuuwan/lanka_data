import json
import os
import shutil
import tempfile
import time

from lanka_data.api.how import HowFactory
from lanka_data.api.what import WhatFactory
from lanka_data.api.where import Regions
from lanka_data.command.CommandHelp import CommandHelp
from utils_future import Log

log = Log("CommandRunnerMixin")


class CommandRunnerMixin:
    DIR_TEMP_DATA = os.path.join(tempfile.gettempdir(), "lanka_data")
    DIR_OUTPUT = os.path.join(DIR_TEMP_DATA, "output")

    def get_where(self):
        return Regions.from_command(self)

    def get_what(self):
        return WhatFactory.from_command(self)

    def get_when(self):
        return self.when_cmd

    def get_how(self):
        return HowFactory.from_command(self)

    def get_result(self):
        return self.get_how().get_result(self)

    def _run(self):
        if self.what_cmd == "Help":
            return CommandHelp.get_help_result()
        return self.get_result()

    def run_unsafe(self, do_open_images, do_use_cache):
        t_start = time.perf_counter()
        dir_cache_base = os.path.join(self.DIR_OUTPUT, self.cmd_id)
        os.makedirs(dir_cache_base, exist_ok=True)
        cache_json_file = os.path.join(dir_cache_base, "Output.json")

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
            log.error(f"Error running command '{self}': {e}")
            return {"error": str(e)}

    @classmethod
    def cache_clear(cls):
        shutil.rmtree(cls.DIR_OUTPUT, ignore_errors=True)
        os.makedirs(cls.DIR_OUTPUT, exist_ok=True)
        log.warning("Cache cleared.")
