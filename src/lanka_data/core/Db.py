import logging
import os
import tempfile
import time
from functools import cached_property

from lanka_data.core.Regions import Regions

log = logging.getLogger(__name__)


class Db:
    def __init__(self, cmd: str):
        self.cmd = cmd

    @cached_property
    def temp_file_path_base(self) -> str:
        tokens = self.cmd.lower().split("/")
        cmd_id = "-".join(tokens)
        temp_dir = os.path.join(tempfile.gettempdir(), "lanka_data")
        os.makedirs(temp_dir, exist_ok=True)
        return os.path.join(temp_dir, cmd_id)

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
                return regions.draw_map(self.temp_file_path_base)

        raise ValueError(f"Invalid command: {self.cmd}")

    def run_unsafe(self, do_open_images):
        t_start = time.perf_counter()
        results = self._run()
        query_time_ms = int((time.perf_counter() - t_start) * 1_000)

        if do_open_images:
            if "image_path" in results:
                image_path = results["image_path"]
                os.system(f"open {image_path}")

        return {"results": results, "query_time_ms": query_time_ms}

    def run(self, do_open_images):
        try:
            return self.run_unsafe(do_open_images)

        except Exception as e:
            log.error(f"Error running command '{self.cmd}': {e}")
            return {"error": str(e)}
