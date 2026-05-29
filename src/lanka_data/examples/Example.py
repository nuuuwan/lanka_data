import hashlib
import os

from lanka_data.db import Db
from utils_future import JSONFile


class Example:
    EXAMPLES_PATH = os.path.join("examples", "examples.json")

    def __init__(self, cmd):
        self.cmd = cmd

    @classmethod
    def list(cls):
        cmd_list = JSONFile(cls.EXAMPLES_PATH).read()
        return [Example(cmd) for cmd in cmd_list]

    @staticmethod
    def cmd_to_hash(cmd):
        return hashlib.md5(cmd.encode()).hexdigest()[:8]

    @classmethod
    def get_output_idx(cls):
        os.makedirs(os.path.join("examples", "outputs"), exist_ok=True)
        cmd_list = JSONFile(cls.EXAMPLES_PATH).read()
        idx = {}
        for cmd in cmd_list:
            output_path = os.path.join(
                "examples", "outputs", f"{Example.cmd_to_hash(cmd)}.json"
            )
            output_file = JSONFile(output_path)
            if not os.path.exists(output_path):
                output = Db(cmd).run(do_open_images=False, do_use_cache=False)
                output["query_time_ms"] = 0
                output_file.write(output)
            else:
                output = output_file.read()

            idx[cmd] = output
        return idx
