import hashlib
import os
import shutil

from lanka_data.db import Db
from utils_future import JSONFile, Log

log = Log("Example")


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
    def get_cmd_list(cls):
        return JSONFile(cls.EXAMPLES_PATH).read()

    @classmethod
    def get_output_idx_hot(cls):
        cmd_list = cls.get_cmd_list()
        idx = {}
        for cmd in cmd_list:
            output = Db(cmd).run(do_open_images=False, do_use_cache=False)
            if "result" not in output:
                raise ValueError(
                    f"Output for cmd '{cmd}' does not contain 'result'"
                )
            output["query_time_ms"] = 0
            idx[cmd] = output
        return idx

    @classmethod
    def build(cls):
        dir_outputs = os.path.join("examples", "outputs")
        if os.path.exists(dir_outputs):
            shutil.rmtree(dir_outputs)
        os.makedirs(dir_outputs)

        idx = cls.get_output_idx_hot()
        for cmd, output in idx.items():
            output_path = os.path.join(
                dir_outputs, f"{Example.cmd_to_hash(cmd)}.json"
            )
            output_file = JSONFile(output_path)
            output_file.write(output)
            log.info(f"Wrote {output_file}")

    @classmethod
    def get_output_idx(cls):
        cmd_list = JSONFile(cls.EXAMPLES_PATH).read()
        idx = {}
        for cmd in cmd_list:
            output_path = os.path.join(
                "examples", "outputs", f"{Example.cmd_to_hash(cmd)}.json"
            )
            output_file = JSONFile(output_path)
            assert output_file.exists()
            output = output_file.read()

            idx[cmd] = output
        return idx
