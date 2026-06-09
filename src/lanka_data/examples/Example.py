import os
import random

from lanka_data.db import Db
from utils_future import JSONFile, Log

log = Log("Example")


class Example:
    EXAMPLES_PATH = os.path.join("examples", "examples.json")
    DIR_EXAMPLES_OUTPUT = os.path.join("examples", "outputs")

    def __init__(self, cmd):
        self.cmd = cmd

    @classmethod
    def get_example_idx(cls):
        idx = JSONFile(cls.EXAMPLES_PATH).read()
        idx = {
            group_name: [Example(cmd) for cmd in cmds]
            for group_name, cmds in idx.items()
        }
        return dict(list(idx.items())[:2])

    @classmethod
    def get_cmd_list(cls):
        idx = cls.get_example_idx()
        cmd_list = []
        for examples in idx.values():
            cmd_list.extend([example.cmd for example in examples])
        cmd_list.sort()
        return cmd_list

    @classmethod
    def get_output_idx_hot(cls):
        cmd_list = cls.get_cmd_list()
        random.shuffle(cmd_list)
        idx = {}
        n_cmds = len(cmd_list)
        for i_cmd, cmd in enumerate(cmd_list, start=1):
            log.info(f"{i_cmd}/{n_cmds}) Building {cmd}.")
            output = Db(cmd).run(do_open_images=False, do_use_cache=True)
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
        os.makedirs(dir_outputs, exist_ok=True)

        idx = cls.get_output_idx_hot()
        for cmd, output in idx.items():
            output_dir = os.path.join(cls.DIR_EXAMPLES_OUTPUT, cmd)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "Output.json")
            output_file = JSONFile(output_path)
            output_file.write(output)
            log.info(f"Wrote {output_file}")

    @classmethod
    def get_output_idx(cls):
        cmd_list = cls.get_cmd_list()
        idx = {}
        for cmd in cmd_list:
            output_dir = os.path.join(
                "examples",
                "outputs",
                cmd,
            )
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "Output.json")
            output_file = JSONFile(output_path)
            assert (
                output_file.exists()
            ), f"Output file for cmd '{cmd}' does not exist at {output_file}"
            output = output_file.read()

            idx[cmd] = output
        return idx
