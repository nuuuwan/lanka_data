import os
import warnings

from lanka_data.datasets.command.CommandRunner import CommandRunner
from utils_future import JSONFile, Log

log = Log("Example")

warnings.warn(
    "ExampleOutputMixin class is deprecated. Use individual command "
    "execution or visual generation methods instead.",
    DeprecationWarning,
    stacklevel=2,
)


class ExampleOutputMixin:
    @classmethod
    def get_output_hot(cls, command_str):
        output = CommandRunner.run(command_str)
        if "result" not in output:
            raise ValueError(
                f"Output for cmd '{command_str}' does not contain 'result'"
            )
        output["query_time_ms"] = 0
        return output

    @classmethod
    def get_output(cls, cmd):
        output_dir = os.path.join("_output", cmd)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "Output.json")
        output_file = JSONFile(output_path)
        if output_file.exists():
            output = output_file.read()
            log.debug(f"Read {output_file}")
        else:
            output = cls.get_output_hot(cmd)
            output_file.write(output)
            log.info(f"Wrote {output_file}")
        return output

    @classmethod
    def get_cmd_to_output(cls):
        cmd_list = cls.get_cmd_list()
        cmd_to_output = {}
        n_cmds = len(cmd_list)
        for i_cmd, cmd in enumerate(cmd_list, start=1):
            log.debug("-" * 40)
            log.debug(f"{i_cmd:02d}/{n_cmds:02d}) {cmd}")
            log.debug("-" * 40)
            cmd_to_output[cmd] = cls.get_output(cmd)
        return cmd_to_output

    @classmethod
    def build(cls):
        cls.get_cmd_to_output()
