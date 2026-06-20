import os

from lanka_data.command import Command
from lanka_data.dataset.DatasetFactory import DatasetFactory
from lanka_data.visual.VisualFactory import VisualFactory
from utils_future import JSONFile, Log

log = Log("Example")


class Example:
    EXAMPLES_PATH = os.path.join("examples", "examples.json")
    DIR_EXAMPLES_OUTPUT = os.path.join("examples", "outputs")
    MAX_EXAMPLES = 1000

    def __init__(self, cmd):
        self.cmd = cmd

    @classmethod
    def get_group_to_cmd_list(cls):
        group_to_cmd_list = JSONFile(cls.EXAMPLES_PATH).read()
        return group_to_cmd_list

    @classmethod
    def get_group_to_examples(cls):
        group_to_cmd_list = cls.get_group_to_cmd_list()
        group_to_examples = {
            group_name: [Example(cmd) for cmd in cmds]
            for group_name, cmds in group_to_cmd_list.items()
        }
        group_to_examples = {}
        n_examples = 0
        for group_name, cmds in group_to_cmd_list.items():
            examples = []
            for cmd in cmds:
                example = Example(cmd)
                examples.append(example)
                n_examples += 1
                if n_examples >= cls.MAX_EXAMPLES:
                    break
            group_to_examples[group_name] = examples
            if n_examples >= cls.MAX_EXAMPLES:
                break
        return group_to_examples

    @classmethod
    def get_cmd_list(cls):
        idx = cls.get_group_to_examples()
        cmd_list = []
        for examples in idx.values():
            cmd_list.extend([example.cmd for example in examples])
        cmd_list.sort()
        return cmd_list

    @classmethod
    def get_output_hot(cls, command_str):
        command = Command.from_str(command_str)
        datasets = DatasetFactory.list_from_command(command)
        visual = VisualFactory.from_commmand_and_datasets(command, datasets)
        output = visual.build()
        if "result" not in output:
            raise ValueError(
                f"Output for cmd '{command_str}' does not contain 'result'"
            )

        # Hack to prevent git diffs
        output["cache_hit"] = True
        output["query_time_ms"] = 0
        return output

    @classmethod
    def get_output(cls, cmd):
        output_dir = os.path.join(
            "examples",
            "outputs",
            cmd,
        )
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
        for cmd in cmd_list:
            output = cls.get_output(cmd)
            cmd_to_output[cmd] = output
        return cmd_to_output

    @classmethod
    def build(cls):
        cls.get_cmd_to_output()
