import os
import warnings

from lanka_data.examples.Example.ExampleOutputMixin import ExampleOutputMixin
from utils_future import JSONFile

warnings.warn(
    "Example class is deprecated. Use individual command execution or "
    "visual generation methods instead.",
    DeprecationWarning,
    stacklevel=2,
)


class Example(ExampleOutputMixin):
    EXAMPLES_PATH = os.path.join("examples", "examples.json")
    MAX_EXAMPLES = 1000

    def __init__(self, cmd):
        self.cmd = cmd

    @classmethod
    def get_group_to_cmd_list(cls):
        return JSONFile(cls.EXAMPLES_PATH).read()

    @classmethod
    def get_group_to_examples(cls):
        group_to_cmd_list = cls.get_group_to_cmd_list()
        group_to_examples = {}
        n_examples = 0
        for group_name, cmds in group_to_cmd_list.items():
            examples = []
            for cmd in cmds:
                examples.append(Example(cmd))
                n_examples += 1
                if n_examples >= cls.MAX_EXAMPLES:
                    break
            group_to_examples[group_name] = examples
            if n_examples >= cls.MAX_EXAMPLES:
                break
        return group_to_examples

    @classmethod
    def get_cmd_list(cls):
        cmd_list = []
        for examples in cls.get_group_to_examples().values():
            cmd_list.extend([example.cmd for example in examples])
        cmd_list.sort()
        return cmd_list
