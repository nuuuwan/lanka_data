from dataclasses import dataclass
from functools import cached_property


@dataclass
class CommandBase:
    what_cmd: str
    when_cmd: str
    where_cmd: str
    how_cmd: str

    @cached_property
    def cmd_id(self):
        return "/".join(
            [self.what_cmd, self.when_cmd, self.where_cmd, self.how_cmd]
        )

    def unpack(self):
        return self.what_cmd, self.when_cmd, self.where_cmd, self.how_cmd

    def __str__(self):
        return f"Command({self.cmd_id})"
