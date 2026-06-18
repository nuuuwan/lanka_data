from dataclasses import dataclass
from functools import cached_property


@dataclass
class CommandBase:
    where_cmd: str
    what_cmd: str
    when_cmd: str
    how_cmd: str

    @cached_property
    def cmd_id(self):
        return "/".join(
            [self.where_cmd, self.what_cmd, self.when_cmd, self.how_cmd]
        )

    def unpack(self):
        return self.where_cmd, self.what_cmd, self.when_cmd, self.how_cmd
