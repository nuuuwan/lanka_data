from dataclasses import dataclass
from functools import cached_property

from lanka_data.command.InvalidCommandError import InvalidCommandError
from lanka_data.command.fields import How, What, When, Where


@dataclass
class CommandBase:
    what_cmd: str
    when_cmd: str
    where_cmd: str
    how_cmd: str

    def __post_init__(self):
        self._validate_parts()
        self._validate_coupling()

    @cached_property
    def cmd_id(self):
        return "/".join(
            [self.what_cmd, self.when_cmd, self.where_cmd, self.how_cmd]
        )

    @cached_property
    def what(self):
        return What(self.what_cmd)

    @cached_property
    def when(self):
        return When(self.when_cmd)

    @cached_property
    def where(self):
        return Where(self.where_cmd)

    @cached_property
    def how(self):
        return How(self.how_cmd)

    def _validate_parts(self):
        self.what
        self.when
        self.where
        self.how

    def _validate_coupling(self):
        if self.how.needs_interval and not self.when.is_interval:
            how_name = self.how.modifier or self.how.base
            raise InvalidCommandError(
                f"{how_name} requires an interval when",
                self.cmd_id,
            )

    def __str__(self):
        return f"Command({self.cmd_id})"

    def __repr__(self):
        return self.__str__()
