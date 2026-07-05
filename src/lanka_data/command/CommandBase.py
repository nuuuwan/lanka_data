from dataclasses import dataclass

from lanka_data.command.InvalidCommandError import InvalidCommandError
from lanka_data.command.fields import How, What, When, Where


@dataclass(init=False)
class CommandBase:
    what: What
    when: When
    where: Where
    how: How

    def __init__(self, what=None, when=None, where=None, how=None, **cmds):
        self.what = self._build_field(What, what, cmds.pop("what_cmd", None))
        self.when = self._build_field(When, when, cmds.pop("when_cmd", None))
        self.where = self._build_field(
            Where, where, cmds.pop("where_cmd", None)
        )
        self.how = self._build_field(How, how, cmds.pop("how_cmd", None))
        if cmds:
            raise TypeError(f"Unknown command fields: {', '.join(cmds)}")
        self._validate_coupling()

    @staticmethod
    def _build_field(field_cls, value, value_cmd):
        value = value if value_cmd is None else value_cmd
        if isinstance(value, field_cls):
            return value
        return field_cls("" if value is None else value)

    @property
    def cmd_id(self):
        return "/".join(
            [self.what_cmd, self.when_cmd, self.where_cmd, self.how_cmd]
        )

    @property
    def what_cmd(self):
        return self.what.value

    @property
    def when_cmd(self):
        return self.when.value

    @property
    def where_cmd(self):
        return self.where.value

    @property
    def how_cmd(self):
        return self.how.value

    def _validate_coupling(self):
        if self.how.needs_interval and not self.when.is_interval:
            raise InvalidCommandError(
                f"{self.how.value} requires an interval when",
                self.cmd_id,
            )

    def __str__(self):
        return f"Command({self.cmd_id})"

    def __repr__(self):
        return self.__str__()
