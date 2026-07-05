from dataclasses import dataclass
from functools import cached_property

from lanka_data.api.command.CommandBase.CommandBaseValidationMixin import (
    CommandBaseValidationMixin,
)
from lanka_data.api.command.fields import How, What, When, Where


@dataclass(init=False)
class CommandBase(CommandBaseValidationMixin):
    what: What
    when: When
    where: Where
    how: How

    def __init__(self, what=None, when=None, where=None, how=None, **cmds):
        self._validate_cmd_names(cmds)
        self.what = self._build_field(What, what, cmds.pop("what_cmd", None))
        self.when = self._build_field(When, when, cmds.pop("when_cmd", None))
        self.where = self._build_field(
            Where, where, cmds.pop("where_cmd", None)
        )
        self.how = self._build_field(How, how, cmds.pop("how_cmd", None))
        self._validate_parts()
        self._validate_coupling()

    @classmethod
    def _build_field(cls, field_cls, value, value_cmd):
        field_value = cls._resolve_field_value(value, value_cmd)
        if isinstance(field_value, field_cls):
            return field_value
        return field_cls(field_value)

    @staticmethod
    def _resolve_field_value(value, value_cmd):
        if value_cmd is not None:
            return value_cmd
        if value is None:
            return ""
        return value

    @cached_property
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

    def __str__(self):
        return f"Command({self.cmd_id})"

    def __repr__(self):
        return self.__str__()
