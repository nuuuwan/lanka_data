from dataclasses import dataclass, replace

from lanka_data.api.command.Command import Command
from lanka_data.api.fields import How, What, When, Where


@dataclass(frozen=True)
class WorkingCommand:
    what: str
    when: str
    where: str
    how: str

    @classmethod
    def from_command(cls, command):
        return cls(
            what=command.what_cmd,
            when=command.when_cmd,
            where=command.where_cmd,
            how=command.how_cmd,
        )

    @property
    def what_field(self):
        return What(self.what)

    @property
    def when_field(self):
        return When(self.when)

    @property
    def where_field(self):
        return Where(self.where)

    @property
    def how_field(self):
        return How(self.how)

    def with_when(self, value):
        return replace(self, when=value)

    def with_where(self, value):
        return replace(self, where=value)

    def with_how(self, value):
        return replace(self, how=value)

    def to_command(self):
        return Command(
            what_cmd=self.what,
            when_cmd=self.when,
            where_cmd=self.where,
            how_cmd=self.how,
        )
