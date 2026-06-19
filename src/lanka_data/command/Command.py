from lanka_data.command.CommandBase import CommandBase
from lanka_data.command.CommandLoaderMixin import CommandLoaderMixin
from lanka_data.command.CommandRunnerMixin import CommandRunnerMixin
from utils_future import Log

log = Log("Command")


class Command(CommandBase, CommandLoaderMixin, CommandRunnerMixin):

    def copy(
        self,
        what_cmd=None,
        when_cmd=None,
        where_cmd=None,
        how_cmd=None,
    ):
        return Command(
            what_cmd=what_cmd or self.what_cmd,
            when_cmd=when_cmd or self.when_cmd,
            where_cmd=where_cmd or self.where_cmd,
            how_cmd=how_cmd or self.how_cmd,
        )
