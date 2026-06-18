from lanka_data.command.CommandBase import CommandBase
from lanka_data.command.CommandLoaderMixin import CommandLoaderMixin
from lanka_data.command.CommandRunnerMixin import CommandRunnerMixin
from utils_future import Log

log = Log("Command")


class Command(CommandBase, CommandLoaderMixin, CommandRunnerMixin):
    pass
