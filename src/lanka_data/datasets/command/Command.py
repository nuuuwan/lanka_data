from lanka_data.datasets.command.CommandBase import CommandBase
from lanka_data.api.command.CommandLoaderMixin import CommandLoaderMixin
from utils_future import Log

log = Log("Command")


class Command(
    CommandBase,
    CommandLoaderMixin,
):

    def copy(
        self,
        what_cmd=None,
        when_cmd=None,
        where_cmd=None,
        how_cmd=None,
    ):
        return Command(
            what_cmd=self.what_cmd if what_cmd is None else what_cmd,
            when_cmd=self.when_cmd if when_cmd is None else when_cmd,
            where_cmd=self.where_cmd if where_cmd is None else where_cmd,
            how_cmd=self.how_cmd if how_cmd is None else how_cmd,
        )
