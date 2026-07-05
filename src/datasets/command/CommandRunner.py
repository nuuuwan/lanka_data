import time

from datasets.command.Command import Command
from api.command.CommandCache import CommandCache
from datasets.command.CommandHelp import CommandHelp
from api.data.DataSource import DataSource
from datasets.dataset.DatasetFactory import DatasetFactory
from datasets.visual.VisualFactory import VisualFactory
from api.utils_future.timer import timer


class CommandRunner:
    _cache = CommandCache()

    @timer
    @staticmethod
    def run(command_str: str):
        t_start = time.perf_counter()

        cached = CommandRunner._cache.get(command_str)
        if cached is not None:
            result, sources = cached
        elif command_str == "Help":
            result = CommandHelp.get_help_result()
            sources = [
                DataSource(
                    name="Lanka Data",
                    url=(
                        "https://github.com/nuuuwan/lanka_data"
                        "/blob/main/README.md"
                    ),
                )
            ]
            CommandRunner._cache.set(command_str, (result, sources))
        else:
            command = Command.from_str(command_str)
            datasets = DatasetFactory.list_from_command(command)
            visual = VisualFactory.from_command_and_datasets(
                command, datasets
            )
            result = visual.build()
            sources = visual.get_sources()
            CommandRunner._cache.set(command_str, (result, sources))

        time_elapsed = time.perf_counter() - t_start
        return dict(
            command_str=command_str,
            result=result,
            sources=[source.__dict__ for source in sources],
            query_time_ms=int(time_elapsed * 1000),
        )
