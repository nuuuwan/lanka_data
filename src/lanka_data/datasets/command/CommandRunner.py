import time

from lanka_data.api.command.Command import Command
from lanka_data.api.command.CommandCache import CommandCache
from lanka_data.api.data.DataSource import DataSource
from lanka_data.datasets.command.CommandHelp import CommandHelp
from lanka_data.datasets.dataset.DatasetFactory import DatasetFactory
from lanka_data.visual.VisualFactory import VisualFactory
from utils_future.timer import timer

HELP_SOURCE = DataSource(
    name="Lanka Data",
    url="https://github.com/nuuuwan/lanka_data/blob/main/README.md",
)


class CommandRunner:
    _cache = CommandCache()

    @timer
    @staticmethod
    def run(command_str: str, policy=None):
        t_start = time.perf_counter()
        result, sources, effective = CommandRunner._resolve(command_str)
        elapsed = time.perf_counter() - t_start
        return CommandRunner._payload(effective, result, sources, elapsed)

    @staticmethod
    def _resolve(command_str):
        if command_str == "Help":
            result, sources = CommandRunner._run_help()
            return result, sources, command_str
        command = Command.from_str(command_str)
        result, sources = CommandRunner._run_or_cache(command)
        return result, sources, command.cmd_id

    @staticmethod
    def _run_help():
        cached = CommandRunner._cache.get("Help")
        if cached is not None:
            return cached
        value = (CommandHelp.get_help_result(), [HELP_SOURCE])
        CommandRunner._cache.set("Help", value)
        return value

    @staticmethod
    def _run_or_cache(command):
        key = command.cmd_id
        cached = CommandRunner._cache.get(key)
        if cached is not None:
            return cached
        datasets = DatasetFactory.list_from_command(command)
        visual = VisualFactory.from_command_and_datasets(command, datasets)
        value = (visual.build(), visual.get_sources())
        CommandRunner._cache.set(key, value)
        return value

    @staticmethod
    def _payload(effective, result, sources, elapsed):
        return dict(
            command_str=effective,
            result=result,
            sources=[source.__dict__ for source in sources],
            query_time_ms=int(elapsed * 1000),
        )
