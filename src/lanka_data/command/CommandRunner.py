import time

from lanka_data.command.Command import Command
from lanka_data.dataset.DatasetFactory import DatasetFactory
from lanka_data.visual.VisualFactory import VisualFactory


class CommandRunner:
    @staticmethod
    def run(command_str: str):
        t_start = time.perf_counter()

        command = Command.from_str(command_str)
        datasets = DatasetFactory.list_from_command(command)
        visual = VisualFactory.from_commmand_and_datasets(command, datasets)
        result = visual.build()
        source_list = visual.get_source_list()

        time_elapsed = time.perf_counter() - t_start
        return dict(
            command_str=command_str,
            result=result,
            source_list=source_list,
            time_elapsed_ms=int(time_elapsed * 1000),
        )
