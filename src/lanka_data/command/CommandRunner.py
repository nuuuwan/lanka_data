from lanka_data.command.Command import Command
from lanka_data.dataset.DatasetFactory import DatasetFactory
from lanka_data.visual.VisualFactory import VisualFactory
from utils_future import timer


class CommandRunner:
    @timer
    @staticmethod
    def run(command_str: str):
        command = Command.from_str(command_str)
        datasets = DatasetFactory.list_from_command(command)
        visual = VisualFactory.from_commmand_and_datasets(command, datasets)

        result = visual.build()
        source_list = visual.get_source_list()

        return dict(
            command_str=command_str,
            result=result,
            source_list=source_list,
        )
