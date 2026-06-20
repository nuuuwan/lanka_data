import json
import sys

from lanka_data import Command, DatasetFactory, VisualFactory


def main(command_str):
    command = Command.from_str(command_str)

    datasets = DatasetFactory.list_from_command(command)
    visual = VisualFactory.from_commmand_and_datasets(command, datasets)
    print(json.dumps(visual.build(), indent=2))


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]))
