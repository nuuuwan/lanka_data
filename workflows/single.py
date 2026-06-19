import sys

from lanka_data import Command, Dataset, Visual


def main(command_str):
    command = Command.from_str(command_str)
    datasets = Dataset.list_from_command(command)
    visual = Visual.from_commmand_and_datasets(command, datasets)
    visual.build()


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]))
