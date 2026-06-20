import json
import sys

from lanka_data.command.CommandRunner import CommandRunner


def main(command_str):

    print(json.dumps(CommandRunner.run(command_str), indent=2))


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]))
