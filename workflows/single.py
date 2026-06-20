import json
import os
import sys

from lanka_data.command.CommandRunner import CommandRunner


def main(command_str):
    output = CommandRunner.run(command_str)
    image_path = output.get("result", {}).get("image_path")
    if image_path:
        os.system(f"open {image_path}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]))
