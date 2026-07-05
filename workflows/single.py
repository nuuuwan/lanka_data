import json
import os
import sys

from datasets.command.CommandRunner import CommandRunner


def main(command_str):
    output = CommandRunner.run(command_str)
    result = output.get("result")
    if result and "image_path" in result:
        image_path = result["image_path"]
        os.system(f"open {image_path}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]))
