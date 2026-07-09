import json
import os
import sys

from lanka_data.datasets.command.CommandRunner import CommandRunner
from utils_future import JSONFile


def _write_output(output):
    cmd_id = output.get("command_str")
    if not cmd_id:
        return
    output_dir = os.path.join("_output", cmd_id)
    os.makedirs(output_dir, exist_ok=True)
    JSONFile(os.path.join(output_dir, "Output.json")).write(output)


def _report_corrections(output):
    corrections = output.get("corrections") or []
    if not corrections:
        return
    original = output.get("original_command_str")
    print(
        f"[corrected] {original} -> {output.get('command_str')}",
        file=sys.stderr,
    )
    for correction in corrections:
        print(
            f"  - [{correction['severity']}] {correction['reason']}",
            file=sys.stderr,
        )


def main(command_str):
    output = CommandRunner.run(command_str)
    _report_corrections(output)
    _write_output(output)
    result = output.get("result")
    if result and "image_path" in result:
        image_path = result["image_path"]
        os.system(f"open {image_path}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]))
