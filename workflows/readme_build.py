import json
import logging
import os
import shutil

from custom_logging import setup_logging

from lanka_data import Db
from utils_future import File, JSONFile

log = logging.getLogger(__name__)


class ReadMe:
    PATH = "README.md"
    MAX_LINES_IN_OUTPUT = 40

    def run_tests(self):
        test_db_data_file = os.path.join("tests", "test_db.data.json")
        test_db_data = JSONFile(test_db_data_file).read()

        idx = {}
        for command in test_db_data.keys():
            output = Db(command).run(
                do_open_images=False,
                do_use_cache=True,
            )
            idx[command] = output
        return idx

    def get_lines_for_sources(self, test_idx):

        lines = [
            "## Data Sources",
            "",
        ]
        source_idx = {}
        for output in test_idx.values():
            if "result" in output:
                result = output["result"]
                source = result["source"]
                source_url = result["source_url"]
                source_idx[source] = source_url

        for source, source_url in sorted(source_idx.items()):
            lines.append(f"- [{source}]({source_url})")
        lines.append("")
        return lines

    def get_lines_for_usage(self):
        return [
            "## Usage",
            "",
            "### Run Code",
            "",
            "```python",
            "from lanka_data import Db",
            "",
            "",
            'db = Db("<cmd>")',
            "output = db.run()",
            "print(output)",
            "",
            "```",
            "",
            "### workflows/run.py",
            "",
            "```bash",
            "python workflows/run.py <cmd>",
            "```",
            "",
            "### workflows/console.py",
            "",
            "```bash",
            "python workflows/console.py <cmd>",
            "",
            "/Where/What/When/How",
            "",
            "> /<cmd>",
            "```",
            "",
        ]

    def get_lines_for_examples(self, test_idx):
        lines = [
            "## Example Commands (`<cmd>`)",
            "",
        ]

        commands = (
            JSONFile(os.path.join("tests", "test_db.data.json")).read().keys()
        )

        for i_command, command in enumerate(commands, start=1):
            output = test_idx[command]

            lines.append(f"### {i_command:02d}. `{command}`")
            lines.append("")
            lines.append("```json")
            output_json = json.dumps(output, indent=4)
            output_json_lines = output_json.splitlines()
            if len(output_json_lines) > self.MAX_LINES_IN_OUTPUT:
                i_split = self.MAX_LINES_IN_OUTPUT // 2
                n_spaces = len(output_json_lines[i_split - 1]) - len(
                    output_json_lines[i_split - 1].lstrip(" ")
                )
                n_cut = len(output_json_lines) - self.MAX_LINES_IN_OUTPUT
                output_json_lines = (
                    output_json_lines[:i_split]
                    + [" " * n_spaces + f"... // {n_cut} lines ..."]
                    + output_json_lines[-i_split:]
                )
            lines.extend(output_json_lines)
            lines.append("```")
            lines.append("")

            if "result" in output and "image_path" in output["result"]:
                image_path = output["result"]["image_path"]
                new_image_path = os.path.join(
                    "images", "readme", os.path.basename(image_path)
                )
                shutil.copy2(image_path, new_image_path)
                lines.append(f"![{command}]({new_image_path})")
                lines.append("")

        return lines

    def get_lines_for_footer(self):
        return [
            "![Maintainer](https://img.shields.io/badge/maintainer-nuuuwan-red)",
            "![MadeWith](https://img.shields.io/badge/made_with-python-blue)",
            "[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)",
            "",
        ]

    def get_lines(self, test_idx):
        return (
            [
                "# Lanka Data",
                "",
                "This repo implements a simple interface ",
                "to query data about Sri Lanka.",
                "",
            ]
            + self.get_lines_for_sources(test_idx)
            + self.get_lines_for_usage()
            + self.get_lines_for_examples(test_idx)
            + self.get_lines_for_footer()
        )

    def build(self):
        test_idx = self.run_tests()
        lines = self.get_lines(test_idx)
        readme_file = File(self.PATH)
        readme_file.write("\n".join(lines))
        log.info(f"Wrote {readme_file}")


if __name__ == "__main__":
    setup_logging()
    ReadMe().build()
