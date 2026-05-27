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

    def get_lines_for_examples(self):
        lines = [
            "## Example Commands (`<cmd>`)",
            "",
        ]

        commands = (
            JSONFile(os.path.join("tests", "test_db.data.json")).read().keys()
        )
        for i_command, command in enumerate(commands, start=1):
            output = Db(command).run(do_open_images=False, do_use_cache=True)

            lines.append(f"### {i_command:02d}. `{command}`")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(output, indent=4))
            lines.append("```")
            lines.append("")

            if "results" in output and "image_path" in output["results"]:
                image_path = output["results"]["image_path"]
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

    def get_lines(self):
        return (
            [
                "# Lanka Data",
                "",
                "This repo implements a simple interface ",
                "to query data about Sri Lanka.",
                "",
            ]
            + self.get_lines_for_usage()
            + self.get_lines_for_examples()
            + self.get_lines_for_footer()
        )

    def build(self):
        lines = self.get_lines()
        readme_file = File(self.PATH)
        readme_file.write("\n".join(lines))
        log.info(f"Wrote {readme_file}")


if __name__ == "__main__":
    setup_logging()
    ReadMe().build()
