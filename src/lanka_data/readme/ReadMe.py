import json
import os
import shutil

from lanka_data.examples import Example
from utils_future import File, Log

log = Log("ReadMe")


class ReadMe:
    PATH = "README.md"
    MAX_LINES_IN_OUTPUT = 40

    DIR_IMAGES_README = os.path.join("images", "readme")

    def get_lines_for_sources(self, output_idx):

        lines = [
            "## Data Sources",
            "",
        ]
        source_idx = {}
        for output in output_idx.values():
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

    @staticmethod
    def get_lines_for_example(i_group_name, i_cmd, example, output_idx):
        lines = []
        cmd = example.cmd
        output = output_idx[cmd]
        result = output["result"]

        if "what_description" in result:
            what_description = result["what_description"]
            when_description = result["when_description"]
            where_description = result["where_description"]
            how_description = result["how_description"]
            title_text = (
                f"{how_description} of"
                + f" {what_description} ({when_description}) for"
                + f" {where_description}."
            )
        else:
            title_text = cmd

        lines.append(f"#### {i_group_name}.{i_cmd:02d}) {title_text}")
        lines.append("")

        lines.append("```bash")
        lines.append(f"{cmd}")
        lines.append("```")
        lines.append("")

        lines.append("```json")
        output_json = json.dumps(output, indent=4)
        output_json_lines = output_json.splitlines()
        if len(output_json_lines) > ReadMe.MAX_LINES_IN_OUTPUT:
            i_split = ReadMe.MAX_LINES_IN_OUTPUT // 2
            n_spaces = len(output_json_lines[i_split - 1]) - len(
                output_json_lines[i_split - 1].lstrip(" ")
            )
            n_cut = len(output_json_lines) - ReadMe.MAX_LINES_IN_OUTPUT
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
            os.makedirs(ReadMe.DIR_IMAGES_README, exist_ok=True)
            new_image_path = os.path.join(
                ReadMe.DIR_IMAGES_README, os.path.basename(image_path)
            )
            shutil.copy2(image_path, new_image_path)
            lines.append(f"![{cmd}]({new_image_path})")
            lines.append("")

        return lines

    def get_lines_for_examples(self, example_idx, output_idx):
        lines = [
            "## Example cmds (`<cmd>`)",
            "",
        ]

        for i_group_name, (group_name, examples) in enumerate(
            example_idx.items(), start=1
        ):

            lines.append(f"### {i_group_name}) {group_name}")
            lines.append("")
            for i_cmd, example in enumerate(examples, start=1):
                lines.extend(
                    self.get_lines_for_example(
                        i_group_name, i_cmd, example, output_idx
                    )
                )

        return lines

    def get_lines_for_footer(self):
        return [
            "![Maintainer]"
            + "(https://img.shields.io/badge/maintainer-nuuuwan-red)",
            "![MadeWith](https://img.shields.io/badge/made_with-python-blue)",
            "[![License: MIT]"
            + "(https://img.shields.io/badge/License-MIT-yellow.svg)]"
            + "(https://opensource.org/licenses/MIT)",
            "",
        ]

    def get_lines(self, example_idx, output_idx):
        return (
            [
                "# Lanka Data",
                "",
                "This repo implements a simple interface"
                + " to query data about Sri Lanka.",
                "",
            ]
            + self.get_lines_for_sources(output_idx)
            + self.get_lines_for_usage()
            + self.get_lines_for_examples(example_idx, output_idx)
            + self.get_lines_for_footer()
        )

    def cleanup(self):
        if os.path.exists(self.DIR_IMAGES_README):
            shutil.rmtree(self.DIR_IMAGES_README)

    def build(self):
        self.cleanup()
        example_idx = Example.get_example_idx()
        output_idx = Example.get_output_idx()
        lines = self.get_lines(example_idx, output_idx)
        readme_file = File(self.PATH)
        readme_file.write("\n".join(lines))
        log.info(f"Wrote {readme_file}")
