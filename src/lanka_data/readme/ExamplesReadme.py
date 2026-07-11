import os

from lanka_data.readme.ReadMeExamplesMixin.ReadMeExamplesItemMixin import \
    ReadMeExamplesItemMixin
from utils_future import File, JSONFile, Log

log = Log("ExamplesReadme")


class ExamplesReadme(ReadMeExamplesItemMixin):
    PATH = "README.examples.md"

    def get_lines(self, example_idx, output_idx):
        return [
            "# Examples",
            "",
            "This file showcases all examples from `examples.json`"
            + " with their output results.",
            "",
        ] + self.get_lines_for_examples(example_idx, output_idx)

    @staticmethod
    def get_lines_for_example(i_group_name, i_cmd, cmd, output_idx):
        output = output_idx[cmd]
        result = output["result"]
        lines = []
        lines.extend(
            ReadMeExamplesItemMixin.get_lines_for_example_title(
                i_group_name, i_cmd, cmd, result
            )
        )
        lines.extend(ReadMeExamplesItemMixin.get_lines_for_command(cmd))
        lines.extend(
            ReadMeExamplesItemMixin.get_lines_for_output(cmd, output)
        )
        lines.extend(ReadMeExamplesItemMixin.get_lines_for_image(cmd, output))
        return lines

    def get_lines_for_examples(self, example_idx, output_idx):
        lines = []
        for i_group_name, (group_name, cmds) in enumerate(
            example_idx.items(), start=1
        ):
            lines.append(f"## {i_group_name}) {group_name}")
            lines.append("")
            for i_cmd, cmd in enumerate(cmds, start=1):
                lines.extend(
                    self.get_lines_for_example(
                        i_group_name, i_cmd, cmd, output_idx
                    )
                )
        return lines

    def build(self):
        example_idx = JSONFile(
            os.path.join("examples", "examples.json")
        ).read()
        output_idx = {
            cmd: JSONFile(os.path.join("_output", cmd, "Output.json")).read()
            for cmds in example_idx.values()
            for cmd in cmds
        }
        lines = self.get_lines(example_idx, output_idx)
        readme_file = File(self.PATH)
        readme_file.write("\n".join(lines))
        log.info(f"Wrote {readme_file}")
