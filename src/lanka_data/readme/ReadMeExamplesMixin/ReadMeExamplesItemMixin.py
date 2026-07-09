import json
import os


class ReadMeExamplesItemMixin:
    MAX_LINES_IN_OUTPUT = 40
    DIR_EXAMPLES_OUTPUT = "_output"

    @staticmethod
    def get_lines_for_example_title(i_group_name, i_cmd, cmd, result):
        return [f"#### {i_group_name}.{i_cmd:02d}) {cmd}", ""]

    @staticmethod
    def get_lines_for_command(cmd):
        return ["```bash", f"{cmd}", "```", ""]

    @staticmethod
    def get_lines_for_output(cmd, output):
        lines = ["```json"]
        output_json = json.dumps(output, indent=4)
        output_json_lines = output_json.splitlines()
        max_lines = ReadMeExamplesItemMixin.MAX_LINES_IN_OUTPUT
        if len(output_json_lines) > max_lines:
            i_split = max_lines // 2
            n_spaces = len(output_json_lines[i_split - 1]) - len(
                output_json_lines[i_split - 1].lstrip(" ")
            )
            n_cut = len(output_json_lines) - max_lines
            output_json_lines = (
                output_json_lines[:i_split]
                + [" " * n_spaces + f"... // {n_cut} lines ..."]
                + output_json_lines[-i_split:]
            )
        lines.extend(output_json_lines)
        lines.append("```")
        lines.append("")
        output_path = os.path.join(
            ReadMeExamplesItemMixin.DIR_EXAMPLES_OUTPUT, cmd, "Output.json"
        )
        lines.append(f"Source: [{output_path}]({output_path})")
        lines.append("")
        return lines

    @staticmethod
    def get_lines_for_image(cmd, output):
        if not ("result" in output and "image_path" in output["result"]):
            return []
        image_path = output["result"]["image_path"]
        return [
            f"![{cmd}]({image_path})",
            "",
            f"Source: [{image_path}]({image_path})",
            "",
        ]
