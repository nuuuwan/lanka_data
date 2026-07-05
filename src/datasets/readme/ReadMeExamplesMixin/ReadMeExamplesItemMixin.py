import json
import os
import shutil


class ReadMeExamplesItemMixin:
    MAX_LINES_IN_OUTPUT = 40

    @staticmethod
    def get_lines_for_example_title(i_group_name, i_cmd, cmd, result):
        return [f"#### {i_group_name}.{i_cmd:02d}) {cmd}", ""]

    @staticmethod
    def get_lines_for_command(cmd):
        return ["```bash", f"{cmd}", "```", ""]

    @staticmethod
    def get_lines_for_output(cmd, output):
        from datasets.examples.Example.Example import Example

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
            Example.DIR_EXAMPLES_OUTPUT, cmd, "Output.json"
        )
        lines.append(f"Source: [{output_path}]({output_path})")
        lines.append("")
        return lines

    @staticmethod
    def get_lines_for_image(cmd, output):
        from datasets.examples.Example.Example import Example

        if not ("result" in output and "image_path" in output["result"]):
            return []
        lines = []
        image_path = output["result"]["image_path"]
        new_image_dir = os.path.join(Example.DIR_EXAMPLES_OUTPUT, cmd)
        os.makedirs(new_image_dir, exist_ok=True)
        new_image_path = os.path.join(new_image_dir, "Image.png")
        shutil.copy2(image_path, new_image_path)
        lines.append(f"![{cmd}]({new_image_path})")
        lines.append("")
        lines.append(f"Source: [{new_image_path}]({new_image_path})")
        lines.append("")
        return lines
