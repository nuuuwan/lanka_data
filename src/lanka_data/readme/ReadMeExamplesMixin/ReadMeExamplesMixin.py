from lanka_data.readme.ReadMeExamplesMixin.ReadMeExamplesItemMixin import \
    ReadMeExamplesItemMixin


class ReadMeExamplesMixin(ReadMeExamplesItemMixin):
    @staticmethod
    def get_lines_for_example(i_group_name, i_cmd, example, output_idx):
        lines = []
        cmd = example.cmd
        output = output_idx[cmd]
        result = output["result"]
        lines.extend(
            ReadMeExamplesMixin.get_lines_for_example_title(
                i_group_name, i_cmd, cmd, result
            )
        )
        lines.extend(ReadMeExamplesMixin.get_lines_for_command(cmd))
        lines.extend(ReadMeExamplesMixin.get_lines_for_output(cmd, output))
        lines.extend(ReadMeExamplesMixin.get_lines_for_image(cmd, output))
        return lines

    def get_lines_for_examples(self, example_idx, output_idx):
        lines = ["## Example cmds (`<cmd>`)", ""]
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
