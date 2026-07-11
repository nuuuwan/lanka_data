from lanka_data.datasets.command.CommandHelp.CommandHelp import CommandHelp
from utils_future import File, Log

log = Log("HelpReadme")


class HelpReadme:
    PATH = "README.help.md"

    def get_lines(self):
        help_result = CommandHelp.get_help_result()
        lines = [
            "# Help",
            "",
            "Command API documentation",
            "",
        ]
        lines += self.get_what_lines(help_result["What"])
        lines += self.get_when_lines(help_result["When"])
        lines += self.get_where_lines(help_result["Where"])
        lines += self.get_how_lines(help_result["How"])
        return lines

    @staticmethod
    def get_what_lines(what_help):
        lines = ["## What", ""]
        lines.append("Available data categories:")
        lines.append("")
        for group, items in what_help.items():
            lines.append(f"### {group}")
            lines.append("")
            for label, description in items.items():
                lines.append(f"- **{label}**: {description}")
            lines.append("")
        return lines

    @staticmethod
    def get_when_lines(when_help):
        lines = ["## When", ""]
        lines.append(when_help)
        lines.append("")
        return lines

    @staticmethod
    def get_where_lines(where_help):
        lines = ["## Where", ""]
        for pattern, info in where_help.items():
            lines.append(f"### {pattern}")
            lines.append("")
            lines.append(f"{info['description']}")
            lines.append("")
            if "examples" in info and info["examples"]:
                lines.append("**Examples:**")
                for example in info["examples"]:
                    lines.append(f"- `{example}`")
                lines.append("")
        return lines

    @staticmethod
    def get_how_lines(how_help):
        lines = ["## How", ""]
        how_info = how_help.get("<base>:<Optional param>", {})
        bases = how_info.get("bases", {})
        params = how_info.get("params", {})

        lines.append("### Bases")
        lines.append("")
        for base_name, description in bases.items():
            if description:
                lines.append(f"- **{base_name}**: {description}")
            else:
                lines.append(f"- **{base_name}**: (no description)")
        lines.append("")

        lines.append("### Parameters")
        lines.append("")
        for param_name, description in params.items():
            lines.append(f"- **{param_name}**: {description}")
        lines.append("")

        return lines

    def build(self):
        lines = self.get_lines()
        readme_file = File(self.PATH)
        readme_file.write("\n".join(lines))
        log.info(f"Wrote {readme_file}")
