class ReadMeUsageMixin:
    def get_lines_for_usage(self):
        return (
            ["## Usage", ""]
            + [
                "### Run Code",
                "",
                "```python",
                "from lanka_data import CommandRunner",
                "",
                "",
                'output = CommandRunner.run("<cmd>")',
                "",
                "```",
                "",
            ]
            + [
                "### workflows/single.py",
                "",
                "Runs single command.",
                "",
                "```bash",
                "python workflows/single.py <cmd>",
                "```",
                "",
            ]
        )
