class ReadMeUsageMixin:
    def get_lines_for_usage(self):
        return (
            ["## Usage", ""]
            + [
                "### Run Code",
                "",
                "```python",
                "from lanka_data import Command",
                "",
                "",
                'db = Command("<cmd>")',
                "output = db.run()",
                "print(output)",
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
            + [
                "### workflows/console.py",
                "",
                "Console tool for running commands.",
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
        )
