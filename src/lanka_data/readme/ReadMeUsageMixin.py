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
                "### HTTP Request (via Vercel App)",
                "",
                "Runs single command over HTTP.",
                "",
                "```bash",
                "https://lanka-data-phi.vercel.app"
                + "/Religion/2024/LK/Map/Image.png",
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
