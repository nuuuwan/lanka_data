class ReadMeUsageMixin:
    def get_lines_for_usage(self):
        return (
            ["## 2. Usage", ""]
            + self._get_install_lines()
            + self._get_code_lines()
            + self._get_http_lines()
            + self._get_single_lines()
        )

    @staticmethod
    def _get_install_lines():
        return [
            "### Install Library",
            "",
            "```bash",
            "pip install lanka-data",
            "```",
            "",
        ]

    @staticmethod
    def _get_code_lines():
        return [
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

    @staticmethod
    def _get_http_lines():
        return [
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

    @staticmethod
    def _get_single_lines():
        return [
            "### workflows/single.py",
            "",
            "Runs single command.",
            "",
            "```bash",
            "python workflows/single.py <cmd>",
            "```",
            "",
        ]
