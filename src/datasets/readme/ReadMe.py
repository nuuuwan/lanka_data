from datasets.examples import Example
from datasets.readme.ReadMeExamplesMixin import ReadMeExamplesMixin
from datasets.readme.ReadMeFooterMixin import ReadMeFooterMixin
from datasets.readme.ReadMeSourcesMixin import ReadMeSourcesMixin
from datasets.readme.ReadMeUsageMixin import ReadMeUsageMixin
from api.utils_future import File, Log

log = Log("ReadMe")


class ReadMe(
    ReadMeSourcesMixin,
    ReadMeUsageMixin,
    ReadMeExamplesMixin,
    ReadMeFooterMixin,
):
    PATH = "README.md"

    def get_lines(self, example_idx, output_idx):
        return (
            [
                "# Lanka Data",
                "",
                'This repo implements "one API to rule them all":'
                + " a single interface that can express"
                + " *any* query to access public data about Sri Lanka 🇱🇰.",
                "",
                "## 0. Design Philosophy & Code",
                "",
                "See [README.philosophy.md](README.philosophy.md)"
                + " and [README.code.md](README.code.md)",
                "",
            ]
            + self.get_lines_for_sources(output_idx)
            + self.get_lines_for_usage()
            + self.get_lines_for_examples(example_idx, output_idx)
            + self.get_lines_for_footer()
        )

    def build(self):
        example_idx = Example.get_group_to_examples()
        output_idx = Example.get_cmd_to_output()
        lines = self.get_lines(example_idx, output_idx)
        readme_file = File(self.PATH)
        readme_file.write("\n".join(lines))
        log.info(f"Wrote {readme_file}")
