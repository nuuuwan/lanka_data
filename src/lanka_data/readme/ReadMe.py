from lanka_data.examples import Example
from lanka_data.readme.ReadMeExamplesMixin import ReadMeExamplesMixin
from lanka_data.readme.ReadMeFooterMixin import ReadMeFooterMixin
from lanka_data.readme.ReadMeSourcesMixin import ReadMeSourcesMixin
from lanka_data.readme.ReadMeUsageMixin import ReadMeUsageMixin
from utils_future import File, Log

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
                "## Design Philosophy",
                "",
                "See [README.philosophy.md](README.philosophy.md).",
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
