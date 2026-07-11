from lanka_data.latex.ResearchPaperDatasetsMixin import (
    ResearchPaperDatasetsMixin,
)
from lanka_data.latex.ResearchPaperDesignMixin import ResearchPaperDesignMixin
from lanka_data.latex.ResearchPaperPreambleMixin import (
    ResearchPaperPreambleMixin,
)
from utils_future import File, Log

log = Log("ResearchPaper")


class ResearchPaper(
    ResearchPaperPreambleMixin,
    ResearchPaperDesignMixin,
    ResearchPaperDatasetsMixin,
):
    DEFAULT_PATH = "_output/lanka_data.tex"

    def __init__(self, path=None):
        self.path = path or self.DEFAULT_PATH

    def get_lines(self):
        return (
            self.get_lines_for_preamble()
            + self.get_lines_for_design()
            + self.get_lines_for_datasets()
            + self.get_lines_for_end()
        )

    def build(self):
        tex_file = File(self.path)
        tex_file.write("\n".join(self.get_lines()))
        log.info(f"Wrote {tex_file}")
        return tex_file
