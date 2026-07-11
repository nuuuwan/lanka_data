from lanka_data.latex.ResearchPaperFieldsMixin import ResearchPaperFieldsMixin
from lanka_data.latex.ResearchPaperPropertiesMixin import (
    ResearchPaperPropertiesMixin,
)


class ResearchPaperDesignMixin(
    ResearchPaperFieldsMixin,
    ResearchPaperPropertiesMixin,
):
    def get_lines_for_design(self):
        return (
            self._get_lines_for_introduction()
            + self._get_lines_for_grammar()
            + self.get_lines_for_fields()
            + self.get_lines_for_properties()
        )

    def _get_lines_for_introduction(self):
        return [
            "\\section{Introduction}",
            "",
            "The goal of Lanka Data is ``one API to rule them all'': a",
            "single interface that can express \\emph{any} query, rather",
            "than a proliferation of endpoints, methods, libraries, and",
            "parameter sets that each answer one narrow question. Most",
            "data libraries grow by accretion; every new question adds",
            "another function, endpoint, or flag, and no single mental",
            "model survives contact with the result. We wanted the",
            "opposite: a fixed, minimal grammar that a user learns",
            "\\emph{once} and can then aim at anything, and that a",
            "non-technical user can read and write without learning to",
            "program. The current domain is public Sri Lankan data---",
            "census measurements, election results, and administrative",
            "geography---but nothing in the grammar is specific to Sri",
            "Lanka: \\emph{what}, \\emph{when}, \\emph{where}, and",
            "\\emph{how} are the dimensions of essentially any factual",
            "query about the world.",
            "",
        ]

    def _get_lines_for_grammar(self):
        return [
            "\\section{The Command Grammar}",
            "",
            "The public interface is a single string parsed into four",
            "positional fields, delimited by slashes:",
            "",
            "\\begin{center}",
            "\\texttt{What / When / Where / How}",
            "\\end{center}",
            "",
            "There is no secondary configuration surface: no options",
            "object, no builder API, no config files. Each field is an",
            "independent axis of the query, and the value chosen for one",
            "field does not constrain the valid values of another, with a",
            "single intentional coupling described below.",
            "",
        ]
