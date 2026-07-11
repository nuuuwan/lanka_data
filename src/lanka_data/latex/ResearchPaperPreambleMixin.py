class ResearchPaperPreambleMixin:
    TITLE = (
        "Lanka Data: A Four-Field Grammar for Querying "
        "Public Data about Sri Lanka"
    )
    AUTHOR = "Nuwan I. Senaratna"

    def get_lines_for_preamble(self):
        return self._get_lines_for_header() + self._get_lines_for_abstract()

    def _get_lines_for_header(self):
        return [
            "\\documentclass[11pt]{article}",
            "\\usepackage[margin=1in]{geometry}",
            "\\usepackage{booktabs}",
            "\\usepackage{amsmath}",
            "\\usepackage[hidelinks]{hyperref}",
            "",
            "\\title{" + self.TITLE + "}",
            "\\author{" + self.AUTHOR + "\\\\",
            "\\texttt{nuuuwan@gmail.com}}",
            "\\date{\\today}",
            "",
            "\\begin{document}",
            "\\maketitle",
            "",
        ]

    def _get_lines_for_abstract(self):
        return [
            "\\begin{abstract}",
            "Lanka Data is a software library that provides a single,",
            "uniform interface to public data about Sri Lanka. Rather",
            "than exposing a growing collection of endpoints, methods,",
            "and parameters, it reduces every query to one string of",
            "four positional fields---\\emph{what} is measured,",
            "\\emph{when}, \\emph{where}, and \\emph{how} it is",
            "presented---delimited by slashes. The same string serves",
            "unchanged as a Python argument, a command-line argument, a",
            "URL path, and a file path. This paper describes the",
            "motivation for the design, specifies the four-field command",
            "grammar and its single intentional coupling, argues that",
            "the grammar spans the target query space by composition",
            "rather than by accretion, and catalogues the datasets---",
            "census, election, administrative geography, and",
            "hydrology---and their sources that the library exposes.",
            "\\end{abstract}",
            "",
        ]

    def get_lines_for_end(self):
        return ["", "\\end{document}", ""]
