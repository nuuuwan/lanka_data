from lanka_data.latex.ResearchPaperReferencesMixin import (
    ResearchPaperReferencesMixin,
)


class ResearchPaperDatasetsMixin(ResearchPaperReferencesMixin):
    DATASETS = [
        (
            "Administrative geography",
            "Current and historical boundary epochs",
            "Department of Census and Statistics",
        ),
        (
            "Census of Population and Housing",
            "2001, 2012, 2024",
            "Department of Census and Statistics",
        ),
        (
            "Elections (Presidential, Parliamentary, Local)",
            "Multiple election years",
            "Election Commission of Sri Lanka",
        ),
        (
            "Rivers (length, catchment)",
            "2026",
            "HydroRIVERS (via lk\\_rivers)",
        ),
    ]

    def get_lines_for_datasets(self):
        return (
            self._get_lines_for_datasets_intro()
            + self._get_lines_for_table()
            + self._get_lines_for_references()
        )

    def _get_lines_for_datasets_intro(self):
        return [
            "\\section{Datasets and Sources}",
            "",
            "The \\textbf{What} vocabulary is organised into four data",
            "categories. \\emph{census-population} exposes measurements",
            "such as ethnicity, religion, age group, education,",
            "employment, and literacy; \\emph{census-housing} exposes",
            "housing characteristics such as construction materials,",
            "water, lighting, and toilet facilities;",
            "\\emph{election} exposes presidential, parliamentary, and",
            "local government results and summaries; and \\emph{rivers}",
            "exposes river length and catchment statistics. Each",
            "response carries the provenance of the sources below.",
            "",
        ]

    def _get_lines_for_table(self):
        lines = [
            "\\begin{table}[h]",
            "\\centering",
            "\\begin{tabular}{lll}",
            "\\toprule",
            "\\textbf{Dataset} & \\textbf{Coverage} & \\textbf{Source} \\\\",
            "\\midrule",
        ]
        for dataset, coverage, source in self.DATASETS:
            lines.append(
                dataset + " & " + coverage + " & " + source + " \\\\"
            )
        lines += [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Datasets exposed by Lanka Data and their sources.}",
            "\\end{table}",
            "",
        ]
        return lines
