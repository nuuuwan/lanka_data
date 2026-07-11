import os
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
    DEFAULT_PATH = "latex/lanka_data.tex"

    def __init__(self, path=None):
        self.path = path or self.DEFAULT_PATH

    def build(self):
        tex_dir = os.path.dirname(self.path)
        if tex_dir and not os.path.exists(tex_dir):
            os.makedirs(tex_dir)
        self._create_bibliography_file(tex_dir)
        self._create_bst_file(tex_dir)

        tex_content = self._generate_tex()
        tex_file = File(self.path)
        tex_file.write(tex_content)
        log.info(f"Wrote {tex_file}")
        return tex_file

    def _generate_tex(self):
        """Generate complete LaTeX document as string."""
        lines = []
        lines.append('\\documentclass[10pt,a4paper]{article}%')
        lines.append('\\usepackage[T1]{fontenc}%')
        lines.append('\\usepackage[utf8]{inputenc}%')
        lines.append('\\usepackage{lmodern}%')
        lines.append('\\usepackage{textcomp}%')
        lines.append('\\usepackage{lastpage}%')
        lines.append('\\usepackage{times}%')
        lines.append('\\usepackage{natbib}%')
        lines.append('\\usepackage{hyperref}%')
        lines.append('\\usepackage{booktabs}%')
        lines.append('\\usepackage{amsmath}%')
        lines.append('\\usepackage{acl}%')
        lines.append('')
        lines.append(f'\\title{{{self.TITLE}}}')
        author_line = (
            f'\\author{{{self.AUTHOR}\\\\' + '\\texttt{nuwans@stanford.edu}}'
        )
        lines.append(author_line)
        lines.append('\\date{\\today}')
        lines.append('')
        lines.append('\\begin{document}')
        lines.append('\\maketitle')
        lines.append('')
        lines.extend(self._get_abstract_lines())
        lines.extend(self._get_design_lines())
        lines.extend(self._get_dataset_lines())
        lines.append('')
        lines.append('\\bibliographystyle{acl_natbib}')
        lines.append('\\bibliography{lanka_data}')
        lines.append('\\end{document}')

        return '\n'.join(lines)

    def _get_abstract_lines(self):
        """Get abstract section lines."""
        return [
            '\\begin{abstract}',
            'Lanka Data is a software library that provides a single,',
            'uniform interface to public data about Sri Lanka. Rather',
            'than exposing a growing collection of endpoints, methods,',
            'and parameters, it reduces every query to one string of',
            'four positional fields---\\emph{what} is measured,',
            '\\emph{when}, \\emph{where}, and \\emph{how} it is',
            'presented---delimited by slashes. The same string serves',
            'unchanged as a Python argument, a command-line argument, a',
            'URL path, and a file path. This paper describes the',
            'motivation for the design, specifies the four-field command',
            'grammar and its single intentional coupling, argues that',
            'the grammar spans the target query space by composition',
            'rather than by accretion, and catalogues the datasets---',
            'census, election, administrative geography, and',
            'hydrology---and their sources that the library exposes.',
            '\\end{abstract}',
            '',
        ]

    def _get_design_lines(self):
        """Get design section lines."""
        return (
            self._get_intro_lines()
            + self._get_grammar_lines()
            + self._get_fields_lines()
        )

    def _get_intro_lines(self):
        """Get introduction section lines."""
        return [
            '\\section{Introduction}',
            '',
            'The goal of Lanka Data is ``one API to rule them all\'\': a',
            'single interface that can express \\emph{any} query, rather',
            'than a proliferation of endpoints, methods, libraries, and',
            'parameter sets that each answer one narrow question. Most',
            'data libraries grow by accretion; every new question adds',
            'another function, endpoint, or flag, and no single mental',
            'model survives contact with the result. We wanted the',
            'opposite: a fixed, minimal grammar that a user learns',
            '\\emph{once} and can then aim at anything, and that a',
            'non-technical user can read and write without learning to',
            'program. The current domain is public Sri Lankan data---',
            'census measurements, election results, and administrative',
            'geography---but nothing in the grammar is specific to Sri',
            'Lanka: \\emph{what}, \\emph{when}, \\emph{where}, and',
            '\\emph{how} are the dimensions of essentially any factual',
            'query about the world.',
            '',
        ]

    def _get_grammar_lines(self):
        """Get grammar section lines."""
        return [
            '\\section{The Command Grammar}',
            '',
            'The public interface is a single string parsed into four',
            'positional fields, delimited by slashes:',
            '',
            '\\begin{center}',
            '\\texttt{What / When / Where / How}',
            '\\end{center}',
            '',
            'There is no secondary configuration surface: no options',
            'object, no builder API, no config files. Each field is an',
            'independent axis of the query, and the value chosen for one',
            'field does not constrain the valid values of another, with a',
            'single intentional coupling described below.',
            '',
        ]

    def _get_fields_lines(self):
        """Get What/When/Where/How subsection lines."""
        return (
            self._get_what_lines()
            + self._get_when_lines()
            + self._get_where_lines()
            + self._get_how_lines()
        )

    def _get_what_list_lines(self):
        """Return What field values by category."""
        return [
            '\\subsubsection*{census-population (29 values)}',
            'AgeGroup, AgriOccupations, Attendance, Dependency,',
            'Disability, Economy, Education, Employment, Enrollment,',
            'Ethnicity, Fertility, Gender, Growth, Inactive, Industry,',
            'Laborforce, Literacy, Marital, Migration,',
            'NonAgriEmployment, NotAttending, Occupations, Population,',
            'RelationshipToHead, Religion, Sectoral, Sectors,',
            'Speaking, Unemployment',
            '',
            '\\subsubsection*{census-housing (22 values)}',
            'Communication, ConstructionYear, Electricity, Floor, Fuel,',
            'Housing, Informal, Lighting, Materials, Occupancy,',
            'Ownership, Persons, Quarters, Roof, Rooms, Structure,',
            'Tenure, Toilet, Unit, Walls, Waste, Water',
            '',
            '\\subsubsection*{election (6 values)}',
            'Local, LocalSummary, Parliamentary, ParliamentarySummary,',
            'Presidential, PresidentialSummary',
            '',
            '\\subsubsection*{rivers (2 values)}',
            'Catchment, RiverLen',
            '',
        ]

    def _get_what_lines(self):
        """Get What field lines."""
        lines = [
            '\\subsection{What --- the measurement}',
            '',
            '\\textbf{What} identifies the quantity being retrieved. It',
            'is a measurement, independent of time, region, and',
            'presentation. Measurements include census quantities and',
            'election results. A reserved keyword denotes the',
            '\\emph{absence} of a measurement: a request for region',
            'geometry with no data bound to it, used for base maps and',
            'for inspecting boundary changes in isolation. Because',
            '\\textbf{What} encodes only the measurement type, a single',
            'measurement is reused across every combination of the',
            'other three fields without expanding the vocabulary.',
            '',
        ]
        return lines + self._get_what_list_lines()

    def _get_when_list_lines(self):
        """Return When field examples and syntax."""
        return [
            'Supported formats:',
            '\\begin{itemize}',
            '\\item Single year: e.g., \\texttt{2024}',
            '\\item Interval: e.g., \\texttt{2012-2024}',
            '\\item If exact year unavailable, closest available data',
            'is returned',
            '\\end{itemize}',
            '',
        ]

    def _get_when_lines(self):
        """Get When field lines."""
        lines = [
            '\\subsection{When --- the observation time}',
            '',
            '\\textbf{When} binds the measurement to the point or',
            'interval in time at which it happened. It accepts either a',
            'single year or an interval. An interval is a single value,',
            'not two concatenated queries.',
            '',
        ]
        return lines + self._get_when_list_lines()

    def _get_where_list_lines(self):
        """Return Where field syntax patterns."""
        return [
            'Supported syntax patterns:',
            '\\begin{itemize}',
            '\\item \\texttt{<region\\_id>}: Single region (e.g., LK)',
            '\\item \\texttt{<region\\_id>:<type>}: Child regions',
            '(e.g., LK:district)',
            '\\item \\texttt{<r1>,<r2>}: Multiple regions',
            '(e.g., LK-1,LK-2)',
            '\\item \\texttt{<r1>...<r2>}: Range between endpoints',
            '(e.g., LK-1...LK-2)',
            '\\item \\texttt{<region>@<distance>}: Regions within',
            'distance of same type',
            '\\end{itemize}',
            '',
        ]

    def _get_where_lines(self):
        """Get Where field lines."""
        lines = [
            '\\subsection{Where --- the region}',
            '',
            '\\textbf{Where} identifies the region under measurement: its',
            'identity within the administrative hierarchy, not merely a',
            'coordinate. It carries the most syntax of the four fields',
            'because it is the axis along which real queries vary most.',
            'The supported forms are a single region, resolution into',
            'child regions of a given type, an explicit set of named',
            'regions, a contiguous range between two endpoints, and a',
            'historical boundary variant that selects a region\'s',
            'boundaries as they existed before a given boundary',
            'redesign. Observation time and boundary epoch are kept as',
            'separate values so that counts are not misattributed to the',
            'wrong geometry.',
            '',
        ]
        return lines + self._get_where_list_lines()

    def _get_how_list_lines(self):
        """Return How field visualization bases and modifiers."""
        return [
            'Supported visualization bases (24 total):',
            'BarChart, BivariateMap, BubbleMap, BumpChart, CSV,',
            'ChartSpec, GeoJSON, HexMap, Histogram, JSON, LineChart,',
            'Map, Parquet, PieChart, QuadrantChart, ScatterPlot,',
            'SquareMap, StackedBarChart, TSV, TreeMap, TriangleMap,',
            'UnitHexMap, UnitSquareMap, UnitTriangleMap',
            '',
            'Supported modifiers (12 total):',
            '1st, Top, 2nd, 3rd, Bottom, 1stPct, 2ndPct, Change,',
            'Top3, Diversity, DiversityPew',
            '',
        ]

    def _get_how_lines(self):
        """Get How field lines."""
        lines = [
            '\\subsection{How --- the presentation}',
            '',
            '\\textbf{How} specifies the output representation. The same',
            'measurement can be emitted as a map, a chart, or raw data',
            'without any change to \\textbf{What}. Because presentation',
            'is separated from measurement, \\textbf{How} carries its own',
            'modifier grammar: rankings of a category, shares,',
            'differences between observations, and diversity indices are',
            'transformations applied to the same underlying values. This',
            'is the single intentional coupling of the grammar:',
            'change-based modifiers are valid only when \\textbf{When}',
            'supplies an interval.',
            '',
        ]
        return lines + self._get_how_list_lines()

    def _get_dataset_lines(self):
        """Get datasets and sources section lines."""
        return (
            self._get_datasets_intro_lines()
            + self._get_census_lines()
            + self._get_elections_lines()
            + self._get_admin_lines()
            + self._get_rivers_lines()
        )

    def _get_datasets_intro_lines(self):
        """Get datasets intro lines."""
        return [
            '\\section{Datasets and Sources}',
            '',
            'The \\textbf{What} vocabulary is organised into four data',
            'categories. \\emph{census-population} exposes measurements',
            'such as ethnicity, religion, age group, education,',
            'employment, and literacy; \\emph{census-housing} exposes',
            'housing characteristics such as construction materials,',
            'water, lighting, and toilet facilities;',
            '\\emph{election} exposes presidential, parliamentary, and',
            'local government results and summaries; and \\emph{rivers}',
            'exposes river length and catchment statistics. Each',
            'response carries the provenance of the sources below.',
            '',
        ]

    def _get_census_lines(self):
        """Get census subsection lines."""
        return [
            '\\subsection{Census of Population and Housing}',
            '',
            'The census-population category exposes measurements such as',
            'ethnicity, religion, age group, education, employment, and',
            'literacy. The census-housing category exposes housing',
            'characteristics such as construction materials, water,',
            'lighting, and toilet facilities. Data is available from the',
            '2001, 2012, and 2024 census cycles \\citep{census_2024}.',
            'All census data is sourced from the Department of Census',
            'and Statistics.',
            '',
        ]

    def _get_elections_lines(self):
        """Get elections subsection lines."""
        return [
            '\\subsection{Elections}',
            '',
            'The election category exposes presidential, parliamentary,',
            'and local government results and summaries across multiple',
            'election years. This includes vote tallies, seat',
            'distributions, and electoral statistics at national,',
            'district, and constituency levels. All election data is',
            'sourced from the \\citet{elections_commission}.',
            '',
        ]

    def _get_admin_lines(self):
        """Get administrative geography subsection lines."""
        return [
            '\\subsection{Administrative Geography}',
            '',
            'The administrative boundary data includes both current and',
            'historical boundary epochs, allowing queries to be anchored',
            'to the boundaries as they existed at the time of',
            'observation. This includes provinces, districts, divisional',
            'secretariat divisions, and Grama Niladhari divisions across',
            'different time periods. Data is sourced from the',
            '\\citet{dcs_sri_lanka}.',
            '',
        ]

    def _get_rivers_lines(self):
        """Get rivers subsection lines."""
        return [
            '\\subsection{Rivers}',
            '',
            'The rivers category exposes river length and catchment',
            'statistics for major waterways in Sri Lanka \\citep{'
            'hydrorivers}. This data includes geometric',
            'and hydrological properties of river systems. Data is',
            'sourced from HydroRIVERS via the \\texttt{lk\\_rivers}',
            'project.',
            '',
        ]

    def _create_bibliography_file(self, tex_dir):
        """Create BibTeX file in output directory."""
        bib_path = os.path.join(tex_dir, 'lanka_data.bib')
        lines = self._get_bib_content()
        with open(bib_path, 'w') as f:
            f.write('\n'.join(lines))

    def _create_bst_file(self, tex_dir):
        """Create BibTeX style file in output directory."""
        import shutil

        bst_path = os.path.join(tex_dir, 'acl_natbib.bst')
        if os.path.exists(bst_path):
            return
        src = '/usr/local/texlive/2025/texmf-dist/bibtex/bst/natbib/'
        src += 'plainnat.bst'
        if os.path.exists(src):
            shutil.copy(src, bst_path)
        else:
            log.warning(f"Could not find {src}")

    def _get_census_bib_entries(self):
        """Return census bibliography entries."""
        return [
            '@misc{census_2001,',
            '  author={{Department of Census and Statistics}},',
            '  title={Census of Population and Housing 2001},',
            '  howpublished={\\url{https://www.statistics.gov.lk/',
            '    Population/StaticalInformation/CPH2001}},',
            '  year={2001},',
            '  address={Colombo, Sri Lanka},',
            '  urldate={2024-01-01}',
            '}',
            '',
            '@misc{census_2012,',
            '  author={{Department of Census and Statistics}},',
            '  title={Census of Population and Housing 2012},',
            '  howpublished={\\url{https://www.statistics.gov.lk/',
            '    Resource/en/Population/CPH_2011/',
            '    CPH_2012_5Per_Rpt.pdf}},',
            '  year={2012},',
            '  address={Colombo, Sri Lanka},',
            '  note={Enumeration 2011; report published 2012},',
            '  urldate={2024-01-01}',
            '}',
            '',
            '@misc{census_2024,',
            '  author={{Department of Census and Statistics}},',
            '  title={Census of Population and Housing 2024},',
            '  howpublished={\\url{https://www.statistics.gov.lk/',
            '    Population/StaticalInformation/CPH2024}},',
            '  year={2024},',
            '  address={Colombo, Sri Lanka},',
            '  urldate={2024-01-01}',
            '}',
            '',
        ]

    def _get_organizations_bib_entries(self):
        """Return organization bibliography entries."""
        return [
            '@misc{dcs_sri_lanka,',
            '  author={{Department of Census and Statistics}},',
            '  title={Department of Census and Statistics},',
            '  howpublished={\\url{https://www.statistics.gov.lk/}},',
            '  year={2024},',
            '  note={Official statistics portal, Government},',
            '  urldate={2024-01-01}',
            '}',
            '',
            '@misc{elections_commission,',
            '  author={{Election Commission of Sri Lanka}},',
            '  title={Election Commission of Sri Lanka},',
            '  howpublished={\\url{https://www.elections.gov.lk/}},',
            '  year={2024},',
            '  note={Official elections portal, Government},',
            '  urldate={2024-01-01}',
            '}',
            '',
        ]

    def _get_dataset_bib_entries(self):
        """Return dataset bibliography entries."""
        return [
            '@misc{hydrorivers,',
            '  author={Lehner, Bernhard and Grill, Gunther},',
            '  title={{HydroRIVERS}: A Global Vector River Network},',
            '  howpublished={\\url{https://www.hydrosheds.org/',
            '    products/hydrorivers}},',
            '  year={2013},',
            '  note={Part of HydroSHEDS; Sri Lanka subset via',
            '    \\url{https://github.com/nuuuwan/lk_rivers}},',
            '  urldate={2024-01-01}',
            '}',
            '',
        ]

    def _get_bib_content(self):
        """Return content for BibTeX bibliography file."""
        entries = []
        entries.extend(self._get_organizations_bib_entries())
        entries.extend(self._get_census_bib_entries())
        entries.extend(self._get_dataset_bib_entries())
        return entries
