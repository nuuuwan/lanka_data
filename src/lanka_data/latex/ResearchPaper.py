import os
from pylatex import (
    Center,
    Command,
    Document,
    Itemize,
    NoEscape,
    Package,
    Section,
    Subsection,
    Subsubsection,
)
from pylatex.base_classes import Environment
from lanka_data.latex.ResearchPaperDatasetsMixin import (
    ResearchPaperDatasetsMixin,
)
from lanka_data.latex.ResearchPaperDesignMixin import ResearchPaperDesignMixin
from lanka_data.latex.ResearchPaperPreambleMixin import (
    ResearchPaperPreambleMixin,
)
from utils_future import Log

log = Log("ResearchPaper")


class ResearchPaper(
    ResearchPaperPreambleMixin,
    ResearchPaperDesignMixin,
    ResearchPaperDatasetsMixin,
):
    DEFAULT_PATH = "latex/lanka_data.tex"

    class Abstract(Environment):
        _latex_name = 'abstract'
        escape = False

    def __init__(self, path=None):
        self.path = path or self.DEFAULT_PATH

    def build(self):
        tex_dir = os.path.dirname(self.path)
        if tex_dir and not os.path.exists(tex_dir):
            os.makedirs(tex_dir)
        self._create_bibliography_file(tex_dir)
        self._create_bst_file(tex_dir)
        doc = self._build_document()
        tex_no_ext = os.path.splitext(self.path)[0]
        doc.generate_tex(tex_no_ext)
        log.info(f"Wrote {self.path}")

    def _build_document(self):
        doc = Document(
            documentclass='article',
            document_options=['10pt', 'a4paper'],
            fontenc='T1',
            inputenc='utf8',
            lmodern=True,
            textcomp=True,
            page_numbers=False,
        )
        for pkg in [
            'lastpage',
            'times',
            'natbib',
            'hyperref',
            'booktabs',
            'amsmath',
            'acl',
        ]:
            doc.packages.append(Package(pkg))
        doc.preamble.append(Command('title', NoEscape(self.TITLE)))
        author = NoEscape(
            self.AUTHOR + r'\\' + r'\texttt{nuwans@stanford.edu}'
        )
        doc.preamble.append(Command('author', author))
        doc.preamble.append(Command('date', NoEscape(r'\today')))
        doc.append(NoEscape(r'\maketitle'))
        self._add_body(doc)
        return doc

    def _add_body(self, doc):
        self._add_abstract(doc)
        with doc.create(Section('Introduction')) as sec:
            self._add_intro_body(sec)
        with doc.create(Section('The Command Grammar')) as sec:
            self._add_grammar_body(sec)
        with doc.create(Section('Datasets and Sources')) as sec:
            self._add_datasets_body(sec)
        doc.append(NoEscape(r'\bibliographystyle{acl_natbib}'))
        doc.append(NoEscape(r'\bibliography{lanka_data}'))

    def _add_abstract(self, doc):
        with doc.create(self.Abstract()) as abs_env:
            abs_env.append(
                NoEscape(
                    'Lanka Data is a software library that provides a single, '
                    'uniform interface to public data about Sri Lanka. Rather '
                    'than exposing a growing collection of endpoints, methods, '
                    'and parameters, it reduces every query to one string of '
                    r'four positional fields---\emph{what} is measured, '
                    r'\emph{when}, \emph{where}, and \emph{how} it is '
                    'presented---delimited by slashes. The same string serves '
                    'unchanged as a Python argument, '
                    'a command-line argument, a '
                    'URL path, and a file path. This paper describes the '
                    'motivation for the design, '
                    'specifies the four-field command '
                    'grammar and its single intentional coupling, argues that '
                    'the grammar spans the target query space by composition '
                    'rather than by accretion, and catalogues the datasets---'
                    'census, election, administrative geography, and '
                    'hydrology---and their sources that the library exposes.'
                )
            )

    def _add_intro_body(self, sec):
        sec.append(
            NoEscape(
                "The goal of Lanka Data is ``one API to rule them all'': a "
                r'single interface that can express \emph{any} query, rather '
                'than a proliferation of endpoints, methods, libraries, and '
                'parameter sets that each answer one narrow question. Most '
                'data libraries grow by accretion; every new question adds '
                'another function, endpoint, or flag, and no single mental '
                'model survives contact with the result. We wanted the '
                r'opposite: a fixed, minimal grammar that a user learns '
                r'\emph{once} and can then aim at anything, and that a '
                'non-technical user can read and write without learning to '
                r'program. The current domain is public Sri Lankan data---'
                'census measurements, election results, and administrative '
                r'geography---but nothing in the grammar is specific to Sri '
                r'Lanka: \emph{what}, \emph{when}, \emph{where}, and '
                r'\emph{how} are the dimensions of essentially any factual '
                'query about the world.'
            )
        )

    def _add_grammar_body(self, sec):
        sec.append(
            NoEscape(
                'The public interface is a single string parsed into four '
                'positional fields, delimited by slashes:'
            )
        )
        with sec.create(Center()) as center:
            center.append(NoEscape(r'\texttt{What / When / Where / How}'))
        sec.append(
            NoEscape(
                'There is no secondary configuration surface: no options '
                'object, no builder API, no config files. Each field is an '
                'independent axis of the query, and the value chosen for one '
                'field does not constrain the valid values of another, with a '
                'single intentional coupling described below.'
            )
        )
        with sec.create(Subsection('What --- the measurement')) as ss:
            self._add_what_body(ss)
        with sec.create(Subsection('When --- the observation time')) as ss:
            self._add_when_body(ss)
        with sec.create(Subsection('Where --- the region')) as ss:
            self._add_where_body(ss)
        with sec.create(Subsection('How --- the presentation')) as ss:
            self._add_how_body(ss)

    def _add_what_body(self, sec):
        sec.append(
            NoEscape(
                r'\textbf{What} identifies the quantity being retrieved. It '
                'is a measurement, independent of time, region, and '
                'presentation. Measurements include census quantities and '
                r'election results. A reserved keyword denotes the '
                r'\emph{absence} of a measurement: a request for region '
                'geometry with no data bound to it, used for base maps and '
                'for inspecting boundary changes in isolation. Because '
                r'\textbf{What} encodes only the measurement type, a single '
                'measurement is reused across every combination of the '
                'other three fields without expanding the vocabulary.'
            )
        )
        self._add_what_categories(sec)

    def _add_what_categories(self, sec):
        for title, content in [
            (
                'census-population (29 values)',
                'AgeGroup, AgriOccupations, Attendance, Dependency, '
                'Disability, Economy, Education, Employment, Enrollment, '
                'Ethnicity, Fertility, Gender, Growth, Inactive, Industry, '
                'Laborforce, Literacy, Marital, Migration, '
                'NonAgriEmployment, NotAttending, Occupations, Population, '
                'RelationshipToHead, Religion, Sectoral, Sectors, '
                'Speaking, Unemployment',
            ),
            (
                'census-housing (22 values)',
                'Communication, ConstructionYear, Electricity, Floor, Fuel, '
                'Housing, Informal, Lighting, Materials, Occupancy, '
                'Ownership, Persons, Quarters, Roof, Rooms, Structure, '
                'Tenure, Toilet, Unit, Walls, Waste, Water',
            ),
            (
                'election (6 values)',
                'Local, LocalSummary, Parliamentary, '
                'ParliamentarySummary, Presidential, PresidentialSummary',
            ),
            ('rivers (2 values)', 'Catchment, RiverLen'),
        ]:
            with sec.create(Subsubsection(title, numbering=False)) as ss:
                ss.append(NoEscape(content))

    def _add_when_body(self, sec):
        sec.append(
            NoEscape(
                r'\textbf{When} binds the measurement to the point or '
                'interval in time at which it happened. It accepts either a '
                'single year or an interval. An interval is a single value, '
                'not two concatenated queries.'
            )
        )
        sec.append(NoEscape('Supported formats:'))
        with sec.create(Itemize()) as itemize:
            itemize.add_item(NoEscape(r'Single year: e.g., \texttt{2024}'))
            itemize.add_item(NoEscape(r'Interval: e.g., \texttt{2012-2024}'))
            itemize.add_item(
                NoEscape(
                    'If exact year unavailable, closest available data '
                    'is returned'
                )
            )

    def _add_where_body(self, sec):
        sec.append(
            NoEscape(
                r'\textbf{Where} identifies the region under measurement: its '
                'identity within the administrative hierarchy, not merely a '
                'coordinate. It carries the most syntax of the four fields '
                'because it is the axis along which real queries vary most. '
                'The supported forms are a single region, resolution into '
                'child regions of a given type, an explicit set of named '
                "regions, a contiguous range between two endpoints, and a "
                "historical boundary variant that selects a region's "
                'boundaries as they existed before a given boundary '
                'redesign. Observation time and boundary epoch are kept as '
                'separate values so that counts are not misattributed to the '
                'wrong geometry.'
            )
        )
        self._add_where_items(sec)

    def _add_where_items(self, sec):
        sec.append(NoEscape('Supported syntax patterns:'))
        with sec.create(Itemize()) as itemize:
            itemize.add_item(
                NoEscape(r'\texttt{<region\_id>}: Single region (e.g., LK)')
            )
            itemize.add_item(
                NoEscape(
                    r'\texttt{<region\_id>:<type>}: Child regions '
                    r'(e.g., LK:district)'
                )
            )
            itemize.add_item(
                NoEscape(
                    r'\texttt{<r1>,<r2>}: Multiple regions (e.g., LK-1,LK-2)'
                )
            )
            itemize.add_item(
                NoEscape(
                    r'\texttt{<r1>...<r2>}: Range between endpoints '
                    r'(e.g., LK-1...LK-2)'
                )
            )
            itemize.add_item(
                NoEscape(
                    r'\texttt{<region>@<distance>}: Regions within '
                    'distance of same type'
                )
            )

    def _add_how_body(self, sec):
        sec.append(
            NoEscape(
                r'\textbf{How} specifies the output representation. The same '
                r'measurement can be emitted as a map, a chart, or raw data '
                r'without any change to \textbf{What}. Because presentation '
                r'is separated from measurement, \textbf{How} carries its own '
                'modifier grammar: rankings of a category, shares, '
                'differences between observations, and diversity indices are '
                'transformations applied to the same underlying values. This '
                'is the single intentional coupling of the grammar: '
                r'change-based modifiers are valid only when \textbf{When} '
                'supplies an interval.'
            )
        )
        sec.append(
            NoEscape(
                'Supported visualization bases (24 total):\n'
                'BarChart, BivariateMap, BubbleMap, BumpChart, CSV, '
                'ChartSpec, GeoJSON, HexMap, Histogram, JSON, LineChart, '
                'Map, Parquet, PieChart, QuadrantChart, ScatterPlot, '
                'SquareMap, StackedBarChart, TSV, TreeMap, TriangleMap, '
                'UnitHexMap, UnitSquareMap, UnitTriangleMap'
            )
        )
        sec.append(
            NoEscape(
                'Supported modifiers (12 total):\n'
                '1st, Top, 2nd, 3rd, Bottom, 1stPct, 2ndPct, Change, '
                'Top3, Diversity, DiversityPew'
            )
        )

    def _add_datasets_body(self, sec):
        sec.append(
            NoEscape(
                r'The \textbf{What} vocabulary is organised into four data '
                r'categories. \emph{census-population} exposes measurements '
                'such as ethnicity, religion, age group, education, '
                r'employment, and literacy; \emph{census-housing} exposes '
                'housing characteristics such as construction materials, '
                r'water, lighting, and toilet facilities; '
                r'\emph{election} exposes presidential, parliamentary, and '
                r'local government results and summaries; and \emph{rivers} '
                'exposes river length and catchment statistics. Each '
                'response carries the provenance of the sources below.'
            )
        )
        self._add_dataset_subsections(sec)

    def _add_dataset_subsections(self, sec):
        self._add_census_elections(sec)
        self._add_admin_rivers(sec)

    def _add_census_elections(self, sec):
        with sec.create(Subsection('Census of Population and Housing')) as ss:
            ss.append(
                NoEscape(
                    'The census-population category exposes measurements '
                    'such as ethnicity, religion, age group, education, '
                    'employment, and literacy. The census-housing category '
                    'exposes housing characteristics such as construction '
                    'materials, water, lighting, and toilet facilities. '
                    r'Data is available from the 2001, 2012, and 2024 '
                    r'census cycles \citep{census_2024}. All census data '
                    'is sourced from the Department of Census and Statistics.'
                )
            )
        with sec.create(Subsection('Elections')) as ss:
            ss.append(
                NoEscape(
                    'The election category exposes presidential, '
                    'parliamentary, and local government results and '
                    'summaries across multiple election years. This includes '
                    'vote tallies, seat distributions, and electoral '
                    'statistics at national, district, and constituency '
                    r'levels. All election data is sourced from the '
                    r'\citet{elections_commission}.'
                )
            )

    def _add_admin_rivers(self, sec):
        with sec.create(Subsection('Administrative Geography')) as ss:
            ss.append(
                NoEscape(
                    'The administrative boundary data includes both current '
                    'and historical boundary epochs, allowing queries to be '
                    'anchored to the boundaries as they existed at the time '
                    'of observation. This includes provinces, districts, '
                    'divisional secretariat divisions, and Grama Niladhari '
                    r'divisions across different time periods. Data is '
                    r'sourced from the \citet{dcs_sri_lanka}.'
                )
            )
        with sec.create(Subsection('Rivers')) as ss:
            ss.append(
                NoEscape(
                    'The rivers category exposes river length and catchment '
                    r'statistics for major waterways in Sri Lanka '
                    r'\citep{hydrorivers}. This data includes geometric '
                    'and hydrological properties of river systems. Data is '
                    r'sourced from HydroRIVERS via the \texttt{lk\_rivers} '
                    'project.'
                )
            )

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
