from pylatex import NoEscape, Subsection


class ResearchPaperDatasetsMixin:
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

    SOURCES = [
        (
            "Department of Census and Statistics, Sri Lanka",
            "https://www.statistics.gov.lk/",
        ),
        (
            "Census of Population and Housing 2001",
            "https://www.statistics.gov.lk"
            "/Population/StaticalInformation/CPH2001",
        ),
        (
            "Census of Population and Housing 2012",
            "https://www.statistics.gov.lk/Resource/en/Population"
            "/CPH_2011/CPH_2012_5Per_Rpt.pdf",
        ),
        (
            "Census of Population and Housing 2024",
            "https://www.statistics.gov.lk"
            "/Population/StaticalInformation/CPH2024",
        ),
        (
            "Election Commission of Sri Lanka",
            "https://www.elections.gov.lk",
        ),
        (
            "HydroRIVERS (via lk\\_rivers)",
            "https://github.com/nuuuwan/lk_rivers",
        ),
    ]

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
