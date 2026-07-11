from pylatex import Center, Itemize, NoEscape, Subsection, Subsubsection


class ResearchPaperGrammarMixin:
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
                    r'\texttt{<r1>,<r2>}: Multiple regions '
                    r'(e.g., LK-1,LK-2)'
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
