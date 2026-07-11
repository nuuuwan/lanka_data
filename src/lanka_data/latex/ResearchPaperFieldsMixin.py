class ResearchPaperFieldsMixin:
    def get_lines_for_fields(self):
        return (
            self._get_lines_for_what()
            + self._get_lines_for_when()
            + self._get_lines_for_where()
            + self._get_lines_for_how()
        )

    def _get_lines_for_what(self):
        return [
            "\\subsection{What --- the measurement}",
            "",
            "\\textbf{What} identifies the quantity being retrieved. It",
            "is a measurement, independent of time, region, and",
            "presentation. Measurements include census quantities and",
            "election results. A reserved keyword denotes the",
            "\\emph{absence} of a measurement: a request for region",
            "geometry with no data bound to it, used for base maps and",
            "for inspecting boundary changes in isolation. Because",
            "\\textbf{What} encodes only the measurement type, a single",
            "measurement is reused across every combination of the",
            "other three fields without expanding the vocabulary.",
            "",
        ]

    def _get_lines_for_when(self):
        return [
            "\\subsection{When --- the observation time}",
            "",
            "\\textbf{When} binds the measurement to the point or",
            "interval in time at which it happened. It accepts either a",
            "single year or an interval. An interval is a single value,",
            "not two concatenated queries.",
            "",
        ]

    def _get_lines_for_where(self):
        return [
            "\\subsection{Where --- the region}",
            "",
            "\\textbf{Where} identifies the region under measurement: its",
            "identity within the administrative hierarchy, not merely a",
            "coordinate. It carries the most syntax of the four fields",
            "because it is the axis along which real queries vary most.",
            "The supported forms are a single region, resolution into",
            "child regions of a given type, an explicit set of named",
            "regions, a contiguous range between two endpoints, and a",
            "historical boundary variant that selects a region's",
            "boundaries as they existed before a given boundary",
            "redesign. Observation time and boundary epoch are kept as",
            "separate values so that counts are not misattributed to the",
            "wrong geometry.",
            "",
        ]

    def _get_lines_for_how(self):
        return [
            "\\subsection{How --- the presentation}",
            "",
            "\\textbf{How} specifies the output representation. The same",
            "measurement can be emitted as a map, a chart, or raw data",
            "without any change to \\textbf{What}. Because presentation",
            "is separated from measurement, \\textbf{How} carries its own",
            "modifier grammar: rankings of a category, shares,",
            "differences between observations, and diversity indices are",
            "transformations applied to the same underlying values. This",
            "is the single intentional coupling of the grammar:",
            "change-based modifiers are valid only when \\textbf{When}",
            "supplies an interval.",
            "",
        ]
