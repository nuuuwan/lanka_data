class ResearchPaperPropertiesMixin:
    def get_lines_for_properties(self):
        return (
            self._get_lines_for_orthogonality()
            + self._get_lines_for_portability()
            + self._get_lines_for_coverage()
        )

    def _get_lines_for_orthogonality(self):
        return [
            "\\section{Orthogonality and Generativity}",
            "",
            "The four fields are independent axes, so the set of valid",
            "queries is their Cartesian product rather than an explicit",
            "list. Changing the presentation leaves the region semantics",
            "unchanged; widening the observation time from a point to an",
            "interval leaves the measurement unchanged while enabling the",
            "difference-based modifiers. Orthogonality is what allows a",
            "small vocabulary to generate a large query space. Any",
            "documented commands are samples from that space, not an",
            "exhaustive specification.",
            "",
        ]

    def _get_lines_for_portability(self):
        return [
            "\\section{One Grammar, Everywhere}",
            "",
            "The command string is deliberately free of characters that",
            "need quoting or escaping in any common context. As a",
            "result, the \\emph{same string} is the entire interface in",
            "every place the data is consumed: as a Python library",
            "argument, as a command-line argument, as an HTTP endpoint",
            "path where the four fields \\emph{are} the URL path, and as",
            "a static file location where the four fields \\emph{are} the",
            "directory structure. This is not four interfaces that",
            "happen to look alike; it is one grammar hosted in four",
            "contexts. The string parses in reading order and mirrors how",
            "a question is asked in plain language, so no programming",
            "knowledge is needed to read or modify a command. Every",
            "response includes the sources that produced it and a query",
            "time, so a compact request still yields a traceable result.",
            "",
        ]

    def _get_lines_for_coverage(self):
        return [
            "\\section{Coverage}",
            "",
            "A four-field grammar is only useful if it spans the",
            "required query space. Distinct classes of query---a single",
            "machine-readable fact, a mapped snapshot, a comparison over",
            "time, a ranked view, a local decomposition, an explicit",
            "comparison set, geometry with no data bound, historical",
            "boundaries, and derived indices---are all expressed by",
            "selecting values along the same four axes. Coverage is",
            "obtained by composition, not by adding endpoints.",
            "",
        ]
