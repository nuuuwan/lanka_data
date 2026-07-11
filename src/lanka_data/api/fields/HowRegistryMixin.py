import warnings

from lanka_data.api.fields.HowRegistryBaseLabelsMixin import (
    HowRegistryBaseLabelsMixin,
)

warnings.warn(
    "HowRegistryMixin is deprecated. Use individual visual classes' "
    "get_description() methods and lanka_data.visual.HowParam.HOW_PARAMS "
    "instead.",
    DeprecationWarning,
    stacklevel=2,
)


class HowRegistryMixin(HowRegistryBaseLabelsMixin):
    INTERVAL_BASES = {"BumpChart", "LineChart"}
    SERIES_BASES = {"LineChart"}
    CATEGORY_BASES = {
        "Map",
        "Cartogram",
        "HexMap",
        "UnitHexMap",
        "SquareMap",
        "UnitSquareMap",
        "TriangleMap",
        "UnitTriangleMap",
        "BubbleMap",
        "None",
    }
    PAIR_CATEGORY_BASES = {"BivariateMap", "QuadrantChart", "ScatterPlot"}
    MODIFIERS = {
        "1st": {"label": "Most common", "rank": 0},
        "Top": {"label": "Most common", "rank": 0},
        "2nd": {"label": "2nd most common", "rank": 1},
        "3rd": {"label": "3rd most common", "rank": 2},
        "Bottom": {"label": "Least common", "rank": -1},
        "1stPct": {"label": "Most common share", "pct_rank": 0},
        "2ndPct": {"label": "2nd most common share", "pct_rank": 1},
        "3rdPct": {"label": "3rd most common share", "pct_rank": 2},
        "Change": {"label": "Change", "needs_interval": True},
        "Top3": {"label": "Top 3 fields"},
        "Diversity": {"label": "Diversity"},
        "DiversityPew": {"label": "Pew diversity"},
    }
