class HowRegistryMixin:
    BASE_LABELS = {
        "JSON": None,
        "CSV": None,
        "TSV": None,
        "Table": None,
        "Map": None,
        "Cartogram": "Cartogram (Population based)",
        "HexMap": "HexMap (Population based)",
        "BubbleMap": "BubbleMap (Population based)",
        "BarChart": "Bar Chart",
        "PieChart": "Pie Chart",
        "BumpChart": "Bump Chart",
        "None": None,
    }
    INTERVAL_BASES = {"BumpChart"}
    CATEGORY_BASES = {"Map", "Cartogram", "HexMap", "BubbleMap", "None"}
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
        "Diversity": {"label": "Diversity"},
        "DiversityPew": {"label": "Pew diversity"},
        "Segregation": {"label": "Segregation"},
    }
