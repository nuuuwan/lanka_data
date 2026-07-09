class HowRegistryMixin:
    BASE_LABELS = {
        "JSON": None,
        "CSV": None,
        "TSV": None,
        "GeoJSON": None,
        "Parquet": None,
        "ChartSpec": "Chart Spec",
        "Map": None,
        "Cartogram": "Cartogram (Population based)",
        "HexMap": "HexMap (Population based)",
        "BubbleMap": "BubbleMap (Population based)",
        "BarChart": "Bar Chart",
        "StackedBarChart": "Stacked Bar Chart",
        "PieChart": "Pie Chart",
        "BumpChart": "Bump Chart",
        "TreeMap": "Tree Map",
        "Histogram": "Histogram",
        "ScatterPlot": "Scatter Plot",
        "BivariateMap": "Bivariate Map",
        "QuadrantMap": "Quadrant Map",
        "LineChart": "Line Chart",
        "MapAnimation": "Animated Map",
        "CartogramAnimation": "Animated Cartogram (Population based)",
        "HexMapAnimation": "Animated HexMap (Population based)",
        "BubbleMapAnimation": "Animated BubbleMap (Population based)",
        "None": None,
    }
    ANIMATION_BASE_TO_FRAME_BASE = {
        "MapAnimation": "Map",
        "CartogramAnimation": "Cartogram",
        "HexMapAnimation": "HexMap",
        "BubbleMapAnimation": "BubbleMap",
    }
    INTERVAL_BASES = {"BumpChart", "LineChart"} | set(
        ANIMATION_BASE_TO_FRAME_BASE
    )
    SERIES_BASES = {"LineChart"}
    CATEGORY_BASES = {
        "Map",
        "Cartogram",
        "HexMap",
        "BubbleMap",
        "None",
    } | set(ANIMATION_BASE_TO_FRAME_BASE)
    PAIR_CATEGORY_BASES = {"BivariateMap", "QuadrantMap"}
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
