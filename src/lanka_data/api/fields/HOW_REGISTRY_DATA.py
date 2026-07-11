BASE_LABELS = {
    "JSON": "JSON (Structured data format)",
    "CSV": "CSV (Comma-separated values)",
    "TSV": "TSV (Tab-separated values)",
    "GeoJSON": "GeoJSON (Geographic JSON format)",
    "Parquet": "Parquet (Columnar data format)",
    "ChartSpec": "Chart Spec (Vega-Lite specification)",
    "Map": "Map (Geographic visualization by region)",
    "Cartogram": (
        "Cartogram (Area-based map where region size "
        "represents population)"
    ),
    "HexMap": (
        "HexMap (Geographic map with hexagonal tiles, "
        "sized by population)"
    ),
    "UnitHexMap": (
        "UnitHexMap (Geographic map with one hexagon "
        "per region)"
    ),
    "SquareMap": (
        "SquareMap (Geographic map with square tiles, "
        "sized by population)"
    ),
    "UnitSquareMap": (
        "UnitSquareMap (Geographic map with one square "
        "per region)"
    ),
    "TriangleMap": (
        "TriangleMap (Geographic map with triangular tiles, "
        "sized by population)"
    ),
    "UnitTriangleMap": (
        "UnitTriangleMap (Geographic map with one triangle "
        "per region)"
    ),
    "BubbleMap": (
        "BubbleMap (Geographic map with bubbles "
        "sized by population)"
    ),
    "BarChart": "Bar Chart (Compares values across categories)",
    "StackedBarChart": (
        "Stacked Bar Chart "
        "(Shows composition across categories)"
    ),
    "PieChart": (
        "Pie Chart (Visualizes proportions of a whole)"
    ),
    "BumpChart": (
        "Bump Chart (Tracks ranking changes over time)"
    ),
    "TreeMap": (
        "Tree Map (Displays hierarchical data "
        "as nested rectangles)"
    ),
    "Histogram": "Histogram (Shows distribution of values)",
    "ScatterPlot": (
        "Scatter Plot (Shows relationship "
        "between two variables)"
    ),
    "BivariateMap": (
        "Bivariate Map (Combines two variables "
        "using color combinations)"
    ),
    "QuadrantChart": (
        "Quadrant Chart (Divides space into four quadrants "
        "for comparison)"
    ),
    "LineChart": (
        "Line Chart (Displays trends over time "
        "or ordered categories)"
    ),
    "None": "None (No visualization output)",
}

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
