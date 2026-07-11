class HowRegistryBaseLabelsMixin:
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
