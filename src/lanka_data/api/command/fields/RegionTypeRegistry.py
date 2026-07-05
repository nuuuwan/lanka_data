from lanka_data.datasets.region.RegionTypeUtils import RegionTypeUtils


class RegionTypeRegistry:
    PREFIX_MAPS = RegionTypeUtils._PREFIX_MAPS
    EXAMPLES = [
        "LK",
        "LK:district",
        "LK-1,LK-2",
        "LK-1...LK-2",
        "LK-pre1959",
        "LK-1127025@20",
    ]
