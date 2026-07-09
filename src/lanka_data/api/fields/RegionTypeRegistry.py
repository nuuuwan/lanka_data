class RegionTypeRegistry:
    PREFIX_MAPS = {}
    EXAMPLES = [
        "LK",
        "LK:province",
        "LK:district",
        "LK:dsd",
        "LK:gnd",
        "LK:rivers",
        "LK-1,LK-2",
        "LK-1,LK-2,LK-3",
        "LK-1...LK-2",
        "LK-11...LK-13",
        "LK-pre1959",
        "LK-pre1959:district",
        "LK-1127025@20",
        "LK:district#5",
        "LK:rivers#10",
    ]

    @classmethod
    def set_prefix_maps(cls, prefix_maps):
        cls.PREFIX_MAPS = prefix_maps
