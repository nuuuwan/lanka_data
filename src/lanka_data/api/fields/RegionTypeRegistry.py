class RegionTypeRegistry:
    PREFIX_MAPS = {}
    EXAMPLES = [
        "LK",
        "LK:district",
        "LK:rivers",
        "LK-1,LK-2",
        "LK-1...LK-2",
        "LK-pre1959",
        "LK-1127025@20",
        "LK:rivers#10",
    ]

    @classmethod
    def set_prefix_maps(cls, prefix_maps):
        cls.PREFIX_MAPS = prefix_maps
