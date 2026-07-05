from lanka_data.datasets.region.RegionTypeUtils import RegionTypeUtils


class WhereIntrospectionMixin:
    @classmethod
    def available_region_types(cls):
        values = set()
        for prefix_map in RegionTypeUtils._PREFIX_MAPS.values():
            values.update(prefix_map.values())
        return sorted(values)

    @classmethod
    def available_operators(cls):
        return ["single", "comma", "range", "history", "zoom", "child_type"]

    @classmethod
    def available_examples(cls):
        return [
            "LK",
            "LK:district",
            "LK-1,LK-2",
            "LK-1...LK-2",
            "LK-pre1959",
            "LK-1127025@20",
        ]

    @classmethod
    def available_values(cls):
        return cls.available_examples()

    @classmethod
    def describe(cls):
        return dict(
            name="where",
            values=cls.available_values(),
            region_types=cls.available_region_types(),
            operators=cls.available_operators(),
            token_pattern=r"[A-Za-z0-9:,@.\-]+",
        )
