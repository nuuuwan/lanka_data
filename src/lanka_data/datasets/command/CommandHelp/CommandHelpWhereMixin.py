from lanka_data.datasets.region.RegionTypeUtils import RegionTypeUtils

WHERE_VALUES = [
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
WHERE_OPERATORS = [
    "single",
    "comma",
    "range",
    "history",
    "zoom",
    "child_type",
    "top",
]
TOKEN_PATTERN = r"[A-Za-z0-9:,@.#\-]+"


class CommandHelpWhereMixin:
    @staticmethod
    def get_where_region_types():
        prefix_maps = RegionTypeUtils.get_prefix_maps()
        types = {
            region_type
            for id_map in prefix_maps.values()
            for region_type in id_map.values()
        }
        return sorted(types)

    @staticmethod
    def get_where_help():
        return {
            "values": WHERE_VALUES,
            "region_types": (CommandHelpWhereMixin.get_where_region_types()),
            "operators": WHERE_OPERATORS,
            "token_pattern": TOKEN_PATTERN,
        }
