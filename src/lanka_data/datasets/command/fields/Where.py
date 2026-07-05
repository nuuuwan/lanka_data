from lanka_data.api.command.fields.Where import Where as APIWhere
from lanka_data.datasets.region.RegionTypeUtils import RegionTypeUtils


class Where(APIWhere):
    @classmethod
    def available_region_types(cls):
        values = set()
        for prefix_map in RegionTypeUtils._PREFIX_MAPS.values():
            values.update(prefix_map.values())
        return sorted(values)

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
