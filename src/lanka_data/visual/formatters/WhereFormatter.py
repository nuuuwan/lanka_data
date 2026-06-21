from lanka_data.region.RegionRawDataMixin import RegionRawDataMixin
from lanka_data.region.RegionTypeUtils import RegionTypeUtils


class WhereFormatter:
    def __init__(self, where_cmd):
        self.where_cmd = where_cmd

    def format(self) -> str:
        if ":" in self.where_cmd:
            parent_region_id, child_region_type = self.where_cmd.split(":", 1)
            child_region_long_name = RegionTypeUtils.get_long_name(
                child_region_type
            )
            parent_region_long_name = RegionRawDataMixin.get_full_name(
                parent_region_id
            )
            return f"{child_region_long_name}s in {parent_region_long_name}"

        if "@" in self.where_cmd:
            region_id, radius = self.where_cmd.split("@", 1)
            region_full_name = RegionRawDataMixin.get_full_name(region_id)
            return f"Within {radius}km of {region_full_name}"

        if "..." in self.where_cmd:
            region_id_from, region_id_to = self.where_cmd.split("...", 1)
            region_full_name_from = RegionRawDataMixin.get_full_name(
                region_id_from
            )
            region_full_name_to = RegionRawDataMixin.get_full_name(
                region_id_to
            )
            return f"From {region_full_name_from} to {region_full_name_to}"

        region_ids = self.where_cmd.split(",")
        region_full_names = [
            RegionRawDataMixin.get_full_name(region_id.strip())
            for region_id in region_ids
        ]
        return ", ".join(region_full_names)
