from lanka_data.api.fields.Where import Where
from lanka_data.datasets.region.RegionRawDataMixin.RegionRawDataMixin import RegionRawDataMixin
from lanka_data.datasets.region.RegionTypeUtils import RegionTypeUtils


class WhereFormatter:
    def __init__(self, where_cmd):
        self.where_cmd = Where.strip_top(where_cmd)

    def format_regions(self, region_ids):
        region_ids = [id.split("-pre")[0] for id in region_ids]
        region_full_names = [
            RegionRawDataMixin.get_full_name(region_id.strip())
            for region_id in region_ids
        ]
        if len(region_full_names) == 1:
            return region_full_names[0]
        elif len(region_full_names) == 2:
            return f"{region_full_names[0]} & {region_full_names[1]}"
        else:
            return (
                ", ".join(region_full_names[:-1])
                + f", & {region_full_names[-1]}"
            )

    def parse_parent_part(self, parent_part):
        return [id.strip() for id in parent_part.split(",") if id.strip()]

    def _format_without_parent(self) -> str:
        if "..." in self.where_cmd:
            region_id_from, region_id_to = self.where_cmd.split("...", 1)
            return (
                f"From {self.format_regions([region_id_from])}"
                f" to {self.format_regions([region_id_to])}"
            )
        return self.format_regions(self.where_cmd.split(","))

    def format(self) -> str:
        if ":" in self.where_cmd:
            parent_part, child_region_type = self.where_cmd.split(":", 1)
            child_region_long_name = RegionTypeUtils.get_long_name_plural(
                child_region_type
            )
            parent_region_ids = self.parse_parent_part(parent_part)
            parent_regions_long_name = self.format_regions(parent_region_ids)
            return f"{child_region_long_name} in {parent_regions_long_name}"

        if "@" in self.where_cmd:
            region_id, radius = self.where_cmd.split("@", 1)
            region_full_name = self.format_regions([region_id])
            return f"Within {radius}km of {region_full_name}"

        return self._format_without_parent()
