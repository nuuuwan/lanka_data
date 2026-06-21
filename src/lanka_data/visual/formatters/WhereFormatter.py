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
            return f"{child_region_long_name}s in {parent_region_id}"
        else:
            region_id = self.where_cmd
            return f"{region_id}"
