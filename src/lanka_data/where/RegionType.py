from functools import cache


class RegionType:
    @classmethod
    @cache
    def get_region_type(cls, region_id: str) -> str:
        region_type = None
        id_len = len(region_id)
        if region_id.startswith("LK"):
            region_type = {
                2: "country",
                4: "province",
                5: "district",
                7: "dsd",
                10: "gnd",
                #
                9: "lg",
            }.get(id_len)

        if region_id.startswith("EC-"):
            region_type = {
                5: "ed",
                6: "pd",
            }.get(id_len)

        if region_type is not None:
            return region_type

        raise ValueError(f"Invalid region ID format: {region_id}")
