from lanka_data.region.RegionParserMixin import RegionParserMixin
from lanka_data.region.RegionRawDataMixin import RegionRawDataMixin
from utils_future import Log

log = Log("RegionLoadersMixin")


class RegionLoadersMixin(RegionParserMixin, RegionRawDataMixin):
    @classmethod
    def get_description_from_region_year(cls, region_year):
        return f" (pre-{region_year} Map)" if region_year != "Current" else ""

    @classmethod
    def from_command(cls, command):
        raw_regions, region_year, description = cls.parse(command.where_cmd)

        if len(raw_regions) == 0:
            raise ValueError(f"No regions found for token: {command}")

        return cls(
            raw_regions,
            region_year,
            description,
        )
