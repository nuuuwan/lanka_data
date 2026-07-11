from lanka_data.datasets.region.RegionParserMixin.RegionParserMixin import \
    RegionParserMixin
from lanka_data.datasets.region.RegionRawDataMixin.RegionRawDataMixin import \
    RegionRawDataMixin
from utils_future import Log

log = Log("RegionLoadersMixin")


class RegionLoadersMixin(RegionParserMixin, RegionRawDataMixin):

    @classmethod
    def from_command(cls, command):
        raw_regions, region_year = cls.parse(command.where_cmd)

        if len(raw_regions) == 0:
            raise ValueError(f"No regions found for token: {command}")

        return cls(
            raw_regions,
            region_year,
        )
