from datasets.region.RegionParserMixin import RegionParserMixin
from datasets.region.RegionRawDataMixin import RegionRawDataMixin
from api.utils_future import Log

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
