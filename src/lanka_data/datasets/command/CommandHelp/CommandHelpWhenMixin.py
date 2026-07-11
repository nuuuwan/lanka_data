import itertools

from lanka_data.datasets.dataset.custom.Census2001Dataset import (
    Census2001Dataset,
)
from lanka_data.datasets.dataset.custom.Census2012Dataset import (
    Census2012Dataset,
)
from lanka_data.datasets.dataset.custom.Census2024Dataset import (
    Census2024Dataset,
)
from lanka_data.datasets.dataset.custom.ElectionDataset import ElectionDataset
from lanka_data.datasets.dataset.custom.RiversDataset import RiversDataset

CENSUS_CLASSES = [
    Census2001Dataset,
    Census2012Dataset,
    Census2024Dataset,
]
YEAR_PATTERN = r"\d{4}"


class CommandHelpWhenMixin:
    @staticmethod
    def _census_years():
        return itertools.chain.from_iterable(
            cls.get_supported_whens() for cls in CENSUS_CLASSES
        )

    @staticmethod
    def _election_years():
        return itertools.chain.from_iterable(
            ElectionDataset.get_label_to_years().values()
        )

    @staticmethod
    def get_when_years():
        years = set(CommandHelpWhenMixin._census_years())
        years |= set(CommandHelpWhenMixin._election_years())
        years |= set(RiversDataset.get_supported_whens())
        return sorted(years)

    @staticmethod
    def _intervals(years):
        return [f"{a}-{b}" for a, b in itertools.combinations(years, 2)]

    @staticmethod
    def get_when_help():
        years = CommandHelpWhenMixin.get_when_years()
        return {
            "values": years,
            "intervals": CommandHelpWhenMixin._intervals(years),
            "year_pattern": YEAR_PATTERN,
            "supports_interval": True,
        }
