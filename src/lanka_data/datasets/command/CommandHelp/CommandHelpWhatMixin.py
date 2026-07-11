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
from lanka_data.datasets.dataset.custom.ElectionSummaryDataset import (
    ElectionSummaryDataset,
)
from lanka_data.datasets.dataset.custom.RiversDataset import RiversDataset

CENSUS_CLASSES = [
    Census2001Dataset,
    Census2012Dataset,
    Census2024Dataset,
]


class CommandHelpWhatMixin:
    @staticmethod
    def _census_whats():
        labels = itertools.chain.from_iterable(
            cls.get_labels() for cls in CENSUS_CLASSES
        )
        return sorted(set(labels))

    @staticmethod
    def _election_summary_whats():
        return sorted(
            label + "Summary" for label in ElectionSummaryDataset.get_labels()
        )

    @staticmethod
    def get_what_groups():
        return {
            "special": ["Empty"],
            "census": CommandHelpWhatMixin._census_whats(),
            "election": sorted(ElectionDataset.get_labels()),
            "election_summary": (
                CommandHelpWhatMixin._election_summary_whats()
            ),
            "rivers": sorted(RiversDataset.get_labels()),
        }

    @staticmethod
    def get_what_help():
        groups = CommandHelpWhatMixin.get_what_groups()
        values = sorted(set(itertools.chain.from_iterable(groups.values())))
        return {
            "values": values,
            "groups": groups,
            "count": len(values),
        }
