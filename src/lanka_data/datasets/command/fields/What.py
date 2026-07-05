from lanka_data.api.command.fields.What import What as APIWhat
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


class What(APIWhat):
    @classmethod
    def available_groups(cls):
        election = ElectionDataset.get_labels()
        return {
            "special": ["Empty"],
            "census": sorted(set(cls.census_values())),
            "election": election,
            "election_summary": cls.election_summary_values(election),
        }

    @staticmethod
    def election_summary_values(election):
        return [x + "Summary" for x in election]

    @classmethod
    def census_values(cls):
        values = Census2001Dataset.get_labels()
        values += Census2012Dataset.get_labels()
        values += Census2024Dataset.get_labels()
        return values
