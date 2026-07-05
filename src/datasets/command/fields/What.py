from dataclasses import dataclass

from lanka_data.command.UnknownWhatError import UnknownWhatError
from lanka_data.dataset.custom.Census2001Dataset import Census2001Dataset
from lanka_data.dataset.custom.Census2012Dataset import Census2012Dataset
from lanka_data.dataset.custom.Census2024Dataset import Census2024Dataset
from lanka_data.dataset.custom.ElectionDataset import ElectionDataset


@dataclass(frozen=True)
class What:
    value: str

    def __post_init__(self):
        if self.value == "Help":
            return
        if self.value not in self.known_values():
            raise UnknownWhatError(
                f"Unknown what: {self.value}",
                self.value,
                self.suggestions(self.value),
            )

    @classmethod
    def known_values(cls):
        return sorted(
            set(["Empty"] + cls.election_values() + cls.census_values())
        )

    @classmethod
    def election_values(cls):
        election = ElectionDataset.get_labels()
        return election + [x + "Summary" for x in election]

    @classmethod
    def census_values(cls):
        values = Census2001Dataset.get_labels()
        values += Census2012Dataset.get_labels()
        values += Census2024Dataset.get_labels()
        return values

    @classmethod
    def suggestions(cls, value):
        value_lower = value.lower()
        return [x for x in cls.known_values() if value_lower in x.lower()][:5]

    def __str__(self):
        return self.value
