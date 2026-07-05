from dataclasses import dataclass

from lanka_data.api.command.UnknownWhatError import UnknownWhatError
from lanka_data.api.command.fields.CensusDatasetRegistry import (
    CensusDatasetRegistry,
)
from lanka_data.api.command.fields.WhatIntrospectionMixin import (
    WhatIntrospectionMixin,
)
from lanka_data.datasets.dataset.custom.ElectionDataset import ElectionDataset


@dataclass(frozen=True)
class What(WhatIntrospectionMixin):
    value: str

    VALUE_GROUPS = {"special": ["Empty"]}

    @classmethod
    def available_groups(cls):
        election = ElectionDataset.get_labels()
        return {
            "special": cls.VALUE_GROUPS["special"],
            "census": sorted(set(cls.census_values())),
            "election": election,
            "election_summary": cls.election_summary_values(election),
        }

    @staticmethod
    def election_summary_values(election):
        return [x + "Summary" for x in election]

    @classmethod
    def census_values(cls):
        values = []
        for dataset_cls in CensusDatasetRegistry.dataset_classes():
            values += dataset_cls.get_labels()
        return values

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
        return cls.available_values()

    @classmethod
    def suggestions(cls, value):
        value_lower = value.lower()
        return [x for x in cls.known_values() if value_lower in x.lower()][:5]

    def __str__(self):
        return self.value
