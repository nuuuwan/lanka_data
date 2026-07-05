from lanka_data.api.command.fields.RegionTypeRegistry import (
    RegionTypeRegistry,
)
from lanka_data.api.command.fields.WhatRegistry import WhatRegistry
from lanka_data.api.command.fields.WhatWhenRegistry import WhatWhenRegistry
from lanka_data.api.command.fields.WhenRegistry import WhenRegistry
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
from lanka_data.datasets.region.RegionTypeUtils import RegionTypeUtils


class DatasetCommandRegistry:
    CENSUS_DATASET_CLASSES = [
        Census2001Dataset,
        Census2012Dataset,
        Census2024Dataset,
    ]

    @classmethod
    def register(cls):
        WhatRegistry.set_group_providers(
            {
                "census": cls.census_labels,
                "election": ElectionDataset.get_labels,
                "election_summary": cls.election_summary_labels,
            }
        )
        WhenRegistry.set_value_providers([cls.census_whens])
        WhatWhenRegistry.set_pair_providers(
            [cls.census_pairs, cls.election_pairs]
        )
        RegionTypeRegistry.set_prefix_maps(RegionTypeUtils.get_prefix_maps())

    @classmethod
    def census_labels(cls):
        labels = []
        for dataset_cls in cls.CENSUS_DATASET_CLASSES:
            labels.extend(dataset_cls.get_labels())
        return labels

    @classmethod
    def census_whens(cls):
        whens = []
        for dataset_cls in cls.CENSUS_DATASET_CLASSES:
            whens.extend(dataset_cls.get_supported_whens())
        return whens

    @classmethod
    def election_summary_labels(cls):
        return [x + "Summary" for x in ElectionDataset.get_labels()]

    @classmethod
    def census_pairs(cls, when_values):
        pairs = []
        for dataset_cls in cls.CENSUS_DATASET_CLASSES:
            for label in dataset_cls.get_labels():
                for when in dataset_cls.get_supported_whens():
                    pairs.append((label, when))
        return pairs

    @classmethod
    def election_pairs(cls, when_values):
        pairs = []
        for label in ElectionDataset.get_labels():
            for when in when_values:
                pairs.append((label, when))
                pairs.append((label + "Summary", when))
        return pairs


DatasetCommandRegistry.register()
