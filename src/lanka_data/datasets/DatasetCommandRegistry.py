from lanka_data.api.command.fields.CensusDatasetRegistry import \
    CensusDatasetRegistry
from lanka_data.api.command.fields.ElectionDatasetRegistry import \
    ElectionDatasetRegistry
from lanka_data.api.command.fields.RegionTypeRegistry import RegionTypeRegistry
from lanka_data.datasets.dataset.custom.Census2001Dataset import \
    Census2001Dataset
from lanka_data.datasets.dataset.custom.Census2012Dataset import \
    Census2012Dataset
from lanka_data.datasets.dataset.custom.Census2024Dataset import \
    Census2024Dataset
from lanka_data.datasets.dataset.custom.ElectionDataset import ElectionDataset
from lanka_data.datasets.region.RegionTypeUtils import RegionTypeUtils


class DatasetCommandRegistry:
    @classmethod
    def register(cls):
        CensusDatasetRegistry.set_dataset_classes(
            [
                Census2001Dataset,
                Census2012Dataset,
                Census2024Dataset,
            ]
        )
        ElectionDatasetRegistry.set_dataset_class(ElectionDataset)
        RegionTypeRegistry.set_prefix_maps(RegionTypeUtils.get_prefix_maps())
