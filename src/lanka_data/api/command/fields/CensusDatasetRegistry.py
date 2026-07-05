from lanka_data.datasets.dataset.custom.Census2001Dataset import (
    Census2001Dataset,
)
from lanka_data.datasets.dataset.custom.Census2012Dataset import (
    Census2012Dataset,
)
from lanka_data.datasets.dataset.custom.Census2024Dataset import (
    Census2024Dataset,
)


class CensusDatasetRegistry:
    DATASET_CLASSES = [
        Census2001Dataset,
        Census2012Dataset,
        Census2024Dataset,
    ]
