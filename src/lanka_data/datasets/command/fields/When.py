from lanka_data.api.command.fields.When import When as APIWhen
from lanka_data.datasets.dataset.custom.Census2001Dataset import (
    Census2001Dataset,
)
from lanka_data.datasets.dataset.custom.Census2012Dataset import (
    Census2012Dataset,
)
from lanka_data.datasets.dataset.custom.Census2024Dataset import (
    Census2024Dataset,
)


class When(APIWhen):
    @classmethod
    def available_values(cls):
        values = []
        for dataset_cls in cls.dataset_classes():
            values.extend(dataset_cls.get_supported_whens())
        return sorted(set(values))

    @classmethod
    def dataset_classes(cls):
        return [
            Census2001Dataset,
            Census2012Dataset,
            Census2024Dataset,
        ]
