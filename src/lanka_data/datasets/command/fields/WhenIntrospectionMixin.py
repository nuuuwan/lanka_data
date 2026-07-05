from lanka_data.datasets.dataset.custom.Census2001Dataset import (
    Census2001Dataset,
)
from lanka_data.datasets.dataset.custom.Census2012Dataset import (
    Census2012Dataset,
)
from lanka_data.datasets.dataset.custom.Census2024Dataset import (
    Census2024Dataset,
)


class WhenIntrospectionMixin:
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

    @classmethod
    def available_intervals(cls):
        values = cls.available_values()
        return [
            f"{values[i]}-{values[j]}"
            for i in range(len(values))
            for j in range(i + 1, len(values))
        ]

    @classmethod
    def describe(cls):
        return dict(
            name="when",
            values=cls.available_values(),
            intervals=cls.available_intervals(),
            year_pattern=r"\d{4}",
            supports_interval=True,
        )
