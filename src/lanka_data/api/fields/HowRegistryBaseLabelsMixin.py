import warnings

from lanka_data.api.fields.HOW_REGISTRY_DATA import BASE_LABELS

warnings.warn(
    "HowRegistryBaseLabelsMixin is deprecated. Use "
    "lanka_data.api.fields.HOW_REGISTRY_DATA.BASE_LABELS instead.",
    DeprecationWarning,
    stacklevel=2,
)


class HowRegistryBaseLabelsMixin:
    BASE_LABELS = BASE_LABELS
