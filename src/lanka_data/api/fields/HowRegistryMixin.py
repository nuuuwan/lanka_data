import warnings

from lanka_data.api.fields.HOW_REGISTRY_DATA import (
    BASE_LABELS,
    CATEGORY_BASES,
    INTERVAL_BASES,
    MODIFIERS,
    PAIR_CATEGORY_BASES,
    SERIES_BASES,
)

warnings.warn(
    "HowRegistryMixin is deprecated. Use individual visual classes' "
    "get_description() methods and HowParam.list() instead.",
    DeprecationWarning,
    stacklevel=2,
)


class HowRegistryMixin:
    BASE_LABELS = BASE_LABELS
    INTERVAL_BASES = INTERVAL_BASES
    SERIES_BASES = SERIES_BASES
    CATEGORY_BASES = CATEGORY_BASES
    PAIR_CATEGORY_BASES = PAIR_CATEGORY_BASES
    MODIFIERS = MODIFIERS
