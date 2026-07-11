import warnings

from lanka_data.visual.HowParam import HowParam

warnings.warn(
    "HOW_PARAMS_DATA is deprecated. Use HowParam.list() instead.",
    DeprecationWarning,
    stacklevel=2,
)

HOW_PARAMS = HowParam.list()
