from lanka_data.datasets.dataset.custom.Census2001Dataset import \
    Census2001Dataset
from lanka_data.datasets.dataset.custom.Census2012Dataset import \
    Census2012Dataset
from lanka_data.datasets.dataset.custom.Census2024Dataset import \
    Census2024Dataset

CENSUS_CLASSES = [
    Census2001Dataset,
    Census2012Dataset,
    Census2024Dataset,
]
YEAR_PATTERN = r"\d{4}"


class CommandHelpWhenMixin:

    @staticmethod
    def get_when_help():
        return "Specify the year or date for which you want to retrieve data. If the exact time is not available, the closest available data will be used."
