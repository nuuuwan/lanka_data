from lanka_data.datasets.dataset.custom.Census2001Dataset import \
    Census2001Dataset
from lanka_data.datasets.dataset.custom.Census2012Dataset import \
    Census2012Dataset
from lanka_data.datasets.dataset.custom.Census2024Dataset import \
    Census2024Dataset
from lanka_data.datasets.dataset.custom.ElectionDataset import ElectionDataset
from lanka_data.datasets.dataset.custom.ElectionSummaryDataset import \
    ElectionSummaryDataset
from lanka_data.datasets.dataset.custom.RiversDataset import RiversDataset


class DatasetsReadmeConfigMixin:
    DATASETS = [
        {
            'class': Census2001Dataset,
            'name': 'Census of Population and Housing 2001',
            'description': 'Demographics data from the 2001 census',
        },
        {
            'class': Census2012Dataset,
            'name': 'Census of Population and Housing 2012',
            'description': 'Demographics data from the 2012 census',
        },
        {
            'class': Census2024Dataset,
            'name': 'Census of Population and Housing 2024',
            'description': 'Demographics data from the 2024 census',
        },
        {
            'class': ElectionDataset,
            'name': 'Election Data',
            'description':
                'Detailed election results by candidate/party and region',
        },
        {
            'class': ElectionSummaryDataset,
            'name': 'Election Summary Data',
            'description':
                'Election summary statistics (Valid, Rejected,'
                + ' Did Not Vote)',
        },
        {
            'class': RiversDataset,
            'name': 'Rivers Data',
            'description': 'River lengths and catchment areas by region',
        },
    ]