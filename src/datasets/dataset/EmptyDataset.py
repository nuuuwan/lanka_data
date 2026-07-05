from api.data.DataSource import DataSource
from api.dataset.Dataset import Dataset


class EmptyDataset(Dataset):
    def __init__(self, region_data_list):
        Dataset.__init__(self)
        self.region_data_list = region_data_list

    def get_sources(self):
        return [
            DataSource(
                name="Survey Department of Sri Lanka",
                url="https://survey.gov.lk/",
            )
        ]

    def get_data_table(self):
        return self.region_data_list

    def has_values(self):
        return False

    def get_year(self):
        return ""
