from abc import ABC, abstractmethod

from utils_future import Log

log = Log("Dataset")


class Dataset(ABC):

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    def get_source_data_table(self):
        pass

    @abstractmethod
    def clean_data_row(self, row):
        pass

    @abstractmethod
    def get_source_info_list(self):
        pass

    def get_complete_data_table(self) -> list[dict]:
        source_data_table = self.get_source_data_table()
        cleaned_data_table = [
            self.clean_data_row(row) for row in source_data_table
        ]
        return cleaned_data_table
