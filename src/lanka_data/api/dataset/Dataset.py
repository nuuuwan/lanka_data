from abc import ABC, abstractmethod

from lanka_data.api.utils_future import Log

log = Log("Dataset")


class Dataset(ABC):

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__str__()

    def is_diff(self):
        return False

    @abstractmethod
    def get_sources(self):
        pass

    @abstractmethod
    def get_data_table(self) -> list[dict]:
        pass

    @abstractmethod
    def has_values(self) -> bool:
        pass
