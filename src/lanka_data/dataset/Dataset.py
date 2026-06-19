from utils_future import Log

log = Log("Dataset")


class Dataset:

    def __str__(self):
        raise NotImplementedError("Subclasses must implement __str__()")

    def __repr__(self):
        return self.__str__()
