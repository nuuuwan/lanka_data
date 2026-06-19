from utils_future import Log

log = Log("Dataset")


class Dataset:

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__str__()

    def clean_data_row(self, data: dict) -> dict:
        raise NotImplementedError("Subclasses must implement this method.")

    def get_source_data_table(self) -> list[dict]:
        raise NotImplementedError("Subclasses must implement this method.")

    def get_complete_data_table(self) -> list[dict]:
        source_data_table = self.get_source_data_table()
        cleaned_data_table = [
            self.clean_data_row(row) for row in source_data_table
        ]
        return cleaned_data_table
