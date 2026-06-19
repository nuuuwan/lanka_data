from lanka_data.dataset.Dataset import Dataset
from lanka_data.dataset.ValueDataset import ValueDataset


class DiffDataset(Dataset):
    def __init__(self, dataset1: ValueDataset, dataset2: ValueDataset):
        Dataset.__init__(self)
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __str__(self):
        return f"DiffDataset({self.dataset1} - {self.dataset2})"
