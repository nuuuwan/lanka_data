from lanka_data.dataset.Dataset import Dataset
from lanka_data.dataset.RegionValueDataset import RegionValueDataset


class DiffDataset(Dataset):
    def __init__(
        self, dataset1: RegionValueDataset, dataset2: RegionValueDataset
    ):
        Dataset.__init__(self)
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __str__(self):
        return f"DiffDataset({self.dataset1} - {self.dataset2})"
