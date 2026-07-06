from lanka_data.visual.plot.PlotLayoutError import PlotLayoutError


class PlotLayout:
    BASE_SIZE = 9
    COUNT_TO_GRID = {
        1: (1, 1),
        2: (1, 2),
        3: (1, 3),
        4: (2, 2),
    }

    def __init__(self, n_datasets):
        if n_datasets not in self.COUNT_TO_GRID:
            raise PlotLayoutError(n_datasets)
        self.n_datasets = n_datasets

    @property
    def n_rows(self):
        return self.COUNT_TO_GRID[self.n_datasets][0]

    @property
    def n_cols(self):
        return self.COUNT_TO_GRID[self.n_datasets][1]

    @property
    def figsize(self):
        return (self.BASE_SIZE * self.n_cols, self.BASE_SIZE * self.n_rows)

    def position(self, i_dataset):
        return (i_dataset // self.n_cols, i_dataset % self.n_cols)
