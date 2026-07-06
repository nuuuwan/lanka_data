from lanka_data.visual.plot.PlotLayoutError import PlotLayoutError


class PlotLayout:
    BASE_SIZE = 9

    def __init__(self, n_datasets):
        if n_datasets != 1:
            raise PlotLayoutError(n_datasets)
        self.n_datasets = n_datasets

    @property
    def n_rows(self):
        return 1

    @property
    def n_cols(self):
        return 1

    @property
    def figsize(self):
        return (self.BASE_SIZE, self.BASE_SIZE)

    def position(self, i_dataset):
        return (0, 0)
