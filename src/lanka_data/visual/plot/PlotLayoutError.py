class PlotLayoutError(ValueError):
    def __init__(self, n_datasets):
        super().__init__(f"Cannot layout {n_datasets} datasets (max 4).")
        self.n_datasets = n_datasets
