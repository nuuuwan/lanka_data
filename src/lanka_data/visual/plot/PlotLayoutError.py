class PlotLayoutError(ValueError):
    def __init__(self, n_datasets):
        super().__init__(
            f"Cannot layout {n_datasets} datasets (expected exactly 1)."
        )
        self.n_datasets = n_datasets
