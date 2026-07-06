class BivariatePalette:
    NEUTRAL = "#dddddd"

    GRID_3 = [
        ["#e8e8e8", "#b5c0da", "#6c83b5"],
        ["#b8d6be", "#90b2b3", "#567994"],
        ["#73ae80", "#5a9178", "#2a5a5b"],
    ]

    GRID_2 = [
        ["#e8e8e8", "#6c83b5"],
        ["#73ae80", "#2a5a5b"],
    ]

    def __init__(self, n_bins):
        self.n_bins = n_bins
        self.grid = self.GRID_2 if n_bins == 2 else self.GRID_3

    def color(self, x_bin, y_bin):
        if x_bin is None or y_bin is None:
            return self.NEUTRAL
        x_bin = min(max(x_bin, 0), self.n_bins - 1)
        y_bin = min(max(y_bin, 0), self.n_bins - 1)
        return self.grid[y_bin][x_bin]
