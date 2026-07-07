class BivariatePalette:
    NEUTRAL = "#dddddd"

    GRID_3 = [
        ["#e8e8e8", "#8aabd3", "#2d6ebe"],
        ["#d89292", "#896e90", "#3a4b8e"],
        ["#c83c3c", "#87324e", "#46285f"],
    ]

    GRID_2 = [
        ["#e8e8e8", "#2d6ebe"],
        ["#c83c3c", "#46285f"],
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
