class HistogramData:
    @staticmethod
    def _edges(lo, hi, n_bins):
        if hi <= lo:
            hi = lo + 1
        width = (hi - lo) / n_bins
        return [lo + i * width for i in range(n_bins + 1)]

    @staticmethod
    def _index(value, lo, hi, n_bins):
        if hi <= lo:
            return 0
        idx = int((value - lo) / (hi - lo) * n_bins)
        return min(max(idx, 0), n_bins - 1)

    @classmethod
    def bins(cls, values, n_bins):
        values = [v for v in values if v is not None]
        if not values or n_bins < 1:
            return [], []
        lo, hi = min(values), max(values)
        edges = cls._edges(lo, hi, n_bins)
        counts = [0] * n_bins
        for value in values:
            counts[cls._index(value, lo, hi, n_bins)] += 1
        return edges, counts
