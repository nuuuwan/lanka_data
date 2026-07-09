import numpy as np


class ScatterPlotStats:
    @staticmethod
    def _pearson(xs, ys):
        matrix = np.corrcoef(xs, ys)
        r = matrix[0, 1]
        if np.isnan(r):
            return 0.0
        return float(r)

    @classmethod
    def fit(cls, xs, ys):
        if len(xs) < 2:
            return None
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        if xs.std() == 0 or ys.std() == 0:
            return None
        slope, intercept = np.polyfit(xs, ys, 1)
        r = cls._pearson(xs, ys)
        return dict(
            slope=float(slope),
            intercept=float(intercept),
            r=r,
            r2=r * r,
            n=len(xs),
        )

    @staticmethod
    def line_endpoints(fit, xs):
        x_min = min(xs)
        x_max = max(xs)
        return (
            [x_min, x_max],
            [
                fit["slope"] * x_min + fit["intercept"],
                fit["slope"] * x_max + fit["intercept"],
            ],
        )

    @staticmethod
    def text(fit):
        sign = "+" if fit["intercept"] >= 0 else "-"
        return "\n".join(
            [
                f"y = {fit['slope']:.2f}x {sign} "
                + f"{abs(fit['intercept']):.2f}",
                f"r = {fit['r']:.2f}",
                f"R\u00b2 = {fit['r2']:.2f}",
                f"n = {fit['n']}",
            ]
        )
