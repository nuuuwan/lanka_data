import matplotlib.pyplot as plt
import numpy as np
import segno

from lanka_data.visual.plot.Style import Style


class QRCode:
    URL = Style.BRAND_URL
    SIZE_INCHES = 0.8
    BORDER = 2

    def __init__(self, visual):
        self.visual = visual

    def _matrix(self):
        qr = segno.make(self.URL, error="l")
        matrix = np.array([list(row) for row in qr.matrix], dtype=float)
        return np.pad(matrix, self.BORDER, constant_values=0)

    def _axes(self, fig):
        fig_w, fig_h = fig.get_size_inches()
        width = self.SIZE_INCHES / fig_w
        height = self.SIZE_INCHES / fig_h
        left = 1 - Style.MARGIN - width
        bottom = (Style.FOOTER_HEIGHT - height) / 2
        return fig.add_axes(
            [left, max(bottom, 0.005), width, height], zorder=10001
        )

    def draw(self):
        fig = plt.gcf()
        ax = self._axes(fig)
        ax.imshow(self._matrix(), cmap="gray_r", interpolation="nearest")
        ax.axis("off")
