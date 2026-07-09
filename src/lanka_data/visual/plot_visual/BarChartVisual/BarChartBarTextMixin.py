class BarChartBarTextMixin:
    _CHAR_W_RATIO = 1
    _LINE_H_RATIO = 2

    @classmethod
    def _fit_fontsize(cls, bar_h_px, bar_w_px, n_chars, n_lines, dpi):
        pt_per_px = 72 / dpi
        return min(
            bar_w_px * pt_per_px / max(n_chars * cls._CHAR_W_RATIO, 1),
            bar_h_px * pt_per_px / max(n_lines * cls._LINE_H_RATIO, 1),
        )

    def _resolve_bar_label(
        self, bar_h_px, bar_w_px, abs_label, pct_label, dpi
    ):
        if pct_label is None:
            fontsize = self._fit_fontsize(
                bar_h_px, bar_w_px, len(abs_label), 1, dpi
            )
            return abs_label, fontsize
        full_text = f"{abs_label}\n{pct_label}"
        max_line = max(len(abs_label), len(pct_label))
        fontsize = self._fit_fontsize(bar_h_px, bar_w_px, max_line, 2, dpi)
        return full_text, fontsize

    @staticmethod
    def _bar_pixel_size(ax, bar, height):
        tf = ax.transData.transform
        p0 = tf((bar.get_x(), bar.get_y()))
        p1 = tf((bar.get_x() + bar.get_width(), bar.get_y() + height))
        return abs(p1[1] - p0[1]), abs(p1[0] - p0[0])

    @staticmethod
    def _draw_bar_text(ax, bar, height, text, fontsize):
        fc = bar.get_facecolor()
        lum = 0.299 * fc[0] + 0.587 * fc[1] + 0.114 * fc[2]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_y() + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color="#333" if lum > 0.5 else "#eee",
            clip_on=True,
        )
