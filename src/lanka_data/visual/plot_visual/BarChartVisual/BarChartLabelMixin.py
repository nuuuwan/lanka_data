class BarChartLabelMixin:
    _CHAR_W_RATIO = 1
    _LINE_H_RATIO = 1
    _MIN_FONT = 2
    _MAX_FONT = 24

    @classmethod
    def _fit_fontsize(cls, bar_h_px, bar_w_px, n_chars, n_lines, dpi):
        pt_per_px = 72 / dpi
        return min(
            cls._MAX_FONT,
            bar_w_px * pt_per_px / max(n_chars * cls._CHAR_W_RATIO, 1),
            bar_h_px * pt_per_px / max(n_lines * cls._LINE_H_RATIO, 1),
        )

    @staticmethod
    def _compute_pct_label(subregions, is_change, totals, cat, bar):
        height = bar.get_height()
        idx = round(bar.get_x() + bar.get_width() / 2)
        if is_change:
            pct_val = subregions[idx]["pct_values"].get(cat, 0)
            return f"{pct_val * 100:+.1f}pp"
        total = totals.get(idx, 1)
        return f"{abs(height) / total:.1%}"

    def _resolve_bar_label(
        self, bar_h_px, bar_w_px, abs_label, pct_label, dpi
    ):
        full_text = f"{abs_label}\n{pct_label}"
        max_line = max(len(abs_label), len(pct_label))
        fontsize = self._fit_fontsize(bar_h_px, bar_w_px, max_line, 2, dpi)
        if fontsize < self._MIN_FONT:
            fontsize = self._fit_fontsize(
                bar_h_px, bar_w_px, len(pct_label), 1, dpi
            )
            if fontsize < self._MIN_FONT:
                return None, None
            return pct_label, fontsize
        return full_text, fontsize

    def _add_bar_labels(self, ax, subregions):
        is_change = self._is_change_chart(subregions)
        totals = {
            i: sum(abs(v) for v in s["values"].values()) or 1
            for i, s in enumerate(subregions)
        }
        for container in ax.containers:
            cat = container.get_label()
            for bar in container:
                height = bar.get_height()
                if height == 0:
                    continue
                tf = ax.transData.transform
                p0 = tf((bar.get_x(), bar.get_y()))
                p1 = tf((bar.get_x() + bar.get_width(), bar.get_y() + height))
                bar_h_px = abs(p1[1] - p0[1])
                bar_w_px = abs(p1[0] - p0[0])
                abs_label = self._format_millions(height, None)
                pct_label = self._compute_pct_label(
                    subregions, is_change, totals, cat, bar
                )
                text, fontsize = self._resolve_bar_label(
                    bar_h_px,
                    bar_w_px,
                    abs_label,
                    pct_label,
                    ax.get_figure().dpi,
                )
                if text is None:
                    continue
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
