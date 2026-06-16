import colorsys

from lanka_data.api.how.map.OrderColorUtils import OrderColorUtils


class RegionColorUtils:
    @staticmethod
    def _colors_no_values(result_data):
        data_list = result_data["data_list"]
        region_color_map = {
            data["region_id"]: OrderColorUtils.get_color_for_label(
                data["region_id"]
            )
            for data in data_list
        }
        value_to_color = None

        return region_color_map, value_to_color

    @staticmethod
    def _colors_values_key(result_data, how):
        data_list = result_data["data_list"]
        pct_values = [data["pct_values"][how.params] for data in data_list]
        value_to_rank = {v: r for r, v in enumerate(sorted(set(pct_values)))}
        n = len(value_to_rank)
        value_to_color, region_color_map = {}, {}
        for data in data_list:
            value = data["pct_values"][how.params]
            rank = value_to_rank[value]
            color = colorsys.hls_to_rgb((1 - rank / (n - 1)) * 0.67, 0.5, 1.0)
            value_to_color[value] = color
            region_color_map[data["region_id"]] = color
        return region_color_map, value_to_color

    @staticmethod
    def _colors_with_values(result_data, how, what):
        func_key_getter = OrderColorUtils._func_key_getter(how, what)
        if func_key_getter:
            return OrderColorUtils.get_order_color_map(
                result_data, how, what, func_key_getter
            )
        return RegionColorUtils._colors_values_key(result_data, how)

    @staticmethod
    def _compute_diversity(pct_values):
        values_only = pct_values.values()
        # normalised Herfindahl-Simpson concentration measure
        return (
            10
            * (1 - sum(s**2 for s in values_only))
            / (1 - 1 / len(values_only))
        )

    RDI_BANDS = [
        (7.0, 10.0, "#1d6614", "Very High (≥7.0)"),
        (5.5, 7.0, "#6a9f3a", "High (5.5–7.0)"),
        (3.0, 5.5, "#d4b030", "Moderate (3.0–5.5)"),
        (1.0, 3.0, "#e07030", "Low (1.0–3.0)"),
        (0.0, 1.0, "#c03025", "Very Low (<1.0)"),
    ]

    # flake8: noqa: C901
    @staticmethod
    def _get_diversity_label_and_color(diversity):
        for low, high, color, label in RegionColorUtils.RDI_BANDS:
            if low <= diversity <= high:
                return label, color, low, high
        raise ValueError(f"Diversity value {diversity} out of expected range")

    @staticmethod
    def _colors_with_diversity(result_data):

        data_list = result_data["data_list"]
        region_color_map = {}
        value_to_color = {}
        for data in data_list:
            diversity = RegionColorUtils._compute_diversity(
                data["pct_values"]
            )
            label, color, low, high = (
                RegionColorUtils._get_diversity_label_and_color(diversity)
            )
            region_color_map[data["region_id"]] = color

            legend_label = f"{label} ({low:.1f} - {high:.1f})"
            value_to_color[legend_label] = color

        return region_color_map, value_to_color

    @staticmethod
    def get_region_color_map(result_data, how, what):
        data_list = result_data["data_list"]
        if what.get_values(data_list[0]) is None:
            return RegionColorUtils._colors_no_values(result_data)

        if how.params == "Diversity":
            return RegionColorUtils._colors_with_diversity(
                result_data,
            )

        return RegionColorUtils._colors_with_values(result_data, how, what)
