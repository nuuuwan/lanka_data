from lanka_data.api.how.map.ColorUtils import ColorUtils
from lanka_data.api.how.map.OrderColorUtils import OrderColorUtils
from lanka_data.api.what.DiffWhat import DiffWhat
from utils_future import GeoUtils


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
            color = ColorUtils.p_to_color(1 - rank / (n - 1))
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
    def _compute_diversity(pct_values, is_pew=False):
        if is_pew:
            # See
            # https://www.pewresearch.org/religion/2026/02/12/religious-diversity-around-the-world/
            pct_values = {
                "Christians": pct_values['RomanCatholic']
                + pct_values['OtherChristian'],
                'Hindus': pct_values['Hindu'],
                'Muslims': pct_values['Islam'],
                'Buddhists': pct_values['Buddhist'],
                'Jews': 0,
                'ReligiouslyUnaffiliated': pct_values['Unaffiliated'],
                'OtherReligions': pct_values['Other'],
            }

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
    def _colors_with_diversity(result_data, is_pew=False):

        data_list = result_data["data_list"]
        region_color_map = {}
        value_to_color = {}
        for data in data_list:
            diversity = RegionColorUtils._compute_diversity(
                data["pct_values"], is_pew=False
            )
            label, color, low, high = (
                RegionColorUtils._get_diversity_label_and_color(diversity)
            )
            region_color_map[data["region_id"]] = color

            legend_label = f"{label} ({low:.1f} - {high:.1f})"
            value_to_color[legend_label] = color

        return region_color_map, value_to_color

    MAX_NEIGHBOR_DISTANCE_KM = 5

    @staticmethod
    def _colors_with_segregation(result_data):
        region_to_neighbours = {}
        region_idx = {
            result["region_id"]: result for result in result_data["data_list"]
        }
        region_ids = list(region_idx.keys())
        for region_id1 in region_ids:
            lat1, lng1 = (
                region_idx[region_id1]["center_lat"],
                region_idx[region_id1]["center_lng"],
            )
            for region_id2 in region_ids:
                if region_id1 == region_id2:
                    continue
                lat2, lng2 = (
                    region_idx[region_id2]["center_lat"],
                    region_idx[region_id2]["center_lng"],
                )
                distance = GeoUtils.haversine_distance(lat1, lng1, lat2, lng2)
                if distance <= RegionColorUtils.MAX_NEIGHBOR_DISTANCE_KM:
                    region_to_neighbours.setdefault(region_id1, []).append(
                        region_id2
                    )

        region_to_segregation = {}
        for region_id in region_ids:

            neighbors = region_to_neighbours.get(region_id, [])
            if neighbors:
                pct_values = region_idx[region_id]["pct_values"]
                sum_values = {}
                for neighbor_id in neighbors:
                    neighbor_values = region_idx[neighbor_id]["values"]
                    for k, v in neighbor_values.items():
                        sum_values[k] = sum_values.get(k, 0) + v

                total_value = sum(sum_values.values())
                pct_values_for_neighbours = {
                    k: v / total_value for k, v in sum_values.items()
                }

                key_union = set(pct_values.keys()) | set(
                    pct_values_for_neighbours.keys()
                )
                error1_sum = 0
                for k in key_union:
                    pct_value = pct_values.get(k, 0)
                    pct_value_neighbors = pct_values_for_neighbours.get(k, 0)
                    error1 = abs(pct_value - pct_value_neighbors)
                    error1_sum += error1
                segregation = round(error1_sum / len(key_union), 4)
            else:
                segregation = 0

            region_to_segregation[region_id] = segregation

        segregations = list(region_to_segregation.values())
        sorted_segregations = sorted(segregations)

        region_color_map = {}
        value_to_color = {}
        for region_id, segregation in region_to_segregation.items():
            rank_error = sorted_segregations.index(segregation)
            color = ColorUtils.p_to_color(
                (1 - rank_error / (len(sorted_segregations) - 1))
            )
            region_color_map[region_id] = color

            legend_label = f"{segregation:.4f}"
            value_to_color[legend_label] = color

        return region_color_map, value_to_color

    @staticmethod
    def _colors_with_change(result_data):
        data_list = result_data["data_list"]

        changes = [data["change"] for data in data_list]
        sorted_changes = sorted(changes)

        region_color_map = {}
        value_to_color = {}
        for data in data_list:
            change = data["change"]
            rank_error = sorted_changes.index(change)
            color = ColorUtils.p_to_color(
                1 - rank_error / (len(sorted_changes) - 1)
            )
            region_color_map[data["region_id"]] = color

            legend_label = f"{change:.4f}"
            value_to_color[legend_label] = color

        return region_color_map, value_to_color

    @staticmethod
    def _colors_with_flips(result_data):
        data_list = result_data["data_list"]

        region_color_map = {}
        value_to_color = {}
        for data in data_list:
            is_flipped = data["is_flipped"]
            color = ColorUtils.p_to_color(1 if is_flipped else 0)
            region_color_map[data["region_id"]] = color

            legend_label = "Flipped" if is_flipped else "Not Flipped"
            value_to_color[legend_label] = color

        return region_color_map, value_to_color

    @staticmethod
    def get_region_color_map(what, when, where, how):
        result_data = how.get_data(what, when, where)
        data_list = result_data["data_list"]

        if what.get_values(data_list[0]) is None:
            return RegionColorUtils._colors_no_values(result_data)

        if how.params == "Diversity":
            return RegionColorUtils._colors_with_diversity(
                result_data,
                is_pew=False,
            )
        if how.params == "DiversityPew":
            return RegionColorUtils._colors_with_diversity(
                result_data,
                is_pew=True,
            )

        if how.params == "Change":
            if isinstance(what, DiffWhat):
                return RegionColorUtils._colors_with_change(result_data)
            else:
                return RegionColorUtils._colors_with_values(
                    result_data, how.without_params(), what
                )

        if how.params == "Segregation":
            return RegionColorUtils._colors_with_segregation(result_data)

        if how.params == 'Flips':
            return RegionColorUtils._colors_with_flips(result_data)

        return RegionColorUtils._colors_with_values(result_data, how, what)
