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
            color = ColorUtils.p_to_color_for_abs(1 - rank / (n - 1))
            value_to_color[value] = color
            region_color_map[data["region_id"]] = color
        return region_color_map, value_to_color

    @staticmethod
    def _colors_with_values(result_data, how, what):
        func_key_getter = OrderColorUtils._func_key_getter(how, what)
        if func_key_getter:
            return OrderColorUtils.get_region_colors_by_key(
                result_data, func_key_getter
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
                'ReligiouslyUnaffiliated': 0,
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
        (7.0, 10.0, "#1d6614", "1-Very High (≥7.0)"),
        (5.5, 7.0, "#6a9f3a", "2-High (5.5–7.0)"),
        (3.0, 5.5, "#d4b030", "3-Moderate (3.0–5.5)"),
        (1.0, 3.0, "#e07030", "4-Low (1.0–3.0)"),
        (0.0, 1.0, "#c03025", "5-Very Low (<1.0)"),
    ]

    # flake8: noqa: C901
    @staticmethod
    def _get_diversity_label_and_color(diversity):
        for low, high, color, label in RegionColorUtils.RDI_BANDS:
            if low <= diversity <= high:
                return label, color, low, high
        raise ValueError(f"Diversity value {diversity} out of expected range")

    @staticmethod
    def get_region_to_diversity(
        result_data, is_pew=False, pct_values_key='pct_values'
    ):
        data_list = result_data["data_list"]
        region_to_diversity = {}
        for data in data_list:
            diversity = RegionColorUtils._compute_diversity(
                data[pct_values_key], is_pew
            )
            region_to_diversity[data["region_id"]] = diversity
        return region_to_diversity

    @staticmethod
    def _colors_with_diversity(
        result_data, is_pew=False, pct_values_key='pct_values'
    ):

        region_to_diversity = RegionColorUtils.get_region_to_diversity(
            result_data, is_pew, pct_values_key
        )

        region_color_map = {}
        value_to_color = {}
        for region_id, diversity in region_to_diversity.items():
            label, color, low, high = (
                RegionColorUtils._get_diversity_label_and_color(diversity)
            )
            region_color_map[region_id] = color

            legend_label = f"{label} ({low:.1f} - {high:.1f})"
            value_to_color[legend_label] = color

        return region_color_map, value_to_color

    @staticmethod
    def _colors_with_diversity_change(result_data, is_pew=False):

        region_to_diversity1 = RegionColorUtils.get_region_to_diversity(
            result_data, is_pew, 'pct_values1'
        )
        region_to_diversity2 = RegionColorUtils.get_region_to_diversity(
            result_data, is_pew, 'pct_values2'
        )

        common_regions = set(region_to_diversity1.keys()) & set(
            region_to_diversity2.keys()
        )
        uncommon_regions = set(region_to_diversity1.keys()) ^ set(
            region_to_diversity2.keys()
        )

        region_to_diversity_change = {}
        for region_id in common_regions:
            diversity_change = (
                region_to_diversity2[region_id]
                - region_to_diversity1[region_id]
            )
            region_to_diversity_change[region_id] = diversity_change

        for region_id in uncommon_regions:
            region_to_diversity_change[region_id] = None

        region_to_color = {}
        value_to_color = {}
        max_abs_change = max(
            abs(change)
            for change in region_to_diversity_change.values()
            if change is not None
        )
        for region_id, diversity_change in region_to_diversity_change.items():
            if diversity_change is not None:
                p = (
                    max(
                        -max_abs_change, min(max_abs_change, diversity_change)
                    )
                    / (2 * max_abs_change)
                    + 0.5
                )
                value = f"{diversity_change:+.4f}"
                color = ColorUtils.p_to_color_for_diff(p)

                region_to_color[region_id] = color
                value_to_color[value] = color

        value_to_color = dict(
            sorted(value_to_color.items(), key=lambda x: float(x[0]))
        )
        return region_to_color, value_to_color

    MAX_NEIGHBOR_DISTANCE_KM = 5

    @staticmethod
    def get_region_to_neighbours(result_data):
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
        return region_to_neighbours, region_idx

    @staticmethod
    def get_region_to_segregation(
        result_data, pct_values_key='pct_values', values_key='values'
    ):

        region_to_neighbours, region_idx = (
            RegionColorUtils.get_region_to_neighbours(result_data)
        )
        region_to_segregation = {}
        for region_id in region_idx.keys():
            neighbors = region_to_neighbours.get(region_id, [])
            if neighbors:
                pct_values = region_idx[region_id][pct_values_key]
                sum_values = {}
                for neighbor_id in neighbors:
                    neighbor_values = region_idx[neighbor_id][values_key]
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

        return region_to_segregation

    @staticmethod
    def _colors_with_segregation(result_data):

        region_to_segregation = RegionColorUtils.get_region_to_segregation(
            result_data
        )

        segregations = list(region_to_segregation.values())
        sorted_segregations = sorted(segregations)

        region_color_map = {}
        value_to_color = {}
        for region_id, segregation in region_to_segregation.items():
            rank_error = sorted_segregations.index(segregation)
            color = ColorUtils.p_to_color_for_abs(
                (1 - rank_error / (len(sorted_segregations) - 1))
            )
            region_color_map[region_id] = color

            legend_label = f"{segregation:.4f}"
            value_to_color[legend_label] = color

        return region_color_map, value_to_color

    @staticmethod
    def _colors_with_segregation_change(result_data):
        region1_to_segregation = RegionColorUtils.get_region_to_segregation(
            result_data, 'pct_values1', 'values1'
        )
        region2_to_segregation = RegionColorUtils.get_region_to_segregation(
            result_data, 'pct_values2', 'values2'
        )
        common_regions = set(region1_to_segregation.keys()) & set(
            region2_to_segregation.keys()
        )
        uncommon_regions = set(region1_to_segregation.keys()) ^ set(
            region2_to_segregation.keys()
        )
        region_to_segregation_change = {}
        for region_id in common_regions:
            segregation_change = (
                region2_to_segregation[region_id]
                - region1_to_segregation[region_id]
            )
            region_to_segregation_change[region_id] = segregation_change
        for region_id in uncommon_regions:
            region_to_segregation_change[region_id] = None
        region_to_color = {}
        value_to_color = {}
        max_abs_change = max(
            abs(change)
            for change in region_to_segregation_change.values()
            if change is not None
        )
        for (
            region_id,
            segregation_change,
        ) in region_to_segregation_change.items():
            if segregation_change is not None:
                p = (
                    (
                        max(
                            -max_abs_change,
                            min(max_abs_change, segregation_change),
                        )
                        / (2 * max_abs_change)
                        + 0.5
                    )
                    if max_abs_change > 0
                    else 0.5
                )
                value = f"{segregation_change:+.4f}"
                color = ColorUtils.p_to_color_for_diff(p)
                region_to_color[region_id] = color
                value_to_color[value] = color
        value_to_color = dict(
            sorted(value_to_color.items(), key=lambda x: float(x[0]))
        )

        return region_to_color, value_to_color

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
            color = ColorUtils.p_to_color_for_diff(
                1 - rank_error / (len(sorted_changes) - 1)
            )
            region_color_map[data["region_id"]] = color

            legend_label = f"{change:+.4f}"
            value_to_color[legend_label] = color

        return region_color_map, value_to_color

    @staticmethod
    def _colors_with_flips(result_data):
        data_list = result_data["data_list"]

        region_color_map = {}
        value_to_color = {}
        sorted_flips = sorted(list(set([data["flip"] for data in data_list])))

        n_flips = len(sorted_flips)
        for data in data_list:
            flip = data["flip"]
            i_flip = sorted_flips.index(flip)
            color = ColorUtils.p_to_color_for_abs(
                i_flip / (n_flips - 1) if n_flips > 1 else 0.5
            )
            region_color_map[data["region_id"]] = color

            value_to_color[flip] = color

        return region_color_map, value_to_color

    @staticmethod
    def get_region_color_map(what, when, where, how):
        result_data = how.get_data(what, when, where)
        data_list = result_data["data_list"]
        is_diff = isinstance(what, DiffWhat)

        if what.get_values(data_list[0]) is None:
            return RegionColorUtils._colors_no_values(result_data)

        if how.params == "Diversity":
            if is_diff:
                return RegionColorUtils._colors_with_diversity_change(
                    result_data,
                    is_pew=False,
                )

            return RegionColorUtils._colors_with_diversity(
                result_data,
                is_pew=False,
            )

        if how.params == "DiversityPew":
            if is_diff:
                return RegionColorUtils._colors_with_diversity_change(
                    result_data,
                    is_pew=True,
                )
            return RegionColorUtils._colors_with_diversity(
                result_data,
                is_pew=True,
            )

        if how.params == "Change":
            if is_diff:
                return RegionColorUtils._colors_with_change(result_data)
            return RegionColorUtils._colors_with_values(
                result_data, how.without_params(), what
            )

        if how.params == "Segregation":
            if is_diff:
                return RegionColorUtils._colors_with_segregation_change(
                    result_data
                )
            return RegionColorUtils._colors_with_segregation(result_data)

        if how.params == 'Flips':
            if is_diff:
                return RegionColorUtils._colors_with_flips(result_data)
            return RegionColorUtils._colors_with_values(
                result_data, how.without_params(), what
            )

        return RegionColorUtils._colors_with_values(result_data, how, what)
