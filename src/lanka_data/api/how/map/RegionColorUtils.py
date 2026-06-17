from lanka_data.api.how.map import ColorSpec
from lanka_data.api.how.map.OrderColorUtils import OrderColorUtils
from lanka_data.api.what.DiffWhat import DiffWhat
from utils_future import GeoUtils


class RegionColorUtils:

    @staticmethod
    def get_color_spec_generic(result_data, how, what) -> ColorSpec:
        func_key_getter = OrderColorUtils._func_key_getter(how, what)
        if func_key_getter:
            return ColorSpec.by_custom_key(
                result_data, func_key_getter, False
            )
        return ColorSpec.by_single_pct_value(result_data, how)

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
    def get_colors_from_diversity(
        result_data, is_pew=False, pct_values_key='pct_values'
    ):

        return ColorSpec.by_region_to_custom_value(
            RegionColorUtils.get_region_to_diversity(
                result_data, is_pew, pct_values_key
            ),
            False,
        )

    @staticmethod
    def get_region_to_diversity_change(result_data, is_pew=False):

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

        return region_to_diversity_change

    def get_color_spec_for_diversity_change(result_data, is_pew=False):
        region_to_diversity_change = (
            RegionColorUtils.get_region_to_diversity_change(
                result_data, is_pew
            )
        )
        return ColorSpec.by_region_to_custom_value(
            region_to_diversity_change, True
        )

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
    def get_color_spec_for_segregation(result_data):
        return ColorSpec.by_region_to_custom_value(
            RegionColorUtils.get_region_to_segregation(result_data), False
        )

    @staticmethod
    def get_segregation_change(result_data):
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

        return region_to_segregation_change

    @staticmethod
    def get_color_spec_for_segregation_change(result_data):

        return ColorSpec.by_region_to_custom_value(
            RegionColorUtils.get_segregation_change(result_data), True
        )

    @staticmethod
    def get_region_to_change(result_data):
        region_to_change = {}
        for data in result_data["data_list"]:
            region_to_change[data["region_id"]] = data["change"]
        return region_to_change

    @staticmethod
    def get_color_spec_for_change(result_data):
        return ColorSpec.by_region_to_custom_value(
            RegionColorUtils.get_region_to_change(result_data), True
        )

    @staticmethod
    def get_colors_from_flips(result_data):
        region_to_flip = {
            data["region_id"]: data["flips"]
            for data in result_data["data_list"]
        }
        return ColorSpec.by_region_to_custom_value(region_to_flip, True)

    @staticmethod
    def get_color_spec(what, when, where, how) -> ColorSpec:
        result_data = how.get_data(what, when, where)
        data_list = result_data["data_list"]
        is_diff = isinstance(what, DiffWhat)

        if what.get_values(data_list[0]) is None:
            return ColorSpec.by_custom_key(
                result_data,
                lambda data: data["region_id"],
                True,
            )

        if how.params == "Diversity":
            if is_diff:
                return RegionColorUtils.get_color_spec_for_diversity_change(
                    result_data,
                    is_pew=False,
                )

            return RegionColorUtils.get_colors_from_diversity(
                result_data,
                is_pew=False,
            )

        if how.params == "DiversityPew":
            if is_diff:
                return RegionColorUtils.get_color_spec_for_diversity_change(
                    result_data,
                    is_pew=True,
                )
            return RegionColorUtils.get_colors_from_diversity(
                result_data,
                is_pew=True,
            )

        if how.params == "Change":
            if is_diff:
                return RegionColorUtils._colors_with_change(result_data)
            return RegionColorUtils.get_color(
                result_data, how.without_params(), what
            )

        if how.params == "Segregation":
            if is_diff:
                return RegionColorUtils._colors_with_segregation_change(
                    result_data
                )
            return RegionColorUtils.get_color_spec_for_segregation(
                result_data
            )

        if how.params == 'Flips':
            if is_diff:
                return RegionColorUtils.get_colors_from_flips(result_data)
            return RegionColorUtils.get_color(
                result_data, how.without_params(), what
            )

        return RegionColorUtils.get_color_spec_generic(result_data, how, what)
