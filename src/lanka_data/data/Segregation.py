from utils_future import GeoUtils


class Segregation:
    MAX_NEIGHBOR_DISTANCE_KM = 5
    MIN_NEIGHBORS = 5
    SEGREGATION_LIMIT = 0.2

    @staticmethod
    def get_region_to_neighbours(dataset):
        region_to_neighbours = {}
        region_idx = dataset.get_data_idx()
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
                if distance <= Segregation.MAX_NEIGHBOR_DISTANCE_KM:
                    region_to_neighbours.setdefault(region_id1, []).append(
                        region_id2
                    )
        return region_to_neighbours, region_idx

    @staticmethod
    def get_region_to_segregation(
        result_data, pct_values_key="pct_values", values_key="values"
    ):

        region_to_neighbours, region_idx = (
            Segregation.get_region_to_neighbours(result_data)
        )
        region_to_segregation = {}
        for region_id in region_idx.keys():
            neighbors = region_to_neighbours.get(region_id, [])
            if neighbors and len(neighbors) >= Segregation.MIN_NEIGHBORS:
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
                mean_error1 = error1_sum / len(key_union) if key_union else 0
                if mean_error1 < Segregation.SEGREGATION_LIMIT:
                    segregation = "(No Segregation)"
                else:
                    segregation = "Segregated"
            else:
                segregation = "(Insufficient Data)"

            region_to_segregation[region_id] = segregation

        return region_to_segregation

    @staticmethod
    def get_segregation_change(result_data):
        region1_to_segregation = Segregation.get_region_to_segregation(
            result_data, "pct_values1", "values1"
        )
        region2_to_segregation = Segregation.get_region_to_segregation(
            result_data, "pct_values2", "values2"
        )
        common_regions = set(region1_to_segregation.keys()) & set(
            region2_to_segregation.keys()
        )
        uncommon_regions = set(region1_to_segregation.keys()) ^ set(
            region2_to_segregation.keys()
        )
        region_to_segregation_change = {}
        for region_id in common_regions:
            region1_segregation = region1_to_segregation[region_id]
            region2_segregation = region2_to_segregation[region_id]
            if (
                region1_segregation == "(Insufficient Data)"
                or region2_segregation == ("(Insufficient Data)")
            ):
                segregation_change = "(Insufficient Data)"
            else:
                segregation_change = (
                    "Change"
                    if region2_segregation != region1_segregation
                    else "(No Change)"
                )
            region_to_segregation_change[region_id] = segregation_change
        for region_id in uncommon_regions:
            region_to_segregation_change[region_id] = None

        return region_to_segregation_change
