class SegregationComputeMixin:
    @staticmethod
    def _compute_neighbour_pct_values(region_idx, neighbors, values_key):
        sum_values = {}
        for neighbor_id in neighbors:
            for k, v in region_idx[neighbor_id][values_key].items():
                sum_values[k] = sum_values.get(k, 0) + v
        total_value = sum(sum_values.values())
        return {k: v / total_value for k, v in sum_values.items()}

    @staticmethod
    def _compute_mean_error(pct_values, pct_values_for_neighbours):
        key_union = set(pct_values.keys()) | set(
            pct_values_for_neighbours.keys()
        )
        error1_sum = sum(
            abs(pct_values.get(k, 0) - pct_values_for_neighbours.get(k, 0))
            for k in key_union
        )
        return error1_sum / len(key_union) if key_union else 0
