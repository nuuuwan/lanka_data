from api.data.Segregation.SegregationComputeMixin import (
    SegregationComputeMixin,
)
from api.utils_future import GeoUtils


class Segregation(SegregationComputeMixin):
    MAX_NEIGHBOR_DISTANCE_KM = 5
    MIN_NEIGHBORS = 5
    SEGREGATION_LIMIT = 0.2

    @staticmethod
    def get_region_to_neighbours(dataset):
        region_to_neighbours = {}
        region_idx = dataset.get_data_idx()
        region_ids = list(region_idx.keys())
        for region_id1 in region_ids:
            lat1 = region_idx[region_id1]["center_lat"]
            lng1 = region_idx[region_id1]["center_lng"]
            for region_id2 in region_ids:
                if region_id1 == region_id2:
                    continue
                lat2 = region_idx[region_id2]["center_lat"]
                lng2 = region_idx[region_id2]["center_lng"]
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
                pct_nbrs = Segregation._compute_neighbour_pct_values(
                    region_idx, neighbors, values_key
                )
                mean_error = Segregation._compute_mean_error(
                    pct_values, pct_nbrs
                )
                if mean_error < Segregation.SEGREGATION_LIMIT:
                    segregation = "(No Segregation)"
                else:
                    segregation = "Segregated"
            else:
                segregation = "(Insufficient Data)"
            region_to_segregation[region_id] = segregation
        return region_to_segregation

    @staticmethod
    def get_segregation_change(result_data):
        region1_to_seg = Segregation.get_region_to_segregation(
            result_data, "pct_values1", "values1"
        )
        region2_to_seg = Segregation.get_region_to_segregation(
            result_data, "pct_values2", "values2"
        )
        common = set(region1_to_seg.keys()) & set(region2_to_seg.keys())
        uncommon = set(region1_to_seg.keys()) ^ set(region2_to_seg.keys())
        result = {}
        for region_id in common:
            seg1 = region1_to_seg[region_id]
            seg2 = region2_to_seg[region_id]
            if seg1 == "(Insufficient Data)" or seg2 == "(Insufficient Data)":
                result[region_id] = "(Insufficient Data)"
            else:
                result[region_id] = (
                    "Change" if seg2 != seg1 else "(No Change)"
                )
        for region_id in uncommon:
            result[region_id] = None
        return result
