class GeoUtils:
    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        from math import atan2, cos, radians, sin, sqrt

        R = 6371.0  # Earth radius in kilometers

        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = (
            sin(dlat / 2) ** 2
            + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        )
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance_km = R * c
        return distance_km
