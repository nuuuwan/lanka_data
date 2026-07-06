class RegionPopulationFilter:
    MIN_RATIO = 0.01

    @staticmethod
    def _population(d):
        return abs(d.get("total_value", 0) or 0)

    @classmethod
    def _remove_tiny(cls, sorted_data):
        while len(sorted_data) >= 2:
            smallest = cls._population(sorted_data[0])
            next_smallest = cls._population(sorted_data[1])
            if smallest < cls.MIN_RATIO * next_smallest:
                sorted_data.pop(0)
            else:
                break
        return sorted_data

    @classmethod
    def filter(cls, data_list):
        nonzero = [d for d in data_list if cls._population(d) > 0]
        sorted_data = sorted(nonzero, key=cls._population)
        kept = cls._remove_tiny(sorted_data)
        kept_ids = {d["region_id"] for d in kept}
        return [d for d in data_list if d["region_id"] in kept_ids]
