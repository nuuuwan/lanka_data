class ScatterPlotData:
    @staticmethod
    def _top_category(values):
        positive = {k: v for k, v in values.items() if v > 0}
        if not positive:
            return None, None
        total = sum(positive.values())
        label = max(positive, key=positive.get)
        return label, positive[label] / total

    @classmethod
    def points(cls, subregions):
        result = []
        for subregion in subregions:
            values = subregion["values"]
            total = subregion.get("total_value")
            if total is None:
                total = sum(values.values())
            label, share = cls._top_category(values)
            if label is None or total <= 0:
                continue
            result.append((total, share, label, subregion["region_name"]))
        return result
