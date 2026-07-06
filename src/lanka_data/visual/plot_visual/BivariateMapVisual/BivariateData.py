class BivariateData:
    @staticmethod
    def _dominant(pct_values):
        if not pct_values:
            return None, 0.0
        label = max(pct_values, key=pct_values.get)
        return label, pct_values[label]

    @classmethod
    def _point(cls, row):
        pct1 = row.get("pct_values1")
        pct2 = row.get("pct_values2")
        if not pct1 or not pct2:
            return None
        x_label, x = cls._dominant(pct1)
        y_label, y = cls._dominant(pct2)
        return {
            "region_id": row.get("region_id"),
            "region_name": row.get("region_name")
            or str(row.get("region_id")),
            "x": x,
            "y": y,
            "x_label": x_label,
            "y_label": y_label,
        }

    @classmethod
    def points(cls, data_table):
        result = []
        for row in data_table:
            point = cls._point(row)
            if point is not None:
                result.append(point)
        return result

    @staticmethod
    def thresholds(values, n_bins):
        if not values or n_bins < 2:
            return []
        ordered = sorted(values)
        count = len(ordered)
        return [ordered[int(k * count / n_bins)] for k in range(1, n_bins)]

    @staticmethod
    def bin_index(value, thresholds):
        index = 0
        for threshold in thresholds:
            if value >= threshold:
                index += 1
        return index

    @classmethod
    def classify(cls, points, n_bins):
        x_thresholds = cls.thresholds([p["x"] for p in points], n_bins)
        y_thresholds = cls.thresholds([p["y"] for p in points], n_bins)
        for point in points:
            point["x_bin"] = cls.bin_index(point["x"], x_thresholds)
            point["y_bin"] = cls.bin_index(point["y"], y_thresholds)
        return points
