from lanka_data.visual.plot.color_spec.ClusterData import ClusterData
from lanka_data.visual.plot.color_spec.ColorSpec.ColorSpec import ColorSpec


class ClusterColorSpecMixin:
    @staticmethod
    def _region_pct(data):
        pct = data.get("pct_values")
        if pct:
            return pct
        values = data.get("values", {})
        total = sum(values.values())
        if total <= 0:
            return {}
        return {k: v / total for k, v in values.items()}

    @classmethod
    def _mean_pct(cls, data_list):
        totals = {}
        weight_total = 0
        for data in data_list:
            weight = abs(data.get("total_value", 0) or 0)
            weight_total += weight
            for field, share in cls._region_pct(data).items():
                totals[field] = totals.get(field, 0) + weight * share
        if weight_total <= 0:
            return cls._unweighted_mean_pct(data_list)
        mean = {
            field: total / weight_total for field, total in totals.items()
        }
        return sorted(mean.items(), key=lambda item: -item[1])

    @classmethod
    def _unweighted_mean_pct(cls, data_list):
        totals = {}
        for data in data_list:
            for field, share in cls._region_pct(data).items():
                totals[field] = totals.get(field, 0) + share
        n = len(data_list)
        if n == 0:
            return []
        mean = {field: share / n for field, share in totals.items()}
        return sorted(mean.items(), key=lambda item: -item[1])

    @staticmethod
    def _color_for_field(field, share):
        base = ColorSpec.cmap_for_label(field)(1.0)
        return (base[0], base[1], base[2], round(share, 4))

    @staticmethod
    def _format_range(field, shares):
        low = round(min(shares) * 100)
        high = round(max(shares) * 100)
        if low == high:
            return f"{field} ({low}%)"
        return f"{field} ({low}-{high}%)"

    @classmethod
    def _format_label(cls, sorted_pct, data_list):
        top = [field for field, _ in sorted_pct[:2]]
        rows = [
            [cls._region_pct(data).get(field, 0.0) for field in top]
            for data in data_list
        ]
        parts = [
            cls._format_range(field, [row[i] for row in rows])
            for i, field in enumerate(top)
        ]
        others = [max(0.0, 1.0 - sum(row)) for row in rows]
        parts.append(cls._format_range("Other", others))
        return ", ".join(parts)

    @classmethod
    def _cluster_label_and_color(cls, data_list):
        sorted_pct = cls._mean_pct(data_list)
        if not sorted_pct:
            return "(No Data)", ColorSpec.LABEL_TO_COLOR["(No Data)"]
        top_field, top_share = sorted_pct[0]
        return (
            cls._format_label(sorted_pct, data_list),
            cls._color_for_field(top_field, top_share),
        )

    @classmethod
    def _pct_vectors(cls, data_list):
        fields = []
        for data in data_list:
            for field in cls._region_pct(data):
                if field not in fields:
                    fields.append(field)
        return [
            [cls._region_pct(data).get(field, 0.0) for field in fields]
            for data in data_list
        ]

    @classmethod
    def get_color_spec_for_cluster(cls, dataset, n_clusters) -> ColorSpec:
        data_list = dataset.get_data_table()
        vectors = cls._pct_vectors(data_list)
        weights = [abs(d.get("total_value", 0) or 0) for d in data_list]
        labels, _ = ClusterData.cluster(vectors, n_clusters, weights)
        cluster_to_data = {}
        for data, label in zip(data_list, labels):
            cluster_to_data.setdefault(label, []).append(data)
        cluster_to_spec = {
            label: cls._cluster_label_and_color(members)
            for label, members in cluster_to_data.items()
        }
        region_to_color, value_to_color = {}, {}
        for data, label in zip(data_list, labels):
            legend_label, color = cluster_to_spec[label]
            region_to_color[data["region_id"]] = color
            value_to_color[legend_label] = color
        return ColorSpec(region_to_color, value_to_color)
