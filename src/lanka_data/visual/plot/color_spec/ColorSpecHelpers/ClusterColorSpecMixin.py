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
    def _format_label(sorted_pct):
        top = sorted_pct[:2]
        parts = [f"{field} ({share * 100:.0f}%)" for field, share in top]
        other = max(0.0, 1.0 - sum(share for _, share in top))
        parts.append(f"Other ({other * 100:.0f}%)")
        return ", ".join(parts)

    @classmethod
    def _cluster_label_and_color(cls, data_list):
        sorted_pct = cls._mean_pct(data_list)
        if not sorted_pct:
            return "(No Data)", ColorSpec.LABEL_TO_COLOR["(No Data)"]
        top_field, top_share = sorted_pct[0]
        return (
            cls._format_label(sorted_pct),
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
        labels, _ = ClusterData.cluster(vectors, n_clusters)
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
