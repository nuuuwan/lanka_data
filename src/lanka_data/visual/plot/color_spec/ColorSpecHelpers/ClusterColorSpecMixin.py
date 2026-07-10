from lanka_data.visual.plot.color_spec.ClusterData import ClusterData
from lanka_data.visual.plot.color_spec.ColorSpec.ColorSpec import ColorSpec


class ClusterColorSpecMixin:
    @staticmethod
    def _region_total(data):
        total = data.get("total_value")
        if total is None:
            total = sum(data.get("values", {}).values())
        return total

    @staticmethod
    def _format_cluster_center(value):
        av = abs(value)
        if av >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"
        if av >= 1_000:
            return f"{value / 1_000:.0f}K"
        return f"{value:.0f}"

    @classmethod
    def get_color_spec_for_cluster(cls, dataset, n_clusters) -> ColorSpec:
        data_list = dataset.get_data_table()
        totals = [cls._region_total(d) for d in data_list]
        labels, centers = ClusterData.cluster(totals, n_clusters)
        region_to_center = {
            data["region_id"]: centers[label]
            for data, label in zip(data_list, labels)
        }
        return ColorSpec.by_region_to_custom_value(
            region_to_center,
            False,
            value_formatter=cls._format_cluster_center,
        )
