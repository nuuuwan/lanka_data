class BumpChartDataMixin:
    @staticmethod
    def _has_diff_values(subregions):
        return any(
            s.get("values1") is not None and s.get("values2") is not None
            for s in subregions
        )

    @staticmethod
    def _get_region_totals(subregions, value_key):
        return {
            s["region_id"]: sum((s.get(value_key) or {}).values())
            for s in subregions
            if s.get("region_id") is not None
        }

    @staticmethod
    def _get_rank_map(region_to_total):
        ranked = sorted(region_to_total.items(), key=lambda x: (-x[1], x[0]))
        return {rid: rank for rank, (rid, _) in enumerate(ranked, start=1)}

    @classmethod
    def _get_selected_region_ids(cls, rank_map1, rank_map2):
        return sorted(
            set(rank_map1) | set(rank_map2),
            key=lambda rid: (
                min(rank_map1.get(rid, 10**6), rank_map2.get(rid, 10**6)),
                rid,
            ),
        )

    def _get_rank_maps(self, subregions):
        rank_map1 = self._get_rank_map(
            self._get_region_totals(subregions, "values1")
        )
        rank_map2 = self._get_rank_map(
            self._get_region_totals(subregions, "values2")
        )
        return rank_map1, rank_map2

    @staticmethod
    def _build_id_to_name(subregions):
        return {
            s["region_id"]: s.get("region_name") or str(s["region_id"])
            for s in subregions
            if s.get("region_id") is not None
        }
