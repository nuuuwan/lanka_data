class AnnotationsStatMixin:
    OUTLIER_Z = 2.0
    MIN_OUTLIER_ROWS = 3

    def _rows(self):
        rows = []
        for row in self.data_table:
            values = row.get("values") or {}
            total = row.get("total_value")
            if total is None:
                total = sum(values.values())
            name = row.get("region_name") or str(row.get("region_id"))
            rows.append({"name": name, "total": total, "row": row})
        return rows

    @staticmethod
    def _mean(totals):
        return sum(totals) / len(totals)

    @classmethod
    def _std(cls, totals):
        mean = cls._mean(totals)
        variance = sum((t - mean) ** 2 for t in totals) / len(totals)
        return variance**0.5

    def _outliers(self, rows):
        totals = [r["total"] for r in rows]
        if len(totals) < self.MIN_OUTLIER_ROWS:
            return []
        std = self._std(totals)
        if std == 0:
            return []
        mean = self._mean(totals)
        limit = self.OUTLIER_Z * std
        return [r for r in rows if abs(r["total"] - mean) > limit]
