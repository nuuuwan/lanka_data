from lanka_data.correction.Correction import Correction
from lanka_data.correction.CorrectionRule import CorrectionRule
from lanka_data.metadata.Measurements import Measurements


class SnapIntervalEndpoints(CorrectionRule):
    name = "snap_interval_endpoints"
    field = "When"
    severity = "destructive"

    def _years(self, wc):
        return Measurements.observation_years(wc.what)

    def applies(self, wc):
        if not wc.when_field.is_interval:
            return False
        years = self._years(wc)
        if not years:
            return False
        return self._snap(wc) != wc.when

    def _snap(self, wc):
        years = self._years(wc)
        start = int(wc.when_field.start)
        end = int(wc.when_field.end)
        new_start = self._round_down(start, years)
        new_end = self._round_up(end, years)
        if new_start == new_end:
            return str(new_start)
        return f"{new_start}-{new_end}"

    @staticmethod
    def _round_down(value, years):
        below = [y for y in years if y <= value]
        return max(below) if below else min(years)

    @staticmethod
    def _round_up(value, years):
        above = [y for y in years if y >= value]
        return min(above) if above else max(years)

    def apply(self, wc):
        target = self._snap(wc)
        reason = (
            f"interval {wc.when} snapped to {target}"
            f" ({self._direction(wc.when, target)})."
        )
        correction = Correction(
            field=self.field,
            rule=self.name,
            from_value=wc.when,
            to_value=target,
            severity=self.severity,
            reason=reason,
        )
        return wc.with_when(target), correction

    @staticmethod
    def _bounds(value):
        parts = [int(p) for p in value.split("-")]
        return parts[0], parts[-1]

    def _direction(self, old, new):
        old_lo, old_hi = self._bounds(old)
        new_lo, new_hi = self._bounds(new)
        if new_lo <= old_lo and new_hi >= old_hi:
            return "widened"
        if new_lo >= old_lo and new_hi <= old_hi:
            return "narrowed"
        return "snapped"
