from lanka_data.correction.Correction import Correction
from lanka_data.correction.CorrectionRule import CorrectionRule
from lanka_data.metadata.Measurements import Measurements


class SnapObservationYear(CorrectionRule):
    name = "snap_observation_year"
    field = "When"
    severity = "destructive"

    def _years(self, wc):
        return Measurements.observation_years(wc.what)

    def applies(self, wc):
        if wc.when == "" or wc.when_field.is_interval:
            return False
        years = self._years(wc)
        if not years:
            return False
        return int(wc.when) not in years

    @staticmethod
    def _nearest(value, years):
        return min(years, key=lambda y: (abs(y - value), y))

    def apply(self, wc):
        value = int(wc.when)
        years = self._years(wc)
        target = self._nearest(value, years)
        reason = (
            f"{wc.when} is not an observation year for {wc.what};"
            f" snapped to nearest ({target})."
        )
        if self._is_tie(value, target, years):
            reason += " (tie resolved to earlier year)"
        correction = Correction(
            field=self.field,
            rule=self.name,
            from_value=wc.when,
            to_value=str(target),
            severity=self.severity,
            reason=reason,
        )
        return wc.with_when(str(target)), correction

    @staticmethod
    def _is_tie(value, target, years):
        distance = abs(target - value)
        return any(y != target and abs(y - value) == distance for y in years)
