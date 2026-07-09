from lanka_data.correction.Correction import Correction
from lanka_data.correction.CorrectionRule import CorrectionRule
from lanka_data.metadata.Measurements import Measurements


class ChangeRequiresTwoObservations(CorrectionRule):
    name = "change_requires_two_observations"
    field = "How"
    severity = "lossless"

    def _count(self, wc):
        if wc.when == "":
            return None
        return Measurements.count_in_when(wc.what, wc.when_field)

    def applies(self, wc):
        if wc.how == "" or wc.how_field.modifier != "Change":
            return False
        count = self._count(wc)
        return count is not None and count != 2

    def apply(self, wc):
        base = wc.how_field.base
        count = self._count(wc)
        reason = (
            "Change needs exactly two observations;"
            f" {wc.when} resolves to {count}."
        )
        correction = Correction(
            field=self.field,
            rule=self.name,
            from_value=wc.how,
            to_value=base,
            severity=self.severity,
            reason=reason,
        )
        return wc.with_how(base), correction
