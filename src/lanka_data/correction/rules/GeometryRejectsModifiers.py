from lanka_data.correction.Correction import Correction
from lanka_data.correction.CorrectionRule import CorrectionRule
from lanka_data.metadata.Measurements import Measurements


class GeometryRejectsModifiers(CorrectionRule):
    name = "geometry_measurement_rejects_modifiers"
    field = "How"
    severity = "lossless"

    def applies(self, wc):
        if wc.how == "":
            return False
        if Measurements.kind(wc.what) != "geometry":
            return False
        return wc.how_field.modifier is not None

    def apply(self, wc):
        base = wc.how_field.base
        reason = (
            f"{wc.what} binds no data;"
            " ranking modifiers require a categorical measurement."
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
