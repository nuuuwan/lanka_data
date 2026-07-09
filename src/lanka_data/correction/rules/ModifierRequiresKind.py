from lanka_data.correction.Correction import Correction
from lanka_data.correction.CorrectionRule import CorrectionRule
from lanka_data.metadata.Measurements import Measurements
from lanka_data.metadata.Modifiers import Modifiers


class ModifierRequiresKind(CorrectionRule):
    name = "modifier_requires_kind"
    field = "How"
    severity = "lossless"

    def applies(self, wc):
        if wc.how == "":
            return False
        modifier = wc.how_field.modifier
        if not Modifiers.is_categorical_modifier(modifier):
            return False
        return Measurements.kind(wc.what) != "categorical"

    def apply(self, wc):
        modifier = wc.how_field.modifier
        base = wc.how_field.base
        kind = Measurements.kind(wc.what)
        reason = (
            f"{modifier} requires a categorical measurement;"
            f" {wc.what} is {kind}."
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
