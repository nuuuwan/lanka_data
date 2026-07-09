from lanka_data.correction.Correction import Correction
from lanka_data.correction.CorrectionRule import CorrectionRule
from lanka_data.datasets.region.RegionTypeUtils import RegionTypeUtils


class ResolveSelfType(CorrectionRule):
    name = "resolve_self_type"
    field = "Where"
    severity = "lossless"

    def _parent_id(self, wc):
        parent = wc.where_field.parent_part
        return parent.split("@", 1)[0]

    def applies(self, wc):
        child = wc.where_field.child_region_type
        if not child:
            return False
        parent_id = self._parent_id(wc)
        if any(sep in parent_id for sep in (",", "...")):
            return False
        return self._self_typed(parent_id, child)

    @staticmethod
    def _self_typed(parent_id, child):
        try:
            return RegionTypeUtils.get_region_type(parent_id) == child
        except ValueError:
            return False

    def apply(self, wc):
        child = wc.where_field.child_region_type
        new_where = wc.where.replace(f":{child}", "", 1)
        reason = (
            f"{self._parent_id(wc)} is already a {child};"
            f" dropped redundant :{child}."
        )
        correction = Correction(
            field=self.field,
            rule=self.name,
            from_value=wc.where,
            to_value=new_where,
            severity=self.severity,
            reason=reason,
        )
        return wc.with_where(new_where), correction
