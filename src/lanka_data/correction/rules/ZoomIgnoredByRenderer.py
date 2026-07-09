import re

from lanka_data.correction.Correction import Correction
from lanka_data.correction.CorrectionRule import CorrectionRule

ZOOM_RE = re.compile(r"@\d+(?:\.\d+)?")


class ZoomIgnoredByRenderer(CorrectionRule):
    name = "zoom_ignored_by_renderer"
    field = "Where"
    severity = "lossless"

    def applies(self, wc):
        if wc.where_field.zoom is None:
            return False
        return wc.how_field.base != "Map"

    def apply(self, wc):
        new_where = ZOOM_RE.sub("", wc.where)
        reason = (
            f"@{wc.where_field.zoom:g} ignored;"
            f" {wc.how_field.base} is not a map renderer."
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
