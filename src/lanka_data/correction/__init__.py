# lanka_data.correction
#
# Boundary epoch is deliberately NOT reconciled here.
#
# TODO(boundary-epoch): Do not add a rule that reconciles ``When`` against a
# ``pre<year>`` boundary variant, and do not silently reproject an interval
# that straddles a boundary redesign. Reprojection is a spatial operation with
# real error, not a syntax correction; per the design doc (section 2.4),
# collapsing observation time and boundary epoch "would misattribute counts to
# the wrong geometry". Such conflicts must keep raising, never be corrected.

from lanka_data.correction.Correction import Correction
from lanka_data.correction.CorrectionErrors import (
    CorrectionLoopError,
    DestructiveCorrectionError,
    UnknownMeasurementError,
)
from lanka_data.correction.CorrectionPipeline import correct
from lanka_data.correction.CorrectionPolicy import CorrectionPolicy

__all__ = [
    "Correction",
    "CorrectionPolicy",
    "CorrectionLoopError",
    "DestructiveCorrectionError",
    "UnknownMeasurementError",
    "correct",
]
