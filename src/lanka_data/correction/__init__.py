# TODO(boundary-epoch): Do NOT reconcile When against a `pre<year>` boundary
# variant or reproject an interval that straddles a boundary redesign here.
# Reprojection is a spatial operation with real error, not a syntax
# correction (design doc 2.4); such conflicts must keep raising.

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
