from lanka_data.correction.rules.ChangeRequiresTwoObservations import (
    ChangeRequiresTwoObservations,
)
from lanka_data.correction.rules.GeometryRejectsModifiers import (
    GeometryRejectsModifiers,
)
from lanka_data.correction.rules.ModifierRequiresKind import (
    ModifierRequiresKind,
)
from lanka_data.correction.rules.ResolveSelfType import ResolveSelfType
from lanka_data.correction.rules.SnapIntervalEndpoints import (
    SnapIntervalEndpoints,
)
from lanka_data.correction.rules.SnapObservationYear import (
    SnapObservationYear,
)
from lanka_data.correction.rules.ZoomIgnoredByRenderer import (
    ZoomIgnoredByRenderer,
)

DEFAULT_RULES = [
    SnapObservationYear(),
    SnapIntervalEndpoints(),
    GeometryRejectsModifiers(),
    ModifierRequiresKind(),
    ChangeRequiresTwoObservations(),
    ResolveSelfType(),
    ZoomIgnoredByRenderer(),
]

__all__ = [
    "SnapObservationYear",
    "SnapIntervalEndpoints",
    "GeometryRejectsModifiers",
    "ModifierRequiresKind",
    "ChangeRequiresTwoObservations",
    "ResolveSelfType",
    "ZoomIgnoredByRenderer",
    "DEFAULT_RULES",
]
