from dataclasses import dataclass
import re

from lanka_data.api.command.InvalidWhereError import InvalidWhereError
from lanka_data.api.command.fields.WhereIntrospectionMixin import (
    WhereIntrospectionMixin,
)
from lanka_data.datasets.region.RegionTypeUtils import RegionTypeUtils


@dataclass(frozen=True)
class Where(WhereIntrospectionMixin):
    value: str

    @classmethod
    def available_region_types(cls):
        values = set()
        for prefix_map in RegionTypeUtils._PREFIX_MAPS.values():
            values.update(prefix_map.values())
        return sorted(values)

    @classmethod
    def available_examples(cls):
        return [
            "LK",
            "LK:district",
            "LK-1,LK-2",
            "LK-1...LK-2",
            "LK-pre1959",
            "LK-1127025@20",
        ]

    def __post_init__(self):
        if self.value == "":
            return
        if re.fullmatch(r"[A-Za-z0-9:,@.\-]+", self.value or "") is None:
            raise InvalidWhereError(
                f"Invalid where: {self.value}", self.value
            )
        if ".." in self.value.replace("...", ""):
            raise InvalidWhereError(
                f"Invalid where: {self.value}", self.value
            )

    @property
    def parent_part(self):
        return self.value.split(":", 1)[0]

    @property
    def child_region_type(self):
        if ":" not in self.value:
            return None
        return self.value.split(":", 1)[1]

    @property
    def zoom(self):
        if "@" not in self.parent_part:
            return None
        return float(self.parent_part.split("@", 1)[1])

    def __str__(self):
        return self.value
