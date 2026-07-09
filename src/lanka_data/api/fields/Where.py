import re
from dataclasses import dataclass

from lanka_data.api.command_errors.InvalidWhereError import InvalidWhereError
from lanka_data.api.fields.RegionFilter import RegionFilter
from lanka_data.api.fields.RegionTypeRegistry import RegionTypeRegistry
from lanka_data.api.fields.WhereIntrospectionMixin import (
    WhereIntrospectionMixin,
)


@dataclass(frozen=True)
class Where(WhereIntrospectionMixin):
    value: str

    TOP_RE = re.compile(r"#(\d+)$")

    @classmethod
    def strip_top(cls, token):
        return cls.TOP_RE.sub("", token or "")

    @classmethod
    def available_region_types(cls):
        values = set()
        for prefix_map in RegionTypeRegistry.PREFIX_MAPS.values():
            values.update(prefix_map.values())
        return sorted(values)

    @classmethod
    def available_examples(cls):
        return RegionTypeRegistry.EXAMPLES

    def __post_init__(self):
        if self.value == "":
            return
        if re.fullmatch(r"[A-Za-z0-9:,@.#\-]+", self.value or "") is None:
            raise InvalidWhereError(
                f"Invalid where: {self.value}", self.value
            )
        if ".." in self.value.replace("...", ""):
            raise InvalidWhereError(
                f"Invalid where: {self.value}", self.value
            )

    @property
    def base_value(self):
        return self.strip_top(self.value)

    @property
    def top(self):
        match = self.TOP_RE.search(self.value or "")
        if match is None:
            return None
        return int(match.group(1))

    @property
    def region_filter(self):
        if self.top is None:
            return None
        return RegionFilter(kind="rank", direction="Top", count=self.top)

    @property
    def parent_part(self):
        return self.base_value.split(":", 1)[0]

    @property
    def child_region_type(self):
        if ":" not in self.base_value:
            return None
        return self.base_value.split(":", 1)[1]

    @property
    def zoom(self):
        if "@" not in self.parent_part:
            return None
        return float(self.parent_part.split("@", 1)[1])

    def __str__(self):
        return self.value
