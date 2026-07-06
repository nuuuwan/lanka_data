import re
from dataclasses import dataclass

from lanka_data.api.command_errors.UnknownHowError import UnknownHowError
from lanka_data.api.fields.HowFormatMixin import HowFormatMixin
from lanka_data.api.fields.HowIntrospectionMixin import HowIntrospectionMixin
from lanka_data.api.fields.HowRegistryMixin import HowRegistryMixin
from lanka_data.api.fields.RegionFilter import RegionFilter


@dataclass(frozen=True)
class How(HowFormatMixin, HowIntrospectionMixin, HowRegistryMixin):
    value: str

    def __post_init__(self):
        if self.value == "":
            return
        if self.base not in self.BASE_LABELS:
            raise UnknownHowError(f"Unknown how: {self.value}", self.value)
        if (
            self.modifier
            and self.modifier not in self.MODIFIERS
            and self.base not in self.CATEGORY_BASES
            and self.region_filter is None
        ):
            raise UnknownHowError(f"Unknown how: {self.value}", self.value)

    @staticmethod
    def split_camel(text):
        return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)

    @property
    def base(self):
        return self.value.split(":", 1)[0]

    @property
    def modifier(self):
        if ":" not in self.value:
            return None
        return self.value.split(":", 1)[1]

    @property
    def modifier_spec(self):
        return self.MODIFIERS.get(self.modifier, {})

    @property
    def rank(self):
        return self.modifier_spec.get("rank")

    @property
    def pct_rank(self):
        return self.modifier_spec.get("pct_rank")

    @property
    def region_filter(self):
        return RegionFilter.from_modifier(self.modifier)

    @property
    def category(self):
        if self.modifier is None or self.modifier in self.MODIFIERS:
            return None
        region_filter = self.region_filter
        if region_filter is not None:
            return region_filter.category
        return self.modifier

    @property
    def needs_interval(self):
        return self.base in self.INTERVAL_BASES or bool(
            self.modifier_spec.get("needs_interval")
        )

    @property
    def is_animation(self):
        return self.base in self.ANIMATION_BASE_TO_FRAME_BASE

    @property
    def frame_how(self):
        frame_base = self.ANIMATION_BASE_TO_FRAME_BASE.get(self.base)
        if frame_base is None:
            return self.value
        if self.modifier is None:
            return frame_base
        return f"{frame_base}:{self.modifier}"

    @property
    def needs_series(self):
        return self.base in self.SERIES_BASES

    def __str__(self):
        return self.value
