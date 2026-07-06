import re
from dataclasses import dataclass

from lanka_data.api.command_errors.UnknownHowError import UnknownHowError
from lanka_data.api.fields.HowIntrospectionMixin import HowIntrospectionMixin
from lanka_data.api.fields.HowRegistryMixin import HowRegistryMixin


@dataclass(frozen=True)
class How(HowIntrospectionMixin, HowRegistryMixin):
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
    def category(self):
        if self.modifier is None or self.modifier in self.MODIFIERS:
            return None
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
    def base_label(self):
        if self.base == "None":
            return None
        return self.BASE_LABELS.get(self.base, self.split_camel(self.base))

    @property
    def modifier_label(self):
        if self.modifier is None:
            return None
        return self.modifier_spec.get(
            "label", self.split_camel(self.modifier)
        )

    def format(self):
        if not self.modifier:
            return self.base_label or self.split_camel(self.base)
        if self.base_label:
            return f"{self.base_label} by {self.modifier_label}"
        return self.modifier_label

    def __str__(self):
        return self.value
