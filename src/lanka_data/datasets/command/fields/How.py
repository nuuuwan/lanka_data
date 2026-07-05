from dataclasses import dataclass
import re

from lanka_data.api.command.UnknownHowError import UnknownHowError


@dataclass(frozen=True)
class How:
    value: str

    BASE_LABELS = {
        "JSON": None,
        "Map": None,
        "Cartogram": "Cartogram (Population based)",
        "BarChart": "Bar Chart",
        "PieChart": "Pie Chart",
        "BumpChart": "Bump Chart",
        "None": None,
    }
    INTERVAL_BASES = {"BumpChart"}
    MODIFIERS = {
        "1st": {"label": "Most common", "rank": 0},
        "Top": {"label": "Most common", "rank": 0},
        "2nd": {"label": "2nd most common", "rank": 1},
        "3rd": {"label": "3rd most common", "rank": 2},
        "Bottom": {"label": "Least common", "rank": -1},
        "1stPct": {"label": "Most common share", "pct_rank": 0},
        "2ndPct": {"label": "2nd most common share", "pct_rank": 1},
        "3rdPct": {"label": "3rd most common share", "pct_rank": 2},
        "Change": {"label": "Change", "needs_interval": True},
        "Diversity": {"label": "Diversity"},
        "DiversityPew": {"label": "Pew diversity"},
        "Segregation": {"label": "Segregation"},
    }

    def __post_init__(self):
        if self.value == "":
            return
        if self.base not in self.BASE_LABELS:
            raise UnknownHowError(f"Unknown how: {self.value}", self.value)
        if self.modifier and self.modifier not in self.MODIFIERS:
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
    def needs_interval(self):
        return self.base in self.INTERVAL_BASES or bool(
            self.modifier_spec.get("needs_interval")
        )

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
