import re
from dataclasses import dataclass

from lanka_data.api.command_errors.UnknownHowError import UnknownHowError
from lanka_data.api.fields.HowCategoryMixin import HowCategoryMixin
from lanka_data.api.fields.HowFormatMixin import HowFormatMixin
from lanka_data.api.fields.HowRegistryMixin import HowRegistryMixin
from lanka_data.api.fields.RegionFilter import RegionFilter

CLUSTER_RE = re.compile(r"^Cluster(?:-(\d+))?$")
DEFAULT_CLUSTER_N = 5


@dataclass(frozen=True)
class How(HowCategoryMixin, HowFormatMixin, HowRegistryMixin):
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
            and self.base not in self.PAIR_CATEGORY_BASES
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
        if self.modifier in self.MODIFIERS:
            return None
        return RegionFilter.from_modifier(self.modifier)

    @property
    def is_top3(self):
        return self.modifier == "Top3"

    @property
    def is_cluster(self):
        if self.modifier is None:
            return False
        return CLUSTER_RE.fullmatch(self.modifier) is not None

    @property
    def cluster_n(self):
        if self.modifier is None:
            return None
        match = CLUSTER_RE.fullmatch(self.modifier)
        if match is None:
            return None
        return int(match.group(1)) if match.group(1) else DEFAULT_CLUSTER_N

    @property
    def needs_interval(self):
        return self.base in self.INTERVAL_BASES or bool(
            self.modifier_spec.get("needs_interval")
        )

    @property
    def needs_series(self):
        return self.base in self.SERIES_BASES

    def __str__(self):
        return self.value
