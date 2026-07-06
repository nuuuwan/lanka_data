import operator
import re
from dataclasses import dataclass

RANK_RE = re.compile(r"^(Top|Bottom)(\d+)$")
THRESHOLD_RE = re.compile(r"^(.+?)(>=|<=|>|<)(-?\d*\.?\d+)$")
OPERATORS = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
}


@dataclass(frozen=True)
class RegionFilter:
    kind: str
    direction: str = None
    count: int = None
    category: str = None
    op: str = None
    threshold: float = None

    @classmethod
    def from_modifier(cls, modifier):
        if not modifier:
            return None
        rank_match = RANK_RE.fullmatch(modifier)
        if rank_match is not None:
            return cls(
                kind="rank",
                direction=rank_match.group(1),
                count=int(rank_match.group(2)),
            )
        return cls._threshold_from_modifier(modifier)

    @classmethod
    def _threshold_from_modifier(cls, modifier):
        threshold_match = THRESHOLD_RE.fullmatch(modifier)
        if threshold_match is None:
            return None
        return cls(
            kind="threshold",
            category=threshold_match.group(1),
            op=threshold_match.group(2),
            threshold=float(threshold_match.group(3)),
        )

    @property
    def label(self):
        if self.kind == "rank":
            return f"{self.direction} {self.count}"
        return f"{self.category} {self.op} {self.threshold:g}"

    def _apply_rank(self, rows):
        ordered = sorted(
            rows,
            key=lambda row: row.get("total_value", 0),
            reverse=self.direction == "Top",
        )
        return ordered[: self.count]

    def _apply_threshold(self, rows):
        op_func = OPERATORS[self.op]
        return [
            row
            for row in rows
            if op_func(
                row.get("pct_values", {}).get(self.category, 0),
                self.threshold,
            )
        ]

    def apply(self, rows):
        if self.kind == "rank":
            return self._apply_rank(rows)
        return self._apply_threshold(rows)
