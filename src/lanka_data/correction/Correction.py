from dataclasses import dataclass


@dataclass(frozen=True)
class Correction:
    field: str
    rule: str
    from_value: str
    to_value: str
    severity: str
    reason: str

    def to_dict(self):
        return {
            "field": self.field,
            "rule": self.rule,
            "from": self.from_value,
            "to": self.to_value,
            "severity": self.severity,
            "reason": self.reason,
        }

    @property
    def caption(self):
        return f"{self.from_value} \u2192 {self.to_value}"
