from dataclasses import dataclass


@dataclass(frozen=True)
class HowParam:
    name: str
    label: str
    description: str
    rank: int | None = None
    pct_rank: int | None = None
    needs_interval: bool = False
