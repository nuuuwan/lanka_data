from dataclasses import dataclass

from lanka_data.visual.HOW_PARAMS_DATA import HOW_PARAMS


@dataclass(frozen=True)
class HowParam:
    name: str
    label: str
    description: str
    rank: int | None = None
    pct_rank: int | None = None
    needs_interval: bool = False




