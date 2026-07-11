import os
from dataclasses import dataclass

from utils_future import JSONFile


@dataclass(frozen=True)
class HowParam:
    label: str
    description: str
    rank: int | None = None
    pct_rank: int | None = None
    needs_interval: bool = False

    @classmethod
    def list(cls):
        data = JSONFile(
            os.path.join('src', 'lanka_data', 'visual', 'how_params.json')
        ).read()

        return [
            cls(
                label=params["label"],
                description=params["description"],
                rank=params.get("rank"),
                pct_rank=params.get("pct_rank"),
                needs_interval=params.get("needs_interval", False),
            )
            for params in data
        ]
