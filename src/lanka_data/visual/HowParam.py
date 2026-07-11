import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HowParam:
    name: str
    label: str
    description: str
    rank: int | None = None
    pct_rank: int | None = None
    needs_interval: bool = False

    @classmethod
    def list(cls):
        json_path = Path(__file__).parent / "how_params.json"
        with open(json_path) as f:
            data = json.load(f)
        return {
            key: cls(
                name=params["name"],
                label=params["label"],
                description=params["description"],
                rank=params.get("rank"),
                pct_rank=params.get("pct_rank"),
                needs_interval=params.get("needs_interval", False),
            )
            for key, params in data.items()
        }
