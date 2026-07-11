import os
from dataclasses import dataclass, field
from functools import cache

from utils_future import JSONFile


@dataclass(frozen=True)
class WhatLabel:
    label: str
    description: str
    category_labels: list[str] = field(default_factory=list)

    @classmethod
    def definitions_file_path(cls) -> str:
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "what_labels.json",
        )

    @classmethod
    @cache
    def list(cls) -> list["WhatLabel"]:
        definitions = JSONFile(cls.definitions_file_path()).read()
        return [
            cls(
                label=definition["label"],
                description=definition["description"],
                category_labels=list(definition["category_labels"]),
            )
            for definition in definitions
        ]

    @classmethod
    @cache
    def idx(cls) -> dict[str, "WhatLabel"]:
        return {what_label.label: what_label for what_label in cls.list()}

    @classmethod
    def from_label(cls, label: str) -> "WhatLabel":
        return cls.idx().get(label)

    def __str__(self) -> str:
        return self.label
