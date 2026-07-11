import os
from dataclasses import dataclass, field
from functools import cache
from typing import Optional

from utils_future import JSONFile


@dataclass(frozen=True)
class WhatLabel:
    label: str
    description: str
    category_labels: list[str] = field(default_factory=list)

    @property
    def group(self) -> str:
        return '-'.join([label for label in self.category_labels])

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
        labels = [
            cls(
                label=definition["label"],
                description=definition["description"],
                category_labels=list(definition["category_labels"]),
            )
            for definition in definitions
        ]
        labels.sort(key=lambda x: (x.group, x.label))
        return labels

    @classmethod
    @cache
    def idx(cls) -> dict[str, "WhatLabel"]:
        return {what_label.label: what_label for what_label in cls.list()}

    @classmethod
    def from_label(cls, label: str) -> Optional["WhatLabel"]:
        return cls.idx().get(label)

    def __str__(self) -> str:
        return self.label

    @classmethod
    @cache
    def list_by_group(cls):
        groups = {}
        for what_label in WhatLabel.list():
            group = what_label.group
            if group not in groups:
                groups[group] = []
            groups[group].append(what_label)
        return groups
