from dataclasses import dataclass

from lanka_data.api.command.UnknownWhatError import UnknownWhatError
from lanka_data.api.command.fields.WhatIntrospectionMixin import (
    WhatIntrospectionMixin,
)


@dataclass(frozen=True)
class What(WhatIntrospectionMixin):
    value: str

    VALUE_GROUPS = dict(special=["Empty"])

    def __post_init__(self):
        if self.value == "Help":
            return
        if self.value not in self.known_values():
            raise UnknownWhatError(
                f"Unknown what: {self.value}",
                self.value,
                self.suggestions(self.value),
            )

    @classmethod
    def known_values(cls):
        return cls.available_values()

    @classmethod
    def suggestions(cls, value):
        value_lower = value.lower()
        return [x for x in cls.known_values() if value_lower in x.lower()][:5]

    def __str__(self):
        return self.value
