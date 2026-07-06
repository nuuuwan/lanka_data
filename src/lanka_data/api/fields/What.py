from dataclasses import dataclass

from lanka_data.api.command_errors.UnknownWhatError import UnknownWhatError
from lanka_data.api.fields.WhatIntrospectionMixin import (
    WhatIntrospectionMixin,
)
from lanka_data.api.fields.WhatRegistry import WhatRegistry


@dataclass(frozen=True)
class What(WhatIntrospectionMixin):
    value: str

    VALUE_GROUPS = {"special": ["Empty"]}
    COMBINE_DELIM = "+"

    @classmethod
    def available_groups(cls):
        groups = dict(cls.VALUE_GROUPS)
        groups.update(WhatRegistry.groups())
        return groups

    def __post_init__(self):
        if self.value == "Help":
            return
        for part in self.whats:
            self._validate_part(part)

    @classmethod
    def _validate_part(cls, part):
        if part not in cls.known_values():
            raise UnknownWhatError(
                f"Unknown what: {part}",
                part,
                cls.suggestions(part),
            )

    @property
    def is_combined(self):
        return self.COMBINE_DELIM in self.value

    @property
    def whats(self):
        return self.value.split(self.COMBINE_DELIM)

    @classmethod
    def known_values(cls):
        return cls.available_values()

    @classmethod
    def suggestions(cls, value):
        value_lower = value.lower()
        return [x for x in cls.known_values() if value_lower in x.lower()][:5]

    def __str__(self):
        return self.value
