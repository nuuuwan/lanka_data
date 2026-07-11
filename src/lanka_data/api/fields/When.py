import re
from dataclasses import dataclass

from lanka_data.api.command_errors.InvalidWhenError import InvalidWhenError


@dataclass(frozen=True)
class When:
    value: str

    def __post_init__(self):
        if self.value == "":
            return
        if self._is_year(self.value):
            return
        if self._is_interval(self.value):
            return
        raise InvalidWhenError(f"Invalid when: {self.value}", self.value)

    @staticmethod
    def _is_year(value):
        return re.fullmatch(r"\d{4}", value or "") is not None

    @classmethod
    def _is_interval(cls, value):
        parts = (value or "").split("-")
        if len(parts) < 2:
            return False
        if not all(cls._is_year(part) for part in parts):
            return False
        years = [int(part) for part in parts]
        return all(a < b for a, b in zip(years, years[1:]))

    @property
    def is_interval(self):
        return self._is_interval(self.value)

    @property
    def years(self):
        return self.value.split("-") if self.is_interval else [self.value]

    @property
    def start(self):
        return self.years[0]

    @property
    def end(self):
        return self.years[-1]

    def __str__(self):
        return self.value
