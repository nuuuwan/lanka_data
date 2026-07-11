from dataclasses import dataclass


@dataclass(frozen=True)
class What:
    value: str

    COMBINE_DELIM = "+"

    @property
    def is_combined(self):
        return self.COMBINE_DELIM in self.value

    @property
    def whats(self):
        return self.value.split(self.COMBINE_DELIM)

    def __str__(self):
        return self.value
