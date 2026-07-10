from functools import lru_cache

from lanka_data.api.fields.What import What
from lanka_data.api.fields.When import When
from lanka_data.api.fields.WhatWhenRegistry import WhatWhenRegistry

GROUP_KIND = {
    "special": "geometry",
    "census": "categorical",
    "election": "scalar",
    "election_summary": "scalar",
    "rivers": "scalar",
}


class Measurements:
    ANY = "ANY"

    @classmethod
    @lru_cache(maxsize=None)
    def _years_by_what(cls):
        years = {}
        for what, when in WhatWhenRegistry.pairs(When.available_values()):
            years.setdefault(what, set()).add(int(when))
        return {what: sorted(values) for what, values in years.items()}

    @classmethod
    @lru_cache(maxsize=None)
    def _group_of(cls, what):
        for group, values in What.available_groups().items():
            if what in values:
                return group
        return None

    @classmethod
    @lru_cache(maxsize=None)
    def _geometry_observation_years(cls):
        years = set()
        for what in What.available_groups().get("census", []):
            years.update(cls._years_by_what().get(what, []))
        return sorted(years)

    @classmethod
    def _base(cls, what):
        return what.split(What.COMBINE_DELIM, 1)[0]

    @classmethod
    def kind(cls, what):
        group = cls._group_of(cls._base(what))
        return GROUP_KIND.get(group, "scalar")

    @classmethod
    def observation_years(cls, what):
        base = cls._base(what)
        if cls.kind(base) == "geometry":
            return cls._geometry_observation_years()
        return cls._years_by_what().get(base)

    @classmethod
    def count_in_when(cls, what, when):
        years = cls.observation_years(what)
        if years is None:
            return None
        if when.is_interval:
            start, end = int(when.start), int(when.end)
            return len([y for y in years if start <= y <= end])
        return 1 if int(when.value) in years else 0
