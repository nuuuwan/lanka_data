class WhatIntrospectionMixin:
    @classmethod
    def available_values(cls):
        return cls.known_values()

    @classmethod
    def available_groups(cls):
        election = cls.election_values()
        return dict(
            special=["Empty"],
            census=sorted(set(cls.census_values())),
            election=election,
            election_summary=[x + "Summary" for x in election],
        )

    @classmethod
    def describe(cls):
        groups = cls.available_groups()
        return dict(
            name="what",
            values=cls.available_values(),
            groups=groups,
            count=len(cls.available_values()),
        )
