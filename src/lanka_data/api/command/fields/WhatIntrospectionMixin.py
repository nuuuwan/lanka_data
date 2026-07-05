class WhatIntrospectionMixin:
    @classmethod
    def available_values(cls):
        values = []
        for group_values in cls.available_groups().values():
            values.extend(group_values)
        return sorted(set(values))

    @classmethod
    def available_groups(cls):
        return cls.VALUE_GROUPS

    @classmethod
    def describe(cls):
        groups = cls.available_groups()
        return dict(
            name="what",
            values=cls.available_values(),
            groups=groups,
            count=len(cls.available_values()),
        )
