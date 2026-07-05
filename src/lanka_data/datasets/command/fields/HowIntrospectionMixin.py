class HowIntrospectionMixin:
    @classmethod
    def available_bases(cls):
        return sorted(cls.BASE_LABELS)

    @classmethod
    def available_modifiers(cls):
        return sorted(cls.MODIFIERS)

    @classmethod
    def available_values(cls):
        values = set(cls.available_bases())
        for base in cls.available_bases():
            for modifier in cls.available_modifiers():
                values.add(f"{base}:{modifier}")
        return sorted(values)

    @classmethod
    def interval_values(cls):
        return [
            value
            for value in cls.available_values()
            if cls(value).needs_interval
        ]

    @classmethod
    def describe(cls):
        return dict(
            name="how",
            values=cls.available_values(),
            bases=cls.available_bases(),
            modifiers=cls.available_modifiers(),
            interval_values=cls.interval_values(),
        )
