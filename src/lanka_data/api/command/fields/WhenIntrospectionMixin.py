class WhenIntrospectionMixin:
    @classmethod
    def available_values(cls):
        return []

    @classmethod
    def available_intervals(cls):
        values = cls.available_values()
        return [
            f"{values[i]}-{values[j]}"
            for i in range(len(values))
            for j in range(i + 1, len(values))
        ]

    @classmethod
    def describe(cls):
        return dict(
            name="when",
            values=cls.available_values(),
            intervals=cls.available_intervals(),
            year_pattern=r"\d{4}",
            supports_interval=True,
        )
