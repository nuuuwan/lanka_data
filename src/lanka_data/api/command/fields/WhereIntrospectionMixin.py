class WhereIntrospectionMixin:
    @classmethod
    def available_region_types(cls):
        return []

    @classmethod
    def available_operators(cls):
        return ["single", "comma", "range", "history", "zoom", "child_type"]

    @classmethod
    def available_examples(cls):
        return []

    @classmethod
    def available_values(cls):
        return cls.available_examples()

    @classmethod
    def describe(cls):
        return dict(
            name="where",
            values=cls.available_values(),
            region_types=cls.available_region_types(),
            operators=cls.available_operators(),
            token_pattern=r"[A-Za-z0-9:,@.\-]+",
        )
