class WhenRegistry:
    VALUE_PROVIDERS = []

    @classmethod
    def set_value_providers(cls, value_providers):
        cls.VALUE_PROVIDERS = list(value_providers)

    @classmethod
    def values(cls):
        values = []
        for provider in cls.VALUE_PROVIDERS:
            values.extend(provider())
        return sorted(set(values))
