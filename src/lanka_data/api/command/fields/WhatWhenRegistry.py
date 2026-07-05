class WhatWhenRegistry:
    PAIR_PROVIDERS = []

    @classmethod
    def set_pair_providers(cls, pair_providers):
        cls.PAIR_PROVIDERS = list(pair_providers)

    @classmethod
    def pairs(cls, when_values):
        pairs = []
        for provider in cls.PAIR_PROVIDERS:
            pairs.extend(provider(when_values))
        return sorted(set(pairs))
