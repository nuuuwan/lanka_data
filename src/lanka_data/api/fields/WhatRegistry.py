class WhatRegistry:
    GROUP_PROVIDERS = {}

    @classmethod
    def set_group_providers(cls, group_providers):
        cls.GROUP_PROVIDERS = dict(group_providers)

    @classmethod
    def groups(cls):
        return {
            name: sorted(set(provider()))
            for name, provider in cls.GROUP_PROVIDERS.items()
        }
