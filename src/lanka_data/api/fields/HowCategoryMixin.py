import re


class HowCategoryMixin:
    @property
    def category(self):
        if (
            self.modifier is None
            or self.modifier in self.MODIFIERS
            or self.base in self.PAIR_CATEGORY_BASES
            or self.is_cluster
        ):
            return None
        region_filter = self.region_filter
        if region_filter is not None:
            return region_filter.category
        return self.modifier

    @property
    def categories(self):
        if self.base not in self.PAIR_CATEGORY_BASES:
            return []
        if self.modifier is None:
            return []
        return re.split(r"[:+]", self.modifier)
