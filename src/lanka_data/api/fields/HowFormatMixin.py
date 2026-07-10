class HowFormatMixin:
    @property
    def base_label(self):
        if self.base == "None":
            return None
        return self.BASE_LABELS.get(self.base, self.split_camel(self.base))

    @property
    def modifier_label(self):
        if self.modifier is None:
            return None
        if self.is_cluster:
            return f"cluster ({self.cluster_n} groups)"
        return self.modifier_spec.get(
            "label", self.split_camel(self.modifier)
        )

    def _format_with_filter(self):
        label = self.region_filter.label
        if self.base_label:
            return f"{self.base_label} ({label})"
        return label

    def _format_without_filter(self):
        if self.base_label:
            return f"{self.base_label} by {self.modifier_label}"
        return self.modifier_label

    def format(self):
        if not self.modifier:
            return self.base_label or self.split_camel(self.base)
        if self.region_filter is not None:
            return self._format_with_filter()
        return self._format_without_filter()
