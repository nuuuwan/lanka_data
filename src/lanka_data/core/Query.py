from .QueryBase import QueryBase


class Query(QueryBase):

    _DOMAIN_GIG2_PREFIX: dict[str, str] = {
        "population": "population",
        "election": "government-elections",
        "economy": "economy",
        "education": "education",
        "social": "social",
    }

    _DEFAULT_SUB: dict[str, str] = {
        "population": "total",
    }

    _COMPOSITE_SUB_COMPONENTS: frozenset[str] = frozenset(
        {"parties", "summary"}
    )

    _SHORT_LABELS: dict[str, str] = {
        "agegroup": "populationagegroup",
        "ethnicity": "populationethnicity",
        "gender": "populationgender",
        "maritalstatus": "populationmaritalstatus",
        "religion": "populationreligion",
        # social/household short labels (also present in Census2024 datasets)
        "cookingfuel": "socialhouseholdcookingfuel",
        "drinkingwater": "socialhouseholdsourceofdrinkingwater",
        "households": "socialhouseholdnumberofhouseholds",
        "lighting": "socialhouseholdlighting",
        "toilet": "socialhouseholdtoiletfacilities",
    }

    @staticmethod
    def _pop_sub_component(
        sub_parts: list[str],
        composite_subs: frozenset[str],
    ) -> tuple[list[str], str | None]:
        if sub_parts and sub_parts[-1] in composite_subs:
            return sub_parts[:-1], sub_parts[-1]
        return sub_parts, None

    @classmethod
    def _make_what_info(
        cls, prefix: str, domain: str, sub_parts: list[str]
    ) -> str:
        if sub_parts:
            return prefix + "-" + "-".join(sub_parts)
        default = cls._DEFAULT_SUB.get(domain)
        return f"{prefix}-{default}" if default else prefix

    @classmethod
    def _resolve_gig2_key(cls, parts: list) -> tuple[str | None, str | None]:
        direct = cls._SHORT_LABELS.get(parts[0]) if len(parts) == 1 else None
        if direct is not None:
            return direct, None
        prefix = cls._DOMAIN_GIG2_PREFIX.get(parts[0])
        if prefix is None:
            return None, None
        sub_parts, sub_component = cls._pop_sub_component(
            parts[1:], cls._COMPOSITE_SUB_COMPONENTS
        )
        what_info = cls._make_what_info(prefix, parts[0], sub_parts)
        return cls.normalize(what_info), sub_component

    def gig2_key(self) -> tuple[str | None, str | None]:
        """Return (normalized_what_info_key, sub_component_or_None)."""
        if self.is_wildcard_what or not self.what_parts:
            return None, None
        parts = [p.lower() for p in self.what_parts]
        return self._resolve_gig2_key(parts)
