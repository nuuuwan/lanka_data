import re


class Query:

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
    }

    def __init__(self, path: str) -> None:
        parts = path.strip("/").split("/")
        if len(parts) != 3:
            raise ValueError(
                "Query must have exactly 3 segments"
                f" (/<what>/<when>/<where>), got: {path!r}"
            )
        self.what_raw = parts[0]
        self.when_raw = parts[1]
        self.where_raw = parts[2]

    @staticmethod
    def normalize(s: str) -> str:
        """Lower-case with all separators (-, _, space) stripped."""
        return re.sub(r"[-_\s]", "", s).lower()

    @property
    def what_parts(self) -> list[str]:
        return self.what_raw.split(":") if self.what_raw != "*" else []

    @property
    def is_wildcard_what(self) -> bool:
        return self.what_raw == "*"

    @property
    def is_wildcard_when(self) -> bool:
        return self.when_raw == "*"

    @property
    def is_wildcard_where(self) -> bool:
        return self.where_raw == "*"

    @property
    def year(self) -> str | None:
        if self.is_wildcard_when:
            return None
        return self.when_raw[:4]

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
    def _resolve_gig2_key(
        cls, parts: list
    ) -> tuple[str | None, str | None]:
        direct = (
            cls._SHORT_LABELS.get(parts[0]) if len(parts) == 1 else None
        )
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
