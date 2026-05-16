import re


class Where:

    _LEVEL_ALIASES: dict[str, str] = {
        "eds": "ElectoralDistricts",
        "pds": "PollingDivisions",
        "provinces": "Provinces",
        "districts": "Districts",
        "electoraldistricts": "ElectoralDistricts",
        "pollingdivisions": "PollingDivisions",
        "dsds": "DSDs",
        "gnds": "GNDs",
    }

    def __init__(self, where_raw: str) -> None:
        if ":" in where_raw:
            code, raw_level = where_raw.split(":", 1)
            self.region_code = code
            self.level = self._resolve_level(raw_level)
            self._pattern: re.Pattern | None = self._make_pattern(
                self.region_code, self.level
            )
        else:
            self.region_code = where_raw
            self.level = None
            self._pattern = None

    @classmethod
    def _resolve_level(cls, level: str) -> str:
        return cls._LEVEL_ALIASES.get(level.lower(), level)

    @staticmethod
    def _make_pattern(region_code: str, level_canon: str) -> re.Pattern:
        rc = re.escape(region_code)
        match level_canon:
            case "Provinces":
                pat = r"^LK-\d$"
            case "Districts":
                if region_code == "LK":
                    pat = r"^LK-\d{2}$"
                else:
                    pat = rf"^{rc}\d$"
            case "ElectoralDistricts":
                pat = r"^EC-\d{2}$"
            case "PollingDivisions":
                if re.fullmatch(r"EC-\d{2}", region_code):
                    pat = rf"^{rc}[A-Z]$"
                else:
                    pat = r"^EC-\d{2}[A-Z]$"
            case "DSDs":
                pat = r"^LK\d{4}$"
            case "GNDs":
                pat = r"^LK\d{7}$"
            case _:
                raise ValueError(f"Unknown level: {level_canon!r}")
        return re.compile(pat)

    def matches(self, entity_id: str) -> bool:
        if self.region_code == "*":
            return True
        if self.level is None:
            return entity_id == self.region_code
        assert self._pattern is not None
        return bool(self._pattern.fullmatch(entity_id))
