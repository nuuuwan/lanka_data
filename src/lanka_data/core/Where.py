import re

# Mapping from LK administrative district codes to EC electoral district codes.
# Vanni ED (EC-11) covers 4 northern districts.
_LK_TO_EC: dict[str, str] = {
    "LK-11": "EC-01",
    "LK-12": "EC-02",
    "LK-13": "EC-03",
    "LK-21": "EC-04",
    "LK-22": "EC-05",
    "LK-23": "EC-06",
    "LK-31": "EC-07",
    "LK-32": "EC-08",
    "LK-33": "EC-09",
    "LK-41": "EC-10",
    "LK-42": "EC-11",
    "LK-43": "EC-11",
    "LK-44": "EC-11",
    "LK-45": "EC-11",
    "LK-51": "EC-12",
    "LK-52": "EC-13",
    "LK-53": "EC-14",
    "LK-61": "EC-15",
    "LK-62": "EC-16",
    "LK-71": "EC-17",
    "LK-72": "EC-18",
    "LK-81": "EC-19",
    "LK-82": "EC-20",
    "LK-91": "EC-21",
    "LK-92": "EC-22",
}

# Mapping from LK province codes to their constituent EC electoral district codes.
_LK_PROVINCE_TO_ECS: dict[str, list[str]] = {
    "LK-1": ["EC-01", "EC-02", "EC-03"],
    "LK-2": ["EC-04", "EC-05", "EC-06"],
    "LK-3": ["EC-07", "EC-08", "EC-09"],
    "LK-4": ["EC-10", "EC-11"],
    "LK-5": ["EC-12", "EC-13", "EC-14"],
    "LK-6": ["EC-15", "EC-16"],
    "LK-7": ["EC-17", "EC-18"],
    "LK-8": ["EC-19", "EC-20"],
    "LK-9": ["EC-21", "EC-22"],
}


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
                if region_code in _LK_TO_EC:
                    ec = re.escape(_LK_TO_EC[region_code])
                    pat = rf"^{ec}$"
                elif region_code in _LK_PROVINCE_TO_ECS:
                    ecs = "|".join(
                        re.escape(e) for e in _LK_PROVINCE_TO_ECS[region_code]
                    )
                    pat = rf"^({ecs})$"
                else:
                    pat = r"^EC-\d{2}$"
            case "PollingDivisions":
                if re.fullmatch(r"EC-\d{2}", region_code):
                    pat = rf"^{rc}[A-Z]$"
                elif region_code in _LK_TO_EC:
                    ec = re.escape(_LK_TO_EC[region_code])
                    pat = rf"^{ec}[A-Z]$"
                elif region_code in _LK_PROVINCE_TO_ECS:
                    ecs = "|".join(
                        re.escape(e) for e in _LK_PROVINCE_TO_ECS[region_code]
                    )
                    pat = rf"^({ecs})[A-Z]$"
                else:
                    pat = r"^EC-\d{2}[A-Z]$"
            case "DSDs":
                if region_code == "LK":
                    pat = r"^LK-\d{4}$"
                else:
                    n = len(re.sub(r"\D", "", region_code))
                    pat = rf"^{rc}\d{{{4 - n}}}$"
            case "GNDs":
                if region_code == "LK":
                    pat = r"^LK-\d{7}$"
                else:
                    n = len(re.sub(r"\D", "", region_code))
                    pat = rf"^{rc}\d{{{7 - n}}}$"
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
