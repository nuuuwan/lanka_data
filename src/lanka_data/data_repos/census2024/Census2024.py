import pathlib
import urllib.request

from ...core.Query import Query
from ...core.Where import Where

_BASE_URL = (
    "https://raw.githubusercontent.com" "/nuuuwan/lk_census_2024/main/data"
)

_META_COLS: frozenset[str] = frozenset(
    {
        "region_id",
        "region_name",
        "region_name_in_data",
        "region_ent_type",
    }
)


class Census2024:
    _YEAR = "2024"

    _DATASETS: dict[str, str] = {
        "Housing": ("GN_housing_excel" "/Occupied-Housing-Units/data.tsv"),
        "AgeGroup": (
            "GN_population_excel" "/Population-by-Age-Group/data.tsv"
        ),
        "Gender": ("GN_population_excel" "/Population-by-Sex/data.tsv"),
        "Households": ("HH_GND_excel/Number-of-Households/data.tsv"),
        "DrinkingWater": (
            "HH_GND_excel" "/Main-Source-of-Drinking-Water/data.tsv"
        ),
        "CookingFuel": (
            "HH_GND_excel"
            "/Main-Source-of-EnergyFuel-Used-for-Cooking"
            "/data.tsv"
        ),
        "Lighting": ("HH_GND_excel/Main-Source-of-Lighting/data.tsv"),
        "Toilet": ("HH_GND_excel/Toilet-Facilities/data.tsv"),
        "Ethnicity": (
            "Population-Preliminary-Report"
            "/Population-by-ethnicity/data.tsv"
        ),
        "Religion": (
            "Population-Preliminary-Report" "/Population-by-religion/data.tsv"
        ),
    }

    _COL_RENAMES: dict[str, dict[str, str]] = {
        "Housing": {
            "occupied_housing_units": "OccupiedHousingUnits",
        },
        "AgeGroup": {
            "total": "Total",
            "age_0_14": "Age0To14",
            "age_15_59": "Age15To59",
            "age_60_64": "Age60To64",
            "age_65_and_above": "Age65AndAbove",
        },
        "Gender": {
            "total": "Total",
            "male": "Male",
            "female": "Female",
        },
        "Households": {
            "n_households": "Households",
        },
        "DrinkingWater": {
            "protected_well": "ProtectedWell",
            "semi_protected_well": "SemiProtectedWell",
            "unprotected_well": "UnprotectedWell",
            "tube_well": "TubeWell",
            "spring_fountain": "SpringFountain",
            "pipe_borne_nwsdb": "PipeBorneNWSDB",
            "pipe_borne_local_authority": "PipeBorneLocalAuthority",
            "pipe_borne_community": "PipeBorneCommunity",
            "pipe_borne_private": "PipeBornePrivate",
            "tank_river_stream": "TankRiverStream",
            "rain_water": "RainWater",
            "bottled_water": "BottledWater",
            "filter_ro": "FilterRO",
            "bowser": "Bowser",
            "other": "Other",
        },
        "CookingFuel": {
            "firewood": "Firewood",
            "kerosene": "Kerosene",
            "gas": "Gas",
            "electricity": "Electricity",
            "sawdust_paddy_husk": "SawdustOrPaddyHusk",
            "bio_gas": "BioGas",
            "other": "Other",
            "not_relevant": "NotRelevant",
        },
        "Lighting": {
            "electricity_grid": "ElectricityGrid",
            "kerosene_lamp": "KeroseneLamp",
            "solar_grid": "SolarGrid",
            "solar_standalone": "SolarStandalone",
            "bio_gas": "BioGas",
            "generator": "Generator",
            "other": "Other",
        },
        "Toilet": {
            "within_unit_exclusive": "WithinUnitExclusive",
            "within_unit_shared": "WithinUnitShared",
            "within_premises_exclusive": "WithinPremisesExclusive",
            "within_premises_shared": "WithinPremisesShared",
            "no_toilet_sharing": "NoToiletSharing",
            "common_public": "CommonPublic",
            "none": "NoToilet",
        },
        "Ethnicity": {
            "total": "Total",
            "sinhalese": "Sinhalese",
            "sri_lanka_tamil": "SriLankaTamil",
            "indian_tamil_or_malaiyaga_thamilar": "IndianTamil",
            "sri_lanka_moor_or_muslim": "SriLankaMoor",
            "burgher": "Burgher",
            "malay": "Malay",
            "sri_lanka_chetty": "SriLankaChetty",
            "bharatha": "Bharatha",
            "veddhas": "Veddha",
            "other": "Other",
        },
        "Religion": {
            "total": "Total",
            "buddhist": "Buddhist",
            "hindu": "Hindu",
            "islam": "Islam",
            "roman_catholic": "RomanCatholic",
            "other_christian": "OtherChristian",
            "other": "Other",
        },
    }

    _CACHE_DIR = pathlib.Path("/tmp/lanka_data")
    _tsv_cache: dict[str, list] = {}

    @staticmethod
    def _coerce(v: str):
        try:
            return int(v)
        except (ValueError, TypeError):
            pass
        try:
            return float(v)
        except (ValueError, TypeError):
            return v

    @classmethod
    def _load_tsv_text(cls, label: str) -> str:
        safe = label.replace(":", "_")
        cache_file = cls._CACHE_DIR / "census2024" / f"{safe}.tsv"
        if cache_file.exists():
            return cache_file.read_text()
        url = f"{_BASE_URL}/{cls._DATASETS[label]}"
        with urllib.request.urlopen(url) as r:
            text = r.read().decode()
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(text)
        return text

    @staticmethod
    def _parse_tsv(text: str) -> list[dict]:
        lines = text.splitlines()
        headers = lines[0].split("\t")
        return [
            dict(zip(headers, line.split("\t"))) for line in lines[1:] if line
        ]

    @classmethod
    def _fetch_tsv(cls, label: str) -> list[dict]:
        if label not in cls._tsv_cache:
            cls._tsv_cache[label] = cls._parse_tsv(cls._load_tsv_text(label))
        return cls._tsv_cache[label]

    @classmethod
    def _row_data(cls, row: dict) -> dict:
        return {
            k: cls._coerce(v) for k, v in row.items() if k not in _META_COLS
        }

    @classmethod
    def _label_has_entity(cls, label: str, where: Where) -> bool:
        if where.region_code == "*":
            return True
        for row in cls._fetch_tsv(label):
            if where.matches(row.get("region_id", "")):
                return True
        return False

    @classmethod
    def _query_label(cls, label: str, where: Where) -> dict:
        renames = cls._COL_RENAMES.get(label, {})
        result = {}
        for row in cls._fetch_tsv(label):
            eid = row.get("region_id", "")
            if not where.matches(eid):
                continue
            raw = cls._row_data(row)
            data = {renames.get(k, k): v for k, v in raw.items()}
            result[eid] = (
                next(iter(data.values())) if len(data) == 1 else data
            )
        if where.level is not None:
            return result
        return next(iter(result.values()), result)

    @classmethod
    def _entities(cls, label: str) -> dict:
        rows = cls._fetch_tsv(label)
        return {
            "entities": sorted(
                r["region_id"] for r in rows if r.get("region_id")
            )
        }

    @classmethod
    def _catalog(cls, q: Query) -> dict:
        if q.year is not None and q.year != cls._YEAR:
            return {"measurements": []}
        where = Where(q.where_raw)
        labels = [
            lbl for lbl in cls._DATASETS if cls._label_has_entity(lbl, where)
        ]
        return {"measurements": sorted(labels)}

    @classmethod
    def _wildcard_when(cls, q: Query) -> dict:
        if q.is_wildcard_where:
            return {"years": [cls._YEAR]}
        if cls._label_has_entity(q.what_raw, Where(q.where_raw)):
            return {"years": [cls._YEAR]}
        return {}

    @classmethod
    def _query_for_where(cls, q: Query) -> dict:
        if q.is_wildcard_where:
            return cls._entities(q.what_raw)
        return cls._query_label(q.what_raw, Where(q.where_raw))

    @classmethod
    def _query_for_time(cls, q: Query) -> dict:
        if q.is_wildcard_when:
            return cls._wildcard_when(q)
        if q.year != cls._YEAR:
            return {}
        return cls._query_for_where(q)

    @classmethod
    def query(cls, q: Query) -> dict:
        if q.is_wildcard_what:
            return cls._catalog(q)
        if q.what_raw not in cls._DATASETS:
            return {}
        return cls._query_for_time(q)

    @classmethod
    def handles(cls, q: Query) -> bool:
        return q.what_raw in cls._DATASETS
