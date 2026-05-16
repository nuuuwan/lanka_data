import functools
import json
import pathlib
import sys
import urllib.request

from ...core.Query import Query
from ...core.Where import Where


class GIG2:

    _API_URL = "https://api.github.com/repos/nuuuwan/gig-data/contents/gig2"

    _SUMMARY_COLS: frozenset[str] = frozenset(
        {
            "valid",
            "rejected",
            "polled",
            "electors",
            "turnout",
            "p_value",
            "seats",
        }
    )

    _LABELS: dict[str, str] = {
        "economyeconomicactivity": "Economy",
        "educationeducationalattainment": "Education",
        "governmentelectionslocalgovernment": "Election:LocalGovernment",
        "governmentelectionsparliamentary": "Election:Parliamentary",
        "governmentelectionspresidential": "Election:Presidential",
        "populationagegroup": "Population:AgeGroup",
        "populationethnicity": "Population:Ethnicity",
        "populationgender": "Population:Gender",
        "populationmaritalstatus": "Population:MaritalStatus",
        "populationreligion": "Population:Religion",
        "populationtotal": "Population",
        "socialhouseholdcommunicationitems": "Social:Communication",
        "socialhouseholdcookingfuel": "Social:CookingFuel",
        "socialhouseholdfloortype": "Social:Floor",
        "socialhouseholdlighting": "Social:Lighting",
        "socialhouseholdlivingquarters": "Social:LivingQuarters",
        "socialhouseholdnumberofhouseholds": "Social:Households",
        "socialhouseholdnumberofpersons": "Social:Persons",
        "socialhouseholdnumberofrooms": "Social:Rooms",
        "socialhouseholdoccupationstatus": "Social:Occupation",
        "socialhouseholdownershipstatus": "Social:Ownership",
        "socialhouseholdrelationshiptohead": "Social:HeadRelationship",
        "socialhouseholdrooftype": "Social:Roof",
        "socialhouseholdsolidwastedisposal": "Social:SolidWaste",
        "socialhouseholdsourceofdrinkingwater": "Social:DrinkingWater",
        "socialhouseholdstructure": "Social:Structure",
        "socialhouseholdtoiletfacilities": "Social:Toilet",
        "socialhouseholdtypeofunit": "Social:UnitType",
        "socialhouseholdwalltype": "Social:Wall",
        "socialhouseholdyearofconstruction": "Social:YearBuilt",
    }

    _CACHE_DIR = pathlib.Path("/tmp/lanka_data")

    _index: dict | None = None
    _tsv_cache: dict[str, list] = {}

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    @classmethod
    def _parse_index(cls, files: list[dict]) -> dict:
        index: dict[str, list[dict]] = {}
        for f in files:
            name: str = f.get("name", "")
            if not name.endswith(".tsv"):
                continue
            stem = name[:-4]
            parts = stem.split(".")
            if len(parts) != 3:
                continue
            what_info, where_type, year = parts
            key = Query.normalize(what_info)
            index.setdefault(key, []).append(
                {
                    "year": year,
                    "where_type": where_type,
                    "url": f["download_url"],
                }
            )
        return index

    @classmethod
    def _fetch_index(cls) -> dict:
        req = urllib.request.Request(
            cls._API_URL,
            headers={"Accept": "application/vnd.github.v3+json"},
        )
        with urllib.request.urlopen(req) as resp:
            files: list[dict] = json.loads(resp.read())
        return cls._parse_index(files)

    @classmethod
    def _build_index(cls) -> dict:
        if cls._index is not None:
            return cls._index
        cache_file = cls._CACHE_DIR / "data_repo" / "index.json"
        if cache_file.exists():
            with cache_file.open() as f:
                cls._index = json.load(f)
            return cls._index
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cls._index = cls._fetch_index()
        with cache_file.open("w") as f:
            json.dump(cls._index, f)
        return cls._index

    # ------------------------------------------------------------------
    # TSV fetching
    # ------------------------------------------------------------------

    @classmethod
    def _download_tsv(cls, url: str) -> list[dict[str, str]]:
        with urllib.request.urlopen(url) as resp:
            content = resp.read().decode("utf-8")
        lines = content.splitlines()
        rows: list[dict[str, str]] = []
        if lines:
            headers = lines[0].split("\t")
            for line in lines[1:]:
                if line.strip():
                    rows.append(dict(zip(headers, line.split("\t"))))
        return rows

    @classmethod
    def _fetch_tsv(cls, url: str) -> list[dict[str, str]]:
        if url in cls._tsv_cache:
            return cls._tsv_cache[url]
        fname = url.split("/")[-1]
        cache_file = cls._CACHE_DIR / "data_repo" / fname
        if cache_file.exists():
            with cache_file.open() as f:
                rows = json.load(f)
        else:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            rows = cls._download_tsv(url)
            with cache_file.open("w") as f:
                json.dump(rows, f)
        cls._tsv_cache[url] = rows
        return rows

    # ------------------------------------------------------------------
    # Value coercion and row formatting
    # ------------------------------------------------------------------

    @staticmethod
    @functools.cache
    def _coerce(v: str):
        try:
            f = float(v)
            return int(f) if f == int(f) else f
        except (ValueError, OverflowError):
            return v

    @classmethod
    def _format_row(
        cls, row: dict[str, str], sub_component: str | None
    ) -> dict:
        data = {k: cls._coerce(v) for k, v in row.items() if k != "entity_id"}
        if sub_component == "summary":
            return {k: v for k, v in data.items() if k in cls._SUMMARY_COLS}
        if sub_component == "parties":
            return {
                k: v for k, v in data.items() if k not in cls._SUMMARY_COLS
            }
        return data

    # ------------------------------------------------------------------
    # Entry querying
    # ------------------------------------------------------------------

    @classmethod
    def _query_entry(
        cls, entry: dict, where: Where, sub_component: str | None
    ) -> dict:
        rows = cls._fetch_tsv(entry["url"])
        result: dict = {}
        for row in rows:
            entity_id = row.get("entity_id", "")
            if not where.matches(entity_id):
                continue
            formatted = cls._format_row(row, sub_component)
            if where.level is None:
                return formatted
            result[entity_id] = formatted
        return result

    @staticmethod
    def _pick_entry(entries: list[dict], where_raw: str) -> dict:
        if len(entries) == 1:
            return entries[0]
        ec_context = where_raw.startswith("EC") or any(
            lv in where_raw
            for lv in (
                "ElectoralDistricts",
                "EDs",
                "PollingDivisions",
                "PDs",
            )
        )
        preferred = "regions-ec" if ec_context else "regions"
        for e in entries:
            if e["where_type"] == preferred:
                return e
        return entries[0]

    @staticmethod
    def _warn(msg: str) -> None:
        print(f"Warning: {msg}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Catalog query (wildcard what)
    # ------------------------------------------------------------------

    @classmethod
    def _key_matches(
        cls, entries: list, year: str | None, where: Where
    ) -> bool:
        year_entries = (
            [e for e in entries if e["year"] == year] if year else entries
        )
        for entry in year_entries:
            rows = cls._fetch_tsv(entry["url"])
            for row in rows:
                if where.matches(row.get("entity_id", "")):
                    return True
        return False

    @classmethod
    def _query_catalog(cls, q: Query, index: dict) -> dict:
        year = q.year
        where = Where(q.where_raw)
        keys = [
            key
            for key, entries in sorted(index.items())
            if cls._key_matches(entries, year, where)
        ]
        labels = [cls._LABELS.get(k, k) for k in keys]
        return {"measurements": sorted(set(labels))}

    # ------------------------------------------------------------------
    # Time-scoped queries
    # ------------------------------------------------------------------

    @classmethod
    def _query_all_years(
        cls, entries: list, where: Where, sub_component: str | None
    ) -> dict:
        result: dict = {}
        for entry in sorted(entries, key=lambda e: e["year"]):
            data = cls._query_entry(entry, where, sub_component)
            if data:
                result[entry["year"]] = data
        return result

    @classmethod
    def _query_single_year(
        cls, q: Query, entries: list, sub_component: str | None
    ) -> dict:
        year = q.year
        year_entries = [e for e in entries if e["year"] == year]
        if not year_entries:
            available = sorted(e["year"] for e in entries)
            cls._warn(
                f"No data for {q.what_raw!r} in year {year!r}. "
                f"Available years: {available}."
            )
            return {}
        entry = cls._pick_entry(year_entries, q.where_raw)
        return cls._query_entry(entry, Where(q.where_raw), sub_component)

    @classmethod
    def _query_by_time(
        cls, q: Query, entries: list, sub_component: str | None
    ) -> dict:
        where = Where(q.where_raw)
        if q.is_wildcard_when:
            return cls._query_all_years(entries, where, sub_component)
        return cls._query_single_year(q, entries, sub_component)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    @classmethod
    def _get_entries(
        cls, q: Query, index: dict
    ) -> tuple[list | None, str | None]:
        norm_key, sub_component = q.gig2_key()
        if norm_key is None:
            return None, None
        entries = index.get(norm_key, [])
        if not entries:
            cls._warn(f"No data found for {q.what_raw!r}.")
            return None, None
        return entries, sub_component

    @classmethod
    def _run_query(cls, q: Query, index: dict) -> dict:
        if q.is_wildcard_what:
            return cls._query_catalog(q, index)
        entries, sub_component = cls._get_entries(q, index)
        if entries is None:
            return {}
        return cls._query_by_time(q, entries, sub_component)

    @classmethod
    def query(cls, q: Query) -> dict:
        key = f"{q.what_raw}_{q.when_raw}_{q.where_raw}"
        safe = key.replace(":", "-").replace(" ", "_")
        cache_file = cls._CACHE_DIR / "query" / f"{safe}.json"
        if cache_file.exists():
            with cache_file.open() as f:
                return json.load(f)
        index = cls._build_index()
        result = cls._run_query(q, index)
        if result:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with cache_file.open("w") as f:
                json.dump(result, f)
        return result
