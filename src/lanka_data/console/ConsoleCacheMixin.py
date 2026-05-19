import pathlib
import re
import shutil

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)

from lanka_data.data_repos.census2024 import Census2024
from lanka_data.data_repos.gig2 import GIG2
from lanka_data.data_repos.RegionNames import _URL_PATTERNS, RegionNames
from lanka_data.renderers.Map import Map

# Temporary cache: generated outputs that can be re-created on demand.
# Includes query results (depend on WHERE logic) and rendered images.
_TEMP_DIRS = [
    pathlib.Path("/tmp/lanka_data/query"),
    pathlib.Path("/tmp/lanka_data/images"),
]

# Permanent cache: static reference data downloaded from remote sources.
_PERM_DIRS = [
    pathlib.Path("/tmp/lanka_data/geo"),
    pathlib.Path("/tmp/lanka_data/ents"),
    pathlib.Path("/tmp/lanka_data/data_repo"),
    pathlib.Path("/tmp/lanka_data/census2024"),
]

# Region codes to pre-fetch during `pre-load`:
#   9 provinces (LK-1..LK-9) + 25 admin districts + 22 electoral districts
_PRELOAD_CODES = (
    [f"LK-{i}" for i in range(1, 10)]
    + [
        "LK-11",
        "LK-12",
        "LK-13",
        "LK-21",
        "LK-22",
        "LK-23",
        "LK-31",
        "LK-32",
        "LK-33",
        "LK-41",
        "LK-42",
        "LK-43",
        "LK-44",
        "LK-45",
        "LK-51",
        "LK-52",
        "LK-53",
        "LK-61",
        "LK-62",
        "LK-71",
        "LK-72",
        "LK-81",
        "LK-82",
        "LK-91",
        "LK-92",
    ]
    + [f"EC-{i:02d}" for i in range(1, 23)]
)


class ConsoleCacheMixin:

    def _clear_cache(self) -> None:
        """Clear only temporary cache (generated SVG images)."""
        cleared = []
        for d in _TEMP_DIRS:
            if d.exists():
                shutil.rmtree(d)
                cleared.append(str(d))
        if cleared:
            for d in cleared:
                self.console.print(f"[green]Temp cache cleared:[/green] {d}")
        else:
            self.console.print("[yellow]No temp cache found.[/yellow]")

    def _pre_load(self) -> None:
        """Pre-download all permanent (static) reference data."""
        # 1. GIG2 data index
        self.console.print("[dim]  Loading GIG2 data index...[/dim]", end="")
        index = GIG2._build_index()
        self.console.print("[dim] done.[/dim]")

        # 2. All GIG2 TSV files
        all_urls = [
            entry["url"] for entries in index.values() for entry in entries
        ]
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=self.console,
            transient=True,
        ) as prog:
            n = len(all_urls)
            task = prog.add_task(
                f"[dim]  Loading GIG2 TSV files ({n} files)...[/dim]",
                total=n,
            )
            for url in all_urls:
                GIG2._fetch_tsv(url)
                prog.advance(task)
        self.console.print(f"[dim]  GIG2 TSV files loaded ({n}).[/dim]")

        # 3. All Census2024 TSV files
        census_labels = list(Census2024._DATASETS.keys())
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=self.console,
            transient=True,
        ) as prog:
            nc = len(census_labels)
            task = prog.add_task(
                f"[dim]  Loading Census 2024 files ({nc} datasets)...[/dim]",
                total=nc,
            )
            for label in census_labels:
                Census2024._load_tsv_text(label)
                prog.advance(task)
        self.console.print(f"[dim]  Census 2024 files loaded ({nc}).[/dim]")

        # 4. Region names TSVs (one per URL pattern)
        self.console.print(
            "[dim]  Loading region names "
            f"({len(_URL_PATTERNS)} files)...[/dim]",
            end="",
        )
        for _, url in _URL_PATTERNS:
            RegionNames._fetch_and_cache(url)
        self.console.print("[dim] done.[/dim]")

        # 5. Geo rings for all known regions:
        #    static list (provinces/districts/EDs) + every PD and DSD
        #    that RegionNames just loaded from its TSVs.
        pd_codes = sorted(
            c
            for c in RegionNames._cache
            if re.fullmatch(r"EC-\d{2}[A-Za-z]", c)
        )
        dsd_codes = sorted(
            c for c in RegionNames._cache if re.fullmatch(r"LK-\d{4}", c)
        )
        all_codes = _PRELOAD_CODES + pd_codes + dsd_codes

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=self.console,
            transient=True,
        ) as prog:
            n = len(all_codes)
            task = prog.add_task(
                f"[dim]  Loading geo data ({n} regions)...[/dim]",
                total=n,
            )
            for code in all_codes:
                Map._fetch_geo_rings(code)
                prog.advance(task)

        self.console.print(
            f"[green]Pre-load complete.[/green] "
            f"[dim]({len(all_codes)} regions cached)[/dim]"
        )
