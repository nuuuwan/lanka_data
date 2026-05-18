import pathlib
import shutil

from lanka_data.data_repos.gig2 import GIG2

_IMAGES_DIR = pathlib.Path("/tmp/lanka_data/images")
_GEO_CACHE_DIR = pathlib.Path("/tmp/lanka_data/geo")


class ConsoleCacheMixin:

    def _clear_cache(self) -> None:
        cleared = []
        for d in [GIG2._CACHE_DIR, _IMAGES_DIR, _GEO_CACHE_DIR]:
            if d.exists():
                shutil.rmtree(d)
                cleared.append(str(d))
        if cleared:
            for d in cleared:
                self.console.print(f"[green]Cache cleared:[/green] {d}")
        else:
            self.console.print("[yellow]No cache found.[/yellow]")
