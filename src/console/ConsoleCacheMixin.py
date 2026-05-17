import shutil

from lanka_data.data_repos.gig2 import GIG2


class ConsoleCacheMixin:

    def _clear_cache(self) -> None:
        cache_dir = GIG2._CACHE_DIR
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            self.console.print(f"[green]Cache cleared:[/green] {cache_dir}")
        else:
            self.console.print("[yellow]No cache found.[/yellow]")
