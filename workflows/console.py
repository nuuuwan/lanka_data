"""Interactive console for querying lanka_data.

Usage:
    python workflows/console.py
"""

from console.Console import Console


def run() -> None:
    Console().run()


if __name__ == "__main__":
    run()
