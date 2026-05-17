"""Interactive console for querying lanka_data.

Usage:
    python workflows/console.py
"""

from lanka_data import Console


def run() -> None:
    Console().run()


if __name__ == "__main__":
    run()
