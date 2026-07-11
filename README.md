# Lanka Data

Public data about Sri Lanka 🇱🇰.

## Overview

Lanka Data provides *one API to rule them all*: a single interface that can express any query against public Sri Lankan data — census measurements, election results, and administrative geography — without proliferating endpoints or methods.

Every query is a slash-delimited string of four fields:

```
What / When / Where / How
```

| Field | Meaning | Example |
| --- | --- | --- |
| **What** | The measurement | `Religion` |
| **When** | Time of observation | `2012-2024` |
| **Where** | Region scope | `LK:district` |
| **How** | Output format | `Map:1st` |

The same string works as a Python argument, a CLI argument, a URL path, and a file path.

## Installation

```bash
pip install lanka-data
```

## Quick Start

```python
from lanka_data.datasets.command.CommandRunner import CommandRunner

CommandRunner.run("Presidential/2024/LK:ed/Map:NPP")
```

Or from the command line:

```bash
python -m workflows.single Presidential/2024/LK:ed/Map:NPP
```

## Documentation

- [Help](README.help.md) — full API reference for all fields
- [Examples](README.examples.md) — rendered output for every example
- [Datasets](README.datasets.md) — available data sources
- [Philosophy](README.philosophy.md) — design rationale for the query grammar
- [Citation Paper](latex/lanka_data.pdf)
