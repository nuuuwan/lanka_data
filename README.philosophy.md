# The Design of the Lanka Data API

*Rationale for the command grammar.*

## The Problem We Wanted to Solve

The goal is "one API to rule them all": a single interface that can express *any* query, rather than a proliferation of endpoints, methods, libraries, and parameter sets that each answer one narrow question.

Most data libraries grow by accretion. Every new question (e.g. a different measurement, a new time range, a finer region, another chart type) tends to add another function, another endpoint, or another flag. The surface area grows without bound, and no single mental model survives contact with it. The user must learn the library question by question.

We wanted the opposite: a fixed, minimal grammar that a user learns *once* and can then aim at anything — and that a non-technical user can read and write without learning to program. Rather than adding a new entry point for each new question, the same four-field command string is pointed at new values. The set of answerable questions grows, but the interface does not.

For now, "anything" means any query about public Sri Lankan data ("Lanka Data"): census measurements, election results, administrative geography, and their historical variants. But nothing in the grammar is specific to Sri Lanka, to censuses, or to elections. The four fields — *what* is measured, *when*, *where*, and *how* it is presented — are the dimensions of essentially any factual query about the world. The domain is the current scope; the grammar is intended to generalise beyond it.

The rest of this document specifies that grammar and argues that four fields are enough to reach every corner of the query space by composition.

## 1. Overview

The public interface to Lanka Data is a single string parsed into four positional fields, delimited by slashes:

```bash
What / When / Where / How
```

For example:

```bash
Religion/2012-2024/LK:district/Map:Change
```

There is no secondary configuration surface: no options object, no builder API, no config files. Every query is expressible as this one string, and the same string works unchanged as a Python argument, a command-line argument, a URL path, and a file path (§4).

The remainder uses commands drawn directly from [README.md](README.md).

## 2. The Four Fields

Each field is an independent axis of the query: the value chosen for one field does not constrain the valid values of another (with the single, intentional coupling noted in §2.4).

### 2.1 What — the measurement

**What** identifies the quantity being retrieved. It is a measurement, independent of time, region, and presentation.

```bash
Religion/...
Ethnicity/...
Roof/...
Parliamentary/...
Presidential/...
Empty/...
```

`Religion` and `Ethnicity` are census measurements. `Parliamentary`, `Presidential`, and `Local` are election results. `Empty` is a special keyword that denotes the absence of a measurement: a request for region geometry with no data bound to it, used for base maps and for inspecting boundary changes in isolation.

**What** encodes only the measurement type. It does not encode a year, a region, or an output format. This constraint is what allows a single measurement to be reused across every combination of the other three fields without expanding the vocabulary.

### 2.2 When — the observation time

**When** binds the measurement to a point or interval in time when it happened.

A single year:

```bash
Parliamentary/2024/...
Presidential/2015/...
Local/2025/...
Religion/2012/...
```

An interval:

```bash
Religion/2012-2024/...
```

`2012-2024` is a single interval value, not two concatenated queries.

### 2.3 Where — the region

**Where** identifies the region under measurement; its identity within the administrative hierarchy, not merely a coordinate. All current identities are places; the field is structured so that a non-place identity (e.g. a person or a "Who") would occupy the same position without a grammar change. It carries the most syntax of the four fields because it is the axis along which real queries vary most.

The following forms are supported.

**Single region.** A region identifier:

```bash
Parliamentary/2024/LK/JSON
```

**Resolution into child regions.** The colon operator resolves a region into its constituent units of a given type. `LK:province` is Sri Lanka as its nine provinces; `LK:district` as its twenty-five districts:

```bash
Empty/2024/LK:province/Map
Religion/2012-2024/LK:district/Map:1st
```

**Resolution of a named region at a finer level.** The same colon operator composes down the hierarchy — province, district, DSD, PD, LG — with no additional syntax:

```bash
Religion/2012-2024/LK-42:district/BarChart
Religion/2012-2024/LK-43:dsd/BarChart
Religion/2012-2024/LK-11:lg/BarChart
Presidential/2015/LK-11:pd/Map
```

**Explicit set.** A comma-separated list selects exactly the named regions:

```bash
Empty/2024/LK-1,LK-2,LK-3,LK-9,LK-8/Map
Religion/2012-2024/LK-33,LK-82,LK-32:district/BarChart
```

**Contiguous range.** The ellipsis operator expands to all regions between two endpoints:

```bash
Empty/2024/LK-5...LK-8/Map
```

**Explicit zoom.** The `@` operator assigns a region an explicit scale, for framing regions that automatic bounds would render too small or too large:

```bash
Empty/2024/LK-1127025@20/Map
```

**Historical boundary variant.** A `pre<year>` suffix selects the region's boundaries as they existed before a given boundary redesign:

```bash
Empty/2012/LK-pre1845:province/Map
Empty/2012/LK-pre1873:province/Map
Empty/2012/LK-pre1886:province/Map
Empty/2012/LK-pre1889:province/Map
Empty/2012/LK-pre1959:district/Map
Empty/2012/LK-pre1961:district/Map
Religion/2012-2024/LK-31-pre2019:dsd/BarChart
```

The last example demonstrates a deliberate decomposition. The measurement interval is 2012–2024 (**When**), while the boundary set is the pre-2019 definition (part of **Where**). Observation time and boundary epoch are kept as separate values because in a jurisdiction with mutable boundaries they are independent facts; collapsing them would misattribute counts to the wrong geometry.

### 2.4 How — the presentation

**How** specifies the output representation. It is distinct from **What**: **What** is the measured quantity; **How** is the rendering of that quantity. The same measurement can be emitted as a choropleth map, a bar chart, or raw JSON without any change to **What**:

```bash
Religion/2012-2024/LK:district/Map:2nd
Religion/2012-2024/LK-42:district/BarChart
Parliamentary/2024/LK/JSON
```

Because presentation is separated from measurement, **How** carries its own modifier grammar after a colon. Modifiers refine the view, not the data:

```bash
Map:1st            largest category per region
Map:2nd            second largest category
Map:3rd            third largest category
Map:2ndPct         second largest, shaded by share
Map:3rdPct         third largest, shaded by share
Map:Change         difference between the two observations
Map:DiversityPew   diversity index across categories
```

Each modifier is a transformation applied to the same underlying values; `Map:1st` and `Map:3rd` operate on identical data and differ only in the projection computed from it. Keeping **What** independent of **How** means the modifier family can grow — new rankings, differences, or indices — and every existing measurement acquires the new modifiers without changes to the data vocabulary. This is the coupling noted earlier: change-based modifiers such as `Map:Change` are only valid when **When** supplies an interval.

## 3. Orthogonality and Generativity

The four fields are independent axes, so the set of valid queries is their Cartesian product rather than an explicit list. Changing the **How** field from `Map` to `BarChart` leaves the **Where** semantics unchanged; changing **When** from `2012` to `2012-2024` leaves **What** unchanged while enabling the difference-based modifiers.

Orthogonality is what allows a small vocabulary to generate a large query space. The examples in [README.md](README.md) are samples from that space, not an exhaustive specification.

## 4. One Grammar, Everywhere

The command string is deliberately free of characters that need quoting or escaping in any common context. As a result, the *same string* is the entire interface in every place the data is consumed:

- **Python library:**

  ```python
  output = CommandRunner.run('Energy/2012/LK/Map')
  ```

- **Command line:**

  ```bash
  python workflows/single.py Energy/2012/LK/Map
  ```

- **HTTP endpoint** — the four fields *are* the URL path:

  ```text
  https://lanka-data.api/Energy/2012/LK/Map
  ```

- **Static file location** — the four fields *are* the directory structure:

  ```text
  /tmp/lanka_data/Energy/2012/LK/Map
  ```

This is not four interfaces that happen to look alike; it is one grammar hosted in four contexts. A query learned in the browser transfers verbatim to a script, a shell, or a cache path, and a pre-rendered result can be served statically at exactly the address that would compute it dynamically.

Two further properties keep the mental model intact across contexts:

- **Readable by non-technical users.** The string parses in reading order and mirrors how a question is asked in plain language — *what*, *when*, *where*, *how*. `Religion/2012-2024/LK:district/Map:Change` is self-describing: no programming knowledge is needed to read an existing command or to modify one field of it to ask a neighbouring question.

- **Provenance.** Every response includes the `sources` that produced it and a `query_time_ms`, so a compact request still yields a fully traceable result.

## 5. Coverage

A four-field grammar is only useful if it spans the required query space. The following archetypes, each a distinct class of query, are all expressed by selecting values along the same four axes:

- **Single fact, machine-readable.** 2024 parliamentary result for the country as data:
  `Parliamentary/2024/LK/JSON`

- **Snapshot, mapped.** 2015 presidential vote across one district's polling divisions:
  `Presidential/2015/LK-11:pd/Map`

- **Comparison over time.** Religion shift across districts, 2012–2024:
  `Religion/2012-2024/LK:district/Map:Change`

- **Ranked view.** Second-largest religion per district, shaded by share:
  `Religion/2012-2024/LK:district/Map:2ndPct`

- **Local decomposition.** One district broken into its DSDs as a bar chart:
  `Religion/2012-2024/LK-43:dsd/BarChart`

- **Explicit comparison set.** Three named districts side by side:
  `Religion/2012-2024/LK-33,LK-82,LK-32:district/BarChart`

- **Geometry only.** Provinces with no data bound:
  `Empty/2024/LK:province/Map`

- **Historical boundaries.** Country borders before successive reforms:
  `Empty/2012/LK-pre1845:province/Map` … `Empty/2012/LK-pre1961:district/Map`

- **Diversity index.** Religious diversity per district:
  `Religion/2012-2024/LK:district/Map:DiversityPew`

Each is a different query answered by different values on the same four axes. Coverage is obtained by composition, not by adding endpoints.

## 6. Summary

The interface reduces to four independent fields:

```bash
What / When / Where / How
```

The field set is **minimal** (each field is a distinct axis of variation, none redundant), **orthogonal** (fields compose without mutual constraint, apart from the interval requirement of change modifiers), **portable** (the identical string serves as Python argument, CLI argument, URL path, and file path), **intuitive** (it reads left-to-right as a plain-language question, usable without programming), and **complete** for the target domain (the archetypal query classes are all reachable by composition). Measurement, observation time, boundary definition, and presentation are kept as separate values so that none silently contaminates another.
