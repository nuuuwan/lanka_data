# The Design of the Lanka Data API

*Rationale for the command grammar.*

## 0. The Problem We Wanted to Solve

The goal is "one API to rule them all": a single interface that can express *any* query, rather than a proliferation of endpoints, methods, libraries, and parameter sets that each answer one narrow question.

Most data libraries grow by accretion. Every new question tends to add another function, another endpoint, or another flag. The surface area grows without bound, and no single mental model survives contact with it. The user must learn the library question by question.

We wanted the opposite: a fixed, minimal grammar that a user learns *once* and can then aim at anything, and that a non-technical user can read and write without learning to program. Rather than adding a new entry point for each new question, the same four-field command string is pointed at new values. The set of answerable questions grows, but the interface does not.

For now, the domain is public Sri Lankan data: census measurements, election results, administrative geography, and their historical variants. But nothing in the grammar is specific to Sri Lanka, to censuses, or to elections. The four fields, *what* is measured, *when*, *where*, and *how* it is presented, are the dimensions of essentially any factual query about the world. The domain is the current scope; the grammar is intended to generalise beyond it.

The rest of this document specifies that grammar and argues that four fields are enough to reach every corner of the query space by composition.

## 1. Overview

The public interface is a single string parsed into four positional fields, delimited by slashes:

```
What / When / Where / How
```

There is no secondary configuration surface: no options object, no builder API, no config files. Every query is expressible as this one string, and the same string works unchanged as a Python argument, a command-line argument, a URL path, and a file path (§4).

## 2. The Four Fields

Each field is an independent axis of the query: the value chosen for one field does not constrain the valid values of another, with the single intentional coupling noted in §2.4.

### 2.1 What — the measurement

**What** identifies the quantity being retrieved. It is a measurement, independent of time, region, and presentation. Measurements include census quantities and election results. A reserved keyword denotes the *absence* of a measurement: a request for region geometry with no data bound to it, used for base maps and for inspecting boundary changes in isolation.

**What** encodes only the measurement type. It does not encode a year, a region, or an output format. This constraint is what allows a single measurement to be reused across every combination of the other three fields without expanding the vocabulary.

### 2.2 When — the observation time

**When** binds the measurement to the point or interval in time at which it happened. It accepts either a single year or an interval. An interval is a single value, not two concatenated queries.

### 2.3 Where — the region

**Where** identifies the region under measurement: its identity within the administrative hierarchy, not merely a coordinate. All current identities are places, but the field is structured so that a non-place identity would occupy the same position without a grammar change. It carries the most syntax of the four fields because it is the axis along which real queries vary most. The supported forms are:

- **Single region.** A region identifier naming one unit.
- **Resolution into child regions.** An operator that resolves a region into its constituent units of a given type, and that composes down every level of the hierarchy with no additional syntax.
- **Explicit set.** A list that selects exactly the named regions.
- **Contiguous range.** An operator that expands to all regions between two endpoints.
- **Historical boundary variant.** A suffix that selects a region's boundaries as they existed before a given boundary redesign.

The last form supports a deliberate decomposition: observation time (**When**) and boundary epoch (part of **Where**) are kept as separate values. In a jurisdiction with mutable boundaries these are independent facts, and collapsing them would misattribute counts to the wrong geometry.

### 2.4 How — the presentation

**How** specifies the output representation. It is distinct from **What**: **What** is the measured quantity; **How** is the rendering of that quantity. The same measurement can be emitted as a map, a chart, or raw data without any change to **What**.

Because presentation is separated from measurement, **How** carries its own modifier grammar. Modifiers refine the view, not the data: rankings of a category, shares, differences between observations, and diversity indices are all transformations applied to the same underlying values. Keeping **What** independent of **How** means the modifier family can grow, and every existing measurement acquires the new modifiers without changes to the data vocabulary. This is the coupling noted earlier: change-based modifiers are valid only when **When** supplies an interval.

## 3. Orthogonality and Generativity

The four fields are independent axes, so the set of valid queries is their Cartesian product rather than an explicit list. Changing the presentation leaves the region semantics unchanged; widening the observation time from a point to an interval leaves the measurement unchanged while enabling the difference-based modifiers.

Orthogonality is what allows a small vocabulary to generate a large query space. Any documented commands are samples from that space, not an exhaustive specification.

## 4. One Grammar, Everywhere

The command string is deliberately free of characters that need quoting or escaping in any common context. As a result, the *same string* is the entire interface in every place the data is consumed: as a Python library argument, as a command-line argument, as an HTTP endpoint path where the four fields *are* the URL path, and as a static file location where the four fields *are* the directory structure.

This is not four interfaces that happen to look alike; it is one grammar hosted in four contexts. A query learned in the browser transfers verbatim to a script, a shell, or a cache path, and a pre-rendered result can be served statically at exactly the address that would compute it dynamically.

Two further properties keep the mental model intact across contexts:

- **Readable by non-technical users.** The string parses in reading order and mirrors how a question is asked in plain language: *what*, *when*, *where*, *how*. No programming knowledge is needed to read an existing command or to modify one field of it to ask a neighbouring question.
- **Provenance.** Every response includes the sources that produced it and a query time, so a compact request still yields a fully traceable result.

## 5. Coverage

A four-field grammar is only useful if it spans the required query space. Distinct classes of query, a single machine-readable fact, a mapped snapshot, a comparison over time, a ranked view, a local decomposition, an explicit comparison set, geometry with no data bound, historical boundaries, and derived indices, are all expressed by selecting values along the same four axes. Coverage is obtained by composition, not by adding endpoints.

## 6. Summary

The interface reduces to four independent fields:

```
What / When / Where / How
```

The field set is **minimal** (each field is a distinct axis of variation, none redundant), **orthogonal** (fields compose without mutual constraint, apart from the interval requirement of change modifiers), **portable** (the identical string serves as Python argument, CLI argument, URL path, and file path), **intuitive** (it reads left-to-right as a plain-language question, usable without programming), and **complete** for the target domain (the archetypal query classes are all reachable by composition). Measurement, observation time, boundary definition, and presentation are kept as separate values so that none silently contaminates another.
