# The Visual Style of Lanka Data

*A style guide for producing highly professional, publication-grade infographics.*

## 0. Purpose

Every Lanka Data command ends in a picture. The `How` field selects a visual
(`Map`, `HexMap`, `BarChart`, `StackedBarChart`, `PieChart`, `BumpChart`,
`TreeMap`, `Histogram`, `ScatterPlot`), and the plotting layer
in [`src/lanka_data/visual/`](src/lanka_data/visual/) renders it. This document
defines the visual language those renderers must speak: the canvas, the type,
the colour, and the composition rules that make an output look considered rather
than accidental.

The goal is a house style. Two different commands — a religion cartogram and a
parliamentary bump chart — should look like siblings: same font, same margins,
same header bar, same treatment of legends and sources. A reader should
recognise a Lanka Data infographic before reading a single word of it.

This guide is normative. Where it states a concrete value (a hex code, a font
size, a margin), that value is the contract; renderers should read it from a
shared constant rather than re-inventing it locally. Where it states a
principle, new visuals are expected to honour the principle even in cases this
document did not foresee.

---

## 1. Ten Principles

Everything below elaborates these ten rules. They are the summary; the rest of
the document is the specification.

1. **One accessible palette.** Use a single, consistent, colourblind-safe
   palette with real contrast between adjacent hues — never two near-identical
   colours side by side.
2. **Typographic hierarchy by weight, not just size.** Distinguish
   title / subtitle / body with deliberate steps in size *and* weight, so the
   hierarchy survives at a glance and in greyscale.
3. **Ordered, evenly formatted legends.** Legend entries share one number
   format (padding, precision, alignment) and are ordered by value, never by
   accident of iteration.
4. **Data over chrome.** Keep borders, gridlines, and separators light so the
   shapes and quantities dominate; chrome recedes.
5. **Consistent canvas margins.** Reserve uniform padding around the whole
   figure; nothing touches the edge.
6. **Muted, neutral surfaces.** Backgrounds and banners are quiet neutrals, not
   saturated fills competing with the data.
7. **One de-emphasised style for metadata.** Footnotes, source lines, and
   captions share a single smaller, lighter type style.
8. **Consistent depth cues.** Pick one convention for shadows and elevation —
   subtle, or none — and apply it to every element identically.
9. **Automatic label contrast.** Choose dark text on light fills and light text
   on dark fills automatically, so every label stays legible.
10. **A fixed home for scale and source notes.** Anchor legends, scale notes,
    and source text to consistent, predictable positions rather than floating
    them.

---

## 2. Canvas & Output

A fixed canvas is the foundation of a house style; every other measurement in
this guide is expressed relative to it.

- **Aspect & size.** The figure aspect ratio adapts to the number of datasets:
  **1×1** for one, **2×1** for two, **3×1** for three, and **2×2** for four
  (more than four is rejected). Each panel occupies a **9 × 9 inch** square, so
  the canvas is a grid of these squares.
- **Resolution.** Export at **200 DPI**. This is the minimum for crisp text at
  presentation scale; never downscale the DPI to save bytes.
- **Format.** Output is **PNG** — lossless, universally embeddable, and free of
  the compression artefacts that JPEG would smear across flat colour fields.
- **Deterministic layout.** The figure geometry must not depend on the machine.
  The same command produces the same composition every time, so outputs are
  directly comparable across runs and across time.

*(Reference: `PlotLayout.figsize`, `PlotLayout.COUNT_TO_GRID`,
`savefig(dpi=200)`.)*

### 2.1 The three horizontal bands

The canvas is divided top-to-bottom into three fixed bands. This is Principle 5
(consistent margins) and Principle 10 (a fixed home for metadata) made concrete.

| Band       | Vertical extent      | Role                                            |
| ---------- | -------------------- | ----------------------------------------------- |
| **Header** | top band, grows with title lines | Title (what / where / how, as English text, wrapped over as many lines as needed) |
| **Body**   | middle (0.10–0.86)   | The plot(s) — one sub-figure per dataset year   |
| **Footer** | bottom **5 %** (0–0.05) | Source attribution (left) and the GitHub repository link (right) |
| **Spine**  | left margin, full body height | The **Lanka Data** brand mark, set as quiet rotated (vertical) type |

The body is laid out on a grid: one column per dataset, with a fixed inter-panel
gutter, so a two-year comparison reads as two aligned panels rather than two
unrelated charts. A uniform **5 %** margin is reserved on every side, so the
plotted graphics never touch the canvas edge (`Style.MARGIN`). The header band is
not a fixed height: it expands downward to fit however many lines the title
needs, so the title text is never clipped or shrunk to fit one line.

---

## 3. Typography

Type is the most visible signal of quality. One family, a small set of sizes,
and contrast carried by weight and colour — never by a zoo of typefaces.

- **One family, bundled.** The single type family is **Fira Sans**, shipped as a
  TTF inside the repo and installed into the renderer at draw time. Bundling the
  font guarantees identical output on every machine; the guide never falls back
  to a system default.
- **Deliberate size steps.** Sizes are chosen, not arbitrary. The working scale
  is:
  - **Title / header** — the largest step, for the one-line description of the
    query.
  - **Panel labels** (e.g. the year over each sub-figure) — the mid step.
  - **Legend & metadata** — the smallest step, de-emphasised.
- **Hierarchy by weight and colour, not size alone (Principle 2).** A title is
  not merely bigger; it is heavier and darker than the subtitle. This keeps the
  hierarchy readable at thumbnail size and in greyscale, where size differences
  alone collapse.
- **Centred, multi-line headers.** Header and panel text is centre-aligned. The
  header composes its parts with a single middle-dot separator (` · `) so *what*,
  *where*, and *how* read as one balanced phrase; when that phrase is wider than
  the canvas it wraps onto additional lines at full title size rather than
  shrinking, so nothing is ever clipped.
- **Persistent brand and source line.** Every figure carries the **Lanka Data**
  brand mark as a quiet rotated masthead down the left spine and a link to the
  project's GitHub repository in the footer, so an output is self-identifying
  wherever it travels. The brand mark stays out of the header so it never
  competes with the title.
- **In-figure labels earn their place.** Region and data labels are only drawn
  when they fit their shape; a label that cannot be sized legibly is dropped
  rather than overlapped or shrunk into illegibility (see §6).

---

## 4. Colour

Colour is where an infographic most easily looks amateur. Lanka Data therefore
treats colour as a controlled vocabulary, not a free choice per chart.

### 4.1 The semantic palette (Principle 1)

Many categories in Sri Lankan data carry an expected colour — political parties,
ethnicities, and religions have conventional associations, and the national flag
supplies a recognisable set of hues. Lanka Data maintains a **fixed
value → colour map** so that, for example, a given party is the *same* colour in
every chart it ever appears in.

- **Stable meaning.** A category's colour is a property of the category, not of
  the chart. Never recolour a known category to fit a local palette.
- **Reserved neutrals for absence.** Missing, insufficient, and "no change"
  states use a reserved family of greys, kept visually distinct from every
  meaningful category so "no data" never reads as a value.
- **Contrast between neighbours.** Adjacent categories must be
  distinguishable — no two near-identical hues side by side — and the palette is
  chosen to remain separable for common forms of colour-blindness.

### 4.2 Continuous & diverging scales

When the quantity is numeric rather than categorical, colour comes from a
colormap chosen by the *kind* of quantity:

- **Sequential** (magnitudes, e.g. a share of population) → a single-hue,
  light-to-dark ramp. Darker means more; the direction is never ambiguous.
- **Diverging** (signed change around a meaningful zero, e.g. a swing) → a
  two-ended ramp with a neutral midpoint, so the sign of the change is legible
  from colour alone.
- **Categorical fallback** (unlabelled categories with no semantic colour) → an
  evenly spaced qualitative set.

The choice of scale is driven by the `How` modifier (rank, percentage, `Change`,
`Diversity`, `Segregation`), not hand-picked per chart, so the same kind of
question always gets the same kind of scale.

### 4.3 Surfaces (Principle 6)

Backgrounds and banners are quiet neutrals, chosen to sit *behind* the data:
a light header bar, a plain footer, and a neutral plot background. Saturated
fills are reserved for the data itself. The surface should never be the first
thing the eye lands on.

---

## 5. Composition & Chrome

Professional layout is mostly restraint: strong alignment, generous whitespace,
and chrome that stays out of the way (Principle 4).

- **Data over chrome.** Borders between shapes (e.g. the outlines between
  hexagons or regions) are thin and low-contrast, so the shapes read as a field
  of colour rather than a lattice of lines. Gridlines, where present at all, are
  fainter than any data mark.
- **Alignment is the grid.** Panels, titles, legends, and source lines align to
  the same axes. A comparison of two years places the panels on a shared
  baseline with a fixed gutter; the eye should never have to re-calibrate
  between panels.
- **Whitespace is intentional.** The margins of §2.1 are not slack to be filled;
  they are part of the design. Resist the urge to push content into the header,
  footer, or edge padding.
- **One depth convention (Principle 8).** Either every element carries the same
  subtle shadow/elevation, or none does. Mixed depth cues — a shadow on one mark
  and none on the next — read as a mistake. The default is flat.

---

## 6. Legends, Labels & Scale Notes

These are the elements most often left to chance; here they are specified.

### 6.1 Legends (Principles 3 & 10)

- **Ordered by value.** Entries are ordered by the quantity they represent, not
  by insertion order or dictionary order.
- **One number format.** Every entry shares a single format — same precision,
  same padding, same alignment — so the column of labels reads as a table, not a
  ragged list. Percentages are shown as percentages, consistently.
- **Bounded length.** A legend is capped at a small number of entries (about
  ten). When a scale has more, it is sampled evenly across its range — always
  keeping the extremes — so the key stays readable and still conveys the full
  span.
- **A fixed home.** The legend sits in a consistent, predictable location
  relative to the plot, not floating wherever there happened to be space.

### 6.2 In-shape labels (Principle 9)

Labels drawn *inside* a coloured shape (a region on a map, a slice, a bar) must
stay legible against their own fill:

- **Automatic contrast.** Text colour is chosen from the fill's lightness — dark
  text on light fills, light text on dark fills — automatically, for every mark,
  with no per-chart tuning.
- **Fit before drawing.** A label is sized and rotated to fit the shape's
  available box; if even the smallest legible size will not fit, the label is
  truncated or omitted rather than allowed to overflow or collide.

### 6.3 Scale & source notes (Principles 7 & 10)

- **Sources have one home and one style.** Attribution lives in the footer band,
  in the single de-emphasised metadata type style — smaller and lighter than the
  body — never restyled per chart.
- **Notes are anchored, not floated.** Any scale note or caption is anchored to a
  consistent corner or band, sharing the metadata style, so it reads as a
  footnote rather than as stray text competing with the title.

---

## 7. Consistency Across Visual Types

The same rules bind every `How` visual, so the family stays coherent:

- **`Map` / cartogram** — geographic or population-warped regions, filled from
  the palette or a colormap, with fitted in-region labels and light borders.
- **`HexMap`** — one hexagon per unit of population, laid out as a tessellation;
  the same palette, the same light borders, the same automatic label contrast.
- **`BarChart`** — magnitudes as bars, ordered meaningfully, sharing the legend
  and number-format rules.
- **`StackedBarChart`** — the same bars normalized to 100%, comparing categorical
  composition across regions on a shared share axis.
- **`PieChart`** — parts of a whole, from the same categorical palette, with the
  same legend treatment.
- **`BumpChart`** — rank over time, using the categorical palette to trace each
  series consistently across periods.
- **`TreeMap`** — nested rectangles sized by value, showing overall categorical
  composition from the same palette.
- **`Histogram`** — the distribution of region totals, binned and drawn with the
  shared number-format and grid rules.
- **`ScatterPlot`** — two measures per region as points, coloured by dominant
  category from the same palette.

Whatever the type, the header, footer, font, margins, palette, and metadata
style are identical. That identity — not any single chart — is the Lanka Data
visual style.

---

## 8. A Checklist Before Shipping a Visual

Before a new or changed visual is considered done, it should pass every line:

- [ ] Canvas is 16 × 9 at 200 DPI, exported as PNG.
- [ ] Header, body, and footer bands are respected; nothing touches the edge.
- [ ] Font is the bundled Fira Sans; sizes follow the title / label / metadata
      steps.
- [ ] Hierarchy is carried by weight and colour, not size alone, and survives in
      greyscale.
- [ ] Known categories use their semantic colours; absence uses the reserved
      neutrals.
- [ ] Numeric scales use the correct sequential / diverging / categorical ramp.
- [ ] Borders and gridlines are lighter than the data marks.
- [ ] Legend is ordered by value, capped in length, and uses one number format.
- [ ] In-shape labels have automatic contrast and are fitted or dropped, never
      overlapped.
- [ ] Sources sit in the footer in the single metadata style; notes are anchored,
      not floated.
- [ ] Depth cues are uniform (the default is flat).
