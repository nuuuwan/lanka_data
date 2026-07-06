# Analyses using Lanka Data

A curated list of analyses that can be produced using the
[Lanka Data](https://github.com/nuuuwan/lanka_data) library.

Each entry shows the `<What>/<When>/<Where>/<How>` command and a
brief description of what it reveals.

---

## 1. Demographics & Population

| Command | Description |
|---------|-------------|
| `Population/2001/LK:district/Map` | Choropleth of total population by district (2001 census). |
| `AgeGroup/2024/LK:district/Map:1st` | Most-common age group per district (2024). |
| `AgeGroup/2001-2024/LK:district/LineChart` | Age-group share trends over three censuses at district level. |
| `Gender/2024/LK:district/Map` | Male/female balance across every district (2024). |
| `Gender/2012-2024/LK:district/Map:Change` | Shift in gender ratio between the two most-recent censuses. |
| `Dependency/2001/LK:district/Map` | Dependency ratio (young + elderly vs working-age) by district. |
| `Fertility/2001/LK:district/Map` | Fertility-related indicators by district (2001). |
| `Growth/2001/LK:district/Map` | Population-growth rate per district. |
| `Migration/2001/LK:district/Map` | In-/out-migration patterns by district. |
| `AgeGroup/2024/LK:dsd/HexMap` | Population-proportional hex map coloured by dominant age group. |
| `AgeGroup/2024/LK:district/BubbleMap` | Bubble-map where bubble size = population and colour = top age group. |
| `AgeGroup/2024/LK:district/StackedBarChart` | 100 % stacked bars showing age-group composition per district. |
| `AgeGroup/2024/LK:district/TreeMap` | Tree-map of population broken down by age group. |
| `AgeGroup/2024/LK/Histogram` | Distribution of age-group shares across all regions. |

---

## 2. Religion

| Command | Description |
|---------|-------------|
| `Religion/2024/LK:district/Map:1st` | Most-common religion per district. |
| `Religion/2012-2024/LK:district/Map:Change` | Net change in religion composition by district. |
| `Religion/2024/LK:district/Map:Diversity` | Religious diversity index by district (2024). |
| `Religion/2024/LK:district/Map:DiversityPew` | Pew Research Religious Diversity Index by district. |
| `Religion/2024/LK:district/Map:Segregation` | Religious segregation score per district. |
| `Religion/2001-2012-2024/LK:district/LineChart` | Three-census trend of religion shares at district level. |
| `Religion/2024/LK:dsd/Map:1st` | Most-common religion at Divisional Secretariat level. |
| `Religion/2024/LK:district/BarChart` | Side-by-side bar chart of religion composition across districts. |
| `Religion/2024/LK:district/BubbleMap` | Population-proportional bubbles coloured by majority religion. |
| `Religion/2024/LK:district/Map:2ndPct` | Share of the 2nd most-common religion per district. |
| `Religion/2024/LK:district/Map:3rdPct` | Share of the 3rd most-common religion per district. |
| `Religion/2024/LK:district/HexMapAnimation` | Animated hex map cycling through each religion's share. |

---

## 3. Ethnicity & Language

| Command | Description |
|---------|-------------|
| `Ethnicity/2024/LK:district/Map:1st` | Dominant ethnic group per district (2024). |
| `Ethnicity/2012-2024/LK:district/Map:Change` | Change in ethnic composition between the two censuses. |
| `Ethnicity/2024/LK:district/Map:Diversity` | Ethnic diversity index per district. |
| `Ethnicity/2024/LK:district/Map:Segregation` | Ethnic segregation score per district. |
| `Ethnicity/2001-2012-2024/LK:district/LineChart` | Long-run trend in ethnic-group shares. |
| `Ethnicity/2024/LK:dsd/Map:1st` | Dominant ethnic group at DSD level. |
| `Speaking/2001/LK:district/Map:1st` | Most-spoken language per district (2001). |
| `Ethnicity/2024/LK:district/StackedBarChart` | Full ethnic-composition stacked bars per district. |
| `Ethnicity/2024/LK:district/CartogramAnimation` | Animated cartogram cycling through each ethnic group's share. |
| `Ethnicity/2024/LK:district/ScatterPlot` | Scatter of ethnic shares — explore correlations between groups. |

---

## 4. Education & Literacy

| Command | Description |
|---------|-------------|
| `Literacy/2001/LK:district/Map` | Literacy rate by district (2001). |
| `Education/2012/LK:district/Map:Top` | Districts with the highest educational attainment (2012). |
| `Education/2001-2012/LK:district/Map:Change` | Change in education level between censuses. |
| `Attendance/2001/LK:district/Map` | School-attendance rate per district. |
| `Enrollment/2001/LK:district/Map` | School-enrollment rate per district. |
| `NotAttending/2001/LK:district/Map` | Share of children not attending school. |
| `Education/2012/LK:district/BarChart` | Education-level composition bar chart per district. |
| `Literacy/2001/LK:district/HexMap` | Population-proportional hex map coloured by literacy rate. |
| `Education/2012/LK:dsd/Map:1st` | Dominant education level at DSD granularity. |
| `Education/2012-2024/LK:district/LineChart` | Trend in educational attainment at district level. |

---

## 5. Economy & Employment

| Command | Description |
|---------|-------------|
| `Employment/2001/LK:district/Map` | Employment rate by district. |
| `Unemployment/2001/LK:district/Map` | Unemployment rate by district. |
| `Unemployment/2001/LK:district/Map:Top` | Districts with the highest unemployment. |
| `Laborforce/2001/LK:district/Map` | Labour-force participation rate per district. |
| `Inactive/2001/LK:district/Map` | Economically inactive population share. |
| `Industry/2001/LK:district/Map:1st` | Dominant industry per district. |
| `Sectoral/2001/LK:district/BarChart` | Sectoral (primary/secondary/tertiary) composition per district. |
| `Occupations/2001/LK:district/Map:1st` | Most common occupation per district. |
| `AgriOccupations/2001/LK:district/Map` | Share of the population in agricultural occupations. |
| `NonAgriEmployment/2001/LK:district/Map` | Non-agricultural employment rate by district. |
| `Informal/2001/LK:district/Map` | Informal-sector employment rate per district. |
| `Economy/2012/LK:district/BarChart` | Economic-activity composition per district (2012). |
| `Unemployment/2001/LK:district/ScatterPlot` | Scatter of unemployment vs. other socio-economic variables. |

---

## 6. Housing & Infrastructure

| Command | Description |
|---------|-------------|
| `Structure/2024/LK:district/Map:1st` | Dominant housing structure type per district (2024). |
| `Structure/2001-2024/LK:district/Map:Change` | Change in housing-structure type across three censuses. |
| `Walls/2024/LK:district/Map:1st` | Most common wall material per district. |
| `Floor/2024/LK:district/Map:1st` | Most common floor material per district. |
| `Roof/2024/LK:district/Map:1st` | Most common roof material per district. |
| `Water/2024/LK:district/Map:1st` | Primary water source per district. |
| `Fuel/2024/LK:district/Map:1st` | Primary cooking fuel per district. |
| `Lighting/2024/LK:district/Map:1st` | Primary lighting source per district. |
| `Toilet/2024/LK:district/Map:1st` | Most common toilet facility type per district. |
| `Water/2001-2024/LK:district/LineChart` | Trend in water-source access from 2001 to 2024. |
| `Toilet/2001-2024/LK:district/LineChart` | Trend in toilet-facility access from 2001 to 2024. |
| `ConstructionYear/2012/LK:district/Map` | Age of the housing stock per district (2012). |
| `Ownership/2012/LK:district/Map:1st` | Dominant housing-tenure type (owned/rented) per district. |
| `Rooms/2012/LK:district/Map` | Average number of rooms per dwelling. |
| `Occupancy/2012/LK:district/Map` | Occupancy density per district. |
| `Waste/2012/LK:district/Map:1st` | Primary waste-disposal method per district. |
| `Housing/2001/LK:district/StackedBarChart` | Full housing-type composition per district (2001). |
| `Structure/2024/LK:district/BubbleMapAnimation` | Animated population bubbles cycling through structure types. |

---

## 7. Elections & Politics

| Command | Description |
|---------|-------------|
| `Presidential/2024/LK:district/Map:1st` | Winning presidential candidate per district (2024). |
| `Presidential/2024/LK:ed/Map:1st` | Winning candidate per electoral district. |
| `Presidential/2001-2024/LK:district/LineChart` | Party vote-share trends across presidential elections. |
| `Presidential/2024/LK:district/Map:Change` | Swing from previous presidential election. |
| `PresidentialSummary/2024/LK:district/Map` | Top-line presidential-vote summary per district. |
| `Parliamentary/2024/LK:district/Map:1st` | Party with the most seats per district (2024). |
| `Parliamentary/2001-2024/LK:district/LineChart` | Parliamentary vote-share trends across elections. |
| `Parliamentary/2024/LK:pd/Map:1st` | Winning party per polling division. |
| `ParliamentarySummary/2024/LK:district/BarChart` | Parliamentary vote breakdown per district. |
| `Local/2024/LK:district/Map:1st` | Dominant party in local-government elections per district. |
| `LocalSummary/2001-2024/LK:district/LineChart` | Long-run trend in local-government vote share. |
| `Presidential/2024/LK:district/HexMap` | Population-weighted hex map coloured by presidential winner. |
| `Presidential/2024/LK:district/BubbleMap` | Bubble map where size = votes cast, colour = winner. |
| `Presidential/2024/LK:district/CartogramAnimation` | Animated cartogram cycling through each candidate's vote share. |
| `Parliamentary/2024/LK:district/ScatterPlot` | Scatter of party vote shares — explore correlations. |

---

## 8. Cross-cutting & Multi-variable

| Command | Description |
|---------|-------------|
| `Religion/2024/LK:district/CSV` | Export religion data as CSV for further analysis. |
| `Ethnicity/2024/LK:district/JSON` | Export ethnicity data as JSON for programmatic use. |
| `Religion/2024/LK:district/Table` | Formatted table of religion shares per district. |
| `AgeGroup/2024/LK:district/Histogram` | Distribution of age-group shares across districts. |
| `Employment/2001/LK:district/ScatterPlot` | Scatter of employment against other variables (overlay). |
| `Religion/2024-MapAnimation/LK:district` | Animated map cycling through religion groups over the country. |
| `Ethnicity/2024/LK:district/MapAnimation` | Animated map cycling through ethnic groups. |
| `Religion/2024/LK:gnd/Map:1st` | Finest-grain (GND) map of dominant religion. |
| `Ethnicity/2024/LK:gnd/Map:1st` | Finest-grain (GND) map of dominant ethnic group. |
| `Structure/2024/LK:dsd/HexMapAnimation` | Animated population-hex map for housing structure at DSD level. |

---

## Existing Analyses

| File | Summary |
|------|---------|
| [religion-2012-to-2024.md](religion-2012-to-2024.md) | How religious composition changed between the 2012 and 2024 censuses, at district and DSD level. |
| [a-brief-history-of-admin-regions.md](a-brief-history-of-admin-regions.md) | History of Sri Lanka's provinces and districts from 1833 to the present. |
| [so-sri-lankan/](so-sri-lankan/) | Ranks polling divisions by how closely they mirror the national average across demographics, housing, and infrastructure. |
