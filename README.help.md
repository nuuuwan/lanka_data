# Help

Command API documentation

## What

Available data categories:

### census-housing

- **Communication**: Distribution of households by communication items.
- **ConstructionYear**: Distribution of housing by year of construction.
- **Electricity**: Distribution of electricity access by sector.
- **Floor**: Distribution of housing by floor material type.
- **Fuel**: Distribution of housing by cooking fuel type.
- **Housing**: Housing distribution by sector.
- **Informal**: Distribution of informal housing.
- **Lighting**: Distribution of housing by lighting source.
- **Materials**: Distribution of housing construction materials.
- **Occupancy**: Distribution of households by occupancy status.
- **Ownership**: Distribution of housing by ownership status.
- **Persons**: Distribution of households by number of persons.
- **Quarters**: Distribution of households by living quarters type.
- **Roof**: Distribution of housing by roof material type.
- **Rooms**: Distribution of housing by number of rooms.
- **Structure**: Distribution of housing by building structure type.
- **Tenure**: Distribution of housing by tenure.
- **Toilet**: Distribution of housing by toilet facility type.
- **Unit**: Distribution of housing by unit type.
- **Walls**: Distribution of housing by wall material type.
- **Waste**: Distribution of housing by solid waste disposal method.
- **Water**: Distribution of housing by drinking water source.

### census-population

- **AgeGroup**: Distribution of the population by age groups.
- **AgriOccupations**: Distribution of agricultural occupations.
- **Attendance**: School attendance distribution by region.
- **Dependency**: Dependency ratios and age group distribution.
- **Disability**: Disability prevalence distribution.
- **Economy**: Distribution of the population by economic activity.
- **Education**: Distribution of the population by educational attainment.
- **Employment**: Distribution of the population by employment status.
- **Enrollment**: Educational enrollment distribution by region.
- **Ethnicity**: Distribution of the population by ethnic group
- **Fertility**: Fertility rates by sector.
- **Gender**: Distribution of the population by gender.
- **Growth**: Population growth rates and intercensal metrics.
- **Inactive**: Distribution of economically inactive population.
- **Industry**: Distribution of population by employment industry.
- **Laborforce**: Labor force participation rates by gender.
- **Literacy**: Literacy rates by sector and gender.
- **Marital**: Distribution of the population by marital status.
- **Migration**: Distribution of population by migration patterns.
- **NonAgriEmployment**: Non-agricultural employment by occupation.
- **NotAttending**: School non-attendance distribution by region.
- **Occupations**: Distribution of population by occupations.
- **Population**: Population overview with sex ratio and density.
- **RelationshipToHead**: Distribution of the population by relationship to household head.
- **Religion**: Distribution of the population by religious affiliation.
- **Sectoral**: Distribution of population by sector (rural, urban, estate).
- **Sectors**: Employment distribution by sector (agricultural vs. non-agricultural).
- **Speaking**: Speaking disability distribution by region.
- **Unemployment**: Unemployment rates by education and gender.

### election

- **Local**: Local government election results.
- **LocalSummary**: Local government election summary
- **Parliamentary**: Parliamentary election results.
- **ParliamentarySummary**: Parliamentary election summary
- **Presidential**: Presidential election results by region.
- **PresidentialSummary**: Presidential election summary

### rivers

- **Catchment**: River catchment area statistics.
- **RiverLen**: River length statistics.

## When

Specify the year or date for which you want to retrieve data. If the exact time is not available, the closest available data will be used.

## Where

### <region_id>

Returns data for the specified <region_id>.

**Examples:**
- `LK`

### <region_id>:<region_type>

Returns data for child regions of type <region_type> in <region_id>.

**Examples:**
- `LK:district`

### <region_id1>,<region_id2>

Returns data for a list of regions.

**Examples:**
- `LK-1,LK-2`

### <region_id1>,<region_id2>,<region_id3>

Returns data for a list of regions.

**Examples:**
- `LK-1,LK-2,LK-3`

### <region_id1>...<region_id2>

Returns data for a range of regions.

**Examples:**
- `LK-1...LK-2`

### <region_id>@<distance>

Returns regions of the same type within a specified distance of <region_id>.

**Examples:**
- `LK-1127025@20`

## How

<base>:<Optional param>

### Bases

- **BubbleMap**: Renders data as a map with bubble markers sized by values and colored by categories
- **GeoJSON**: Exports data as GeoJSON format with geometries
- **PieChart**: Renders data as a pie chart with slices representing categories sized by their values
- **TreeMap**: Renders data as a tree map with rectangles sized by values and colored by categories
- **ScatterPlot**: Renders data as a scatter plot comparing two categories with fitted correlation line and statistics
- **SquareMap**: Renders data as a square tile map with each region assigned a square colored by values
- **LineChart**: Renders data as a line chart with categories on x-axis and values as lines over time or categories
- **QuadrantChart**: Renders data as a quadrant chart dividing regions into 4 quadrants based on two variable values
- **TriangleMap**: Renders data as a triangular tile map with each region assigned a triangle colored by values
- **TSV**: Exports data as TSV format with regions and categories
- **Parquet**: Exports data as Parquet columnar format
- **Map**: Map
- **BivariateMap**: Renders data as a bivariate map showing correlation between two variables using a 3x3 color palette
- **BarChart**: Renders data as bar chart with regions on x-axis
- **ChartSpec**: Exports data as chart specification in JSON format
- **BumpChart**: Renders data as a bump chart showing ranking changes of items across categories or time periods
- **CSV**: Exports data as CSV format with regions and categories
- **HexMap**: Renders data as a hexagonal tile map with each region assigned a hexagon colored by values
- **Histogram**: Renders data as a histogram with binned intervals showing frequency distribution
- **JSON**: Exports data as JSON format with region and category values
- **StackedBarChart**: Renders data as a stacked bar chart with regions on x-axis and categories as stacked segments
- **UnitHexMap**: Renders data as a unit hexagonal map with exactly one hexagon per region
- **UnitSquareMap**: Renders data as a unit square map with exactly one square per region
- **UnitTriangleMap**: Renders data as a unit triangular map with exactly one triangle per region

### Parameters

- **1st**: Highlights the most common category in each region
- **Top**: Highlights the most common category in each region
- **2nd**: Highlights the 2nd most common category in each region
- **3rd**: Highlights the 3rd most common category in each region
- **Bottom**: Highlights the least common category in each region
- **1stPct**: Shows the percentage share of the most common category in each region
- **2ndPct**: Shows the percentage share of the 2nd most common category in each region
- **Change**: Shows the change in the selected metric between two time periods. Requires an interval (two years) in the When field.
- **Top3**: Colors each region based on its top 3 categories combined, assigning a unique color to each unique combination
- **Diversity**: Shows the Religious Diversity Index (RDI) for each region, measuring how evenly distributed the categories are
- **DiversityPew**: Shows the Pew-adjusted Religious Diversity Index for each region, using grouped categories similar to Pew Research methodology
