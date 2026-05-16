# Data Repositories

## <https://github.com/nuuuwan/gig-data/tree/master/gig2>

Stores data in tabular form as TSV files.

File name has three parts.

```bash

<what info>.<where info>.<when info>.tsv

```

- <what info> is a "-" seperated description of the what. (e.g. "social-household-lighting)
- <where info> is either "regions", which means data is by multiple region types, or "regions-ec", which means region types pertaining to elections.
- <when info> is an year

The first column "entity_id" represents the region (e.g. LK-11). The remaining columns represent fields.

## <https://github.com/nuuuwan/lk_census_2024>

Stores data from the 2024 Sri Lanka census.

- <https://github.com/nuuuwan/lk_census_2024/blob/main/data/GN_housing_excel/Occupied-Housing-Units/data.tsv> - Contains number of housing units

- <https://github.com/nuuuwan/lk_census_2024/blob/main/data/GN_population_excel/Population-by-Age-Group/data.tsv> - Population by Age Group

- <https://github.com/nuuuwan/lk_census_2024/blob/main/data/GN_population_excel/Population-by-Sex/data.tsv> - Population by Sex

- <https://github.com/nuuuwan/lk_census_2024/tree/main/data/HH_GND_excel> - various household related info
