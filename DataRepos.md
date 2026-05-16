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
