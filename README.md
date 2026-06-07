# Lanka Data

This repo implements a simple interface to query data about Sri Lanka.

## Data Sources

- [Census of Population and Housing 2012](https://www.statistics.gov.lk/Resource/en/Population/CPH_2011/CPH_2012_5Per_Rpt.pdf)
- [Department of Census and Statistics, Sri Lanka](https://www.statistics.gov.lk/)
- [Election Commission of Sri Lanka](https://www.elections.gov.lk)
- [lanka_data](https://github.com/nuuuwan/lanka_data/blob/main/README.md)

## Usage

### Run Code

```python
from lanka_data import Db


db = Db("<cmd>")
output = db.run()
print(output)

```

### workflows/run.py

```bash
python workflows/run.py <cmd>
```

### workflows/console.py

```bash
python workflows/console.py <cmd>

/Where/What/When/How

> /<cmd>
```

## Example cmds (`<cmd>`)

### 1) Help

#### 1.01) `*`

```json
{
    "result": {
        "what_to_whens": {
            "AgeGroup": [
                "2012",
                "2024"
            ],
            "Basic": [
                "2024"
            ],
            "Communication": [
                "2012"
            ],
            "ConstructionYear": [
                "2012"
            ],
            "CookingFuel": [
                "2024"
            ],
            "Economy": [
            ... // 121 lines ...
            "Water": [
                "2012",
                "2024"
            ]
        },
        "where": [
            "LK*",
            "EC-*",
            "LG-*"
        ],
        "how": [
            "JSON",
            "Map"
        ],
        "source": "lanka_data",
        "source_url": "https://github.com/nuuuwan/lanka_data/blob/main/README.md"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

### 2) Selection

#### 2.01) `Basic/2024/LK/Map`

```json
{
    "result": {
        "what_description": "Basic information",
        "when_description": "2024",
        "where_description": "Regions of type country with IDs LK",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/7aad53e5.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Basic/2024/LK/Map](images/readme/7aad53e5.png)

#### 2.02) `Basic/2024/LK-1:district/Map`

```json
{
    "result": {
        "what_description": "Basic information",
        "when_description": "2024",
        "where_description": "Regions of type district within parent region LK-1",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/dcd97d2b.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Basic/2024/LK-1:district/Map](images/readme/dcd97d2b.png)

#### 2.03) `Basic/2024/LK-2,LK-3/Map`

```json
{
    "result": {
        "what_description": "Basic information",
        "when_description": "2024",
        "where_description": "Regions of type province with IDs LK-2,LK-3",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/19881b4c.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Basic/2024/LK-2,LK-3/Map](images/readme/19881b4c.png)

#### 2.04) `Basic/2024/LK-3,LK-9,LK-8/Map`

```json
{
    "result": {
        "what_description": "Basic information",
        "when_description": "2024",
        "where_description": "Regions of type province with IDs LK-3,LK-9,LK-8",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/bbb12bdc.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Basic/2024/LK-3,LK-9,LK-8/Map](images/readme/bbb12bdc.png)

#### 2.05) `Basic/2024/LK-5...LK-8/Map`

```json
{
    "result": {
        "what_description": "Basic information",
        "when_description": "2024",
        "where_description": "Regions of type province, from LK-5 to LK-8",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/4fbd45c2.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Basic/2024/LK-5...LK-8/Map](images/readme/4fbd45c2.png)

#### 2.06) `Basic/2024/LK-1127025@20/Map`

```json
{
    "result": {
        "what_description": "Basic information",
        "when_description": "2024",
        "where_description": "Regions of type gnd within 20 km of LK-1127025",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/b1457038.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Basic/2024/LK-1127025@20/Map](images/readme/b1457038.png)

#### 2.07) `Basic/2024/LK-1103&EC-01B/Map`

```json
{
    "result": {
        "what_description": "Basic information",
        "when_description": "2024",
        "where_description": "Intersection of dsd LK-1103 and pd EC-01B",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/b1d5861a.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Basic/2024/LK-1103&EC-01B/Map](images/readme/b1d5861a.png)

### 3) Sub-Regions

#### 3.01) `Basic/2024/LK-61/Map`

```json
{
    "result": {
        "what_description": "Basic information",
        "when_description": "2024",
        "where_description": "Regions of type district with IDs LK-61",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/aedd9a9d.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Basic/2024/LK-61/Map](images/readme/aedd9a9d.png)

#### 3.02) `Basic/2024/LK-71:dsd/Map`

```json
{
    "result": {
        "what_description": "Basic information",
        "when_description": "2024",
        "where_description": "Regions of type dsd within parent region LK-71",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/a7caacfb.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Basic/2024/LK-71:dsd/Map](images/readme/a7caacfb.png)

#### 3.03) `Basic/2024/LK-81:pd/Map`

```json
{
    "result": {
        "what_description": "Basic information",
        "when_description": "2024",
        "where_description": "Regions of type pd within parent region LK-81",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/099634e3.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Basic/2024/LK-81:pd/Map](images/readme/099634e3.png)

### 4) Religion

#### 4.01) `Religion/2012/LK-23:dsd/Map`

```json
{
    "result": {
        "what_description": "Population distributed by religious affiliation such as Buddhist, Hindu, Islam, and Christian",
        "when_description": "2012",
        "where_description": "Regions of type dsd within parent region LK-23",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/eaa344b5.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Religion/2012/LK-23:dsd/Map](images/readme/eaa344b5.png)

#### 4.02) `Religion/2024/LK-23:dsd/Map`

```json
{
    "result": {
        "what_description": "Population distributed by religious affiliation (e.g. Buddhist, Hindu, Islam)",
        "when_description": "2024",
        "where_description": "Regions of type dsd within parent region LK-23",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/7f56315f.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Religion/2024/LK-23:dsd/Map](images/readme/7f56315f.png)

#### 4.03) `Religion/2012/LK-1103:gnd/JSON`

```json
{
    "result": {
        "what_description": "Population distributed by religious affiliation such as Buddhist, Hindu, Islam, and Christian",
        "when_description": "2012",
        "where_description": "Regions of type gnd within parent region LK-1103",
        "how_description": "JSON data",
        "data_list": [
            {
                "region_id": "LK-1103005",
                "region_name": "Sammanthranapura",
                "region_type": "gnd",
                "history_year": "Current",
                "area_sqkm": 0.18,
                "center_lat": 6.977933,
                "center_lng": 79.878128,
                "num": " ",
                "country_id": "LK",
                "province_id": "LK-1",
                "district_id": "LK-11",
                "dsd_id": "LK-1103",
                ... // 1184 lines ...
                "Other Christian": 10715,
                "Other": 251
            },
            "total_value": 323223,
            "pct_values": {
                "Islam": 0.4177,
                "Hindu": 0.227,
                "Buddhist": 0.1901,
                "Roman Catholic": 0.1313,
                "Other Christian": 0.0332,
                "Other": 0.0008
            }
        },
        "source": "Census of Population and Housing 2012",
        "source_url": "https://www.statistics.gov.lk/Resource/en/Population/CPH_2011/CPH_2012_5Per_Rpt.pdf",
        "description": "Population distributed by religious affiliation such as Buddhist, Hindu, Islam, and Christian"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

#### 4.04) `Religion/2012/LK-1127:gnd/Map`

```json
{
    "result": {
        "what_description": "Population distributed by religious affiliation such as Buddhist, Hindu, Islam, and Christian",
        "when_description": "2012",
        "where_description": "Regions of type gnd within parent region LK-1127",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/d125b289.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Religion/2012/LK-1127:gnd/Map](images/readme/d125b289.png)

#### 4.05) `Religion/2012/LK-41:dsd/Map:2nd`

```json
{
    "result": {
        "what_description": "Population distributed by religious affiliation such as Buddhist, Hindu, Islam, and Christian",
        "when_description": "2012",
        "where_description": "Regions of type dsd within parent region LK-41",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/bd492166.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Religion/2012/LK-41:dsd/Map:2nd](images/readme/bd492166.png)

#### 4.06) `Religion/2012/LK-51:dsd/Map:Buddhist`

```json
{
    "result": {
        "what_description": "Population distributed by religious affiliation such as Buddhist, Hindu, Islam, and Christian",
        "when_description": "2012",
        "where_description": "Regions of type dsd within parent region LK-51",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/3c31bb67.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Religion/2012/LK-51:dsd/Map:Buddhist](images/readme/3c31bb67.png)

### 5) Ethnicity

#### 5.01) `Ethnicity/2024/LK-53:dsd/Map`

```json
{
    "result": {
        "what_description": "Population distributed by ethnic group (e.g. Sinhalese, Sri Lanka Tamil, Moor)",
        "when_description": "2024",
        "where_description": "Regions of type dsd within parent region LK-53",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/577176c5.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Ethnicity/2024/LK-53:dsd/Map](images/readme/577176c5.png)

#### 5.02) `Ethnicity/Latest/LK-2:dsd/Map`

```json
{
    "result": {
        "what_description": "Population distributed by ethnic group (e.g. Sinhalese, Sri Lanka Tamil, Moor)",
        "when_description": "Latest",
        "where_description": "Regions of type dsd within parent region LK-2",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/98bdd86e.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Ethnicity/Latest/LK-2:dsd/Map](images/readme/98bdd86e.png)

#### 5.03) `Ethnicity/2024/LK-23:dsd/Map`

```json
{
    "result": {
        "what_description": "Population distributed by ethnic group (e.g. Sinhalese, Sri Lanka Tamil, Moor)",
        "when_description": "2024",
        "where_description": "Regions of type dsd within parent region LK-23",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/db891200.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Ethnicity/2024/LK-23:dsd/Map](images/readme/db891200.png)

### 6) Elections

#### 6.01) `Parliamentary/2024/LK/JSON`

```json
{
    "result": {
        "what_description": "Results of the 2024 Sri Lankan Parliamentary Election",
        "when_description": "2024",
        "where_description": "Regions of type country with IDs LK",
        "how_description": "JSON data",
        "data_list": [
            {
                "region_id": "LK",
                "region_name": "Sri Lanka",
                "region_type": "country",
                "history_year": "Current",
                "area_sqkm": 65983.58,
                "center_lat": 7.621863,
                "center_lng": 80.698448,
                "summary": {
                    "electors": 17140354,
                    "polled": 11815246,
                    "valid": 11148006,
                    "rejected": 667240,
                    ... // 1335 lines ...
                "IND28-13": 0.0,
                "IND36-13": 0.0,
                "IND42-13": 0.0,
                "IND34-13": 0.0,
                "IND26-12": 0.0,
                "IND05-13": 0.0,
                "IND07-13": 0.0,
                "IND08-14": 0.0,
                "IND32-13": 0.0,
                "IND40-13": 0.0,
                "IND37-13": 0.0,
                "IND33-13": 0.0
            }
        },
        "source": "Election Commission of Sri Lanka",
        "source_url": "https://www.elections.gov.lk"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

#### 6.02) `Presidential/Latest/LK-2:pd/Map`

```json
{
    "result": {
        "what_description": "Results of the 2024 Sri Lankan Presidential Election",
        "when_description": "Latest",
        "where_description": "Regions of type pd within parent region LK-2",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/514179a3.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Presidential/Latest/LK-2:pd/Map](images/readme/514179a3.png)

### 7) History

#### 7.01) `Basic/2012/LK-pre1984:district/JSON`

```json
{
    "result": {
        "what_description": "Basic information",
        "when_description": "2012",
        "where_description": "Regions of type district within parent region LK",
        "how_description": "JSON data",
        "data_list": [
            {
                "region_id": "LK-11",
                "region_name": "Colombo",
                "region_type": "district",
                "history_year": "1984",
                "area_sqkm": 688.17,
                "center_lat": 6.869822,
                "center_lng": 80.018487,
                "current_ids": [
                    "LK-11"
                ]
            },
            {
            ... // 263 lines ...
            },
            {
                "region_id": "LK-92",
                "region_name": "Kegalle",
                "region_type": "district",
                "history_year": "1984",
                "area_sqkm": 1657.73,
                "center_lat": 7.104294,
                "center_lng": 80.342772,
                "current_ids": [
                    "LK-92"
                ]
            }
        ],
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

#### 7.02) `Religion/2012/LK-pre1959:district/Map`

```json
{
    "result": {
        "what_description": "Population distributed by religious affiliation such as Buddhist, Hindu, Islam, and Christian",
        "when_description": "2012",
        "where_description": "Regions of type district within parent region LK",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/529e2e3b.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Religion/2012/LK-pre1959:district/Map](images/readme/529e2e3b.png)

#### 7.03) `Basic/2012/LK-pre1845:province/Map`

```json
{
    "result": {
        "what_description": "Basic information",
        "when_description": "2012",
        "where_description": "Regions of type province within parent region LK",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/8d311976.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Basic/2012/LK-pre1845:province/Map](images/readme/8d311976.png)

#### 7.04) `Ethnicity/2012/LK-23-pre2019:dsd/Map`

```json
{
    "result": {
        "what_description": "Population distributed by ethnic group such as Sinhalese, Sri Lanka Tamil, Moor, and others",
        "when_description": "2012",
        "where_description": "Regions of type dsd within parent region LK-23",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/32a1e0e6.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Ethnicity/2012/LK-23-pre2019:dsd/Map](images/readme/32a1e0e6.png)

#### 7.05) `Ethnicity/2024/LK-23-pre2019:dsd/Map`

```json
{
    "result": {
        "what_description": "Population distributed by ethnic group (e.g. Sinhalese, Sri Lanka Tamil, Moor)",
        "when_description": "2024",
        "where_description": "Regions of type dsd within parent region LK-23",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/db891200.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Ethnicity/2024/LK-23-pre2019:dsd/Map](images/readme/db891200.png)

### 8) Other-Whats

#### 8.01) `AgeGroup/2024/LK-1:district/Map`

```json
{
    "result": {
        "what_description": "Population distributed across standard age bands",
        "when_description": "2024",
        "where_description": "Regions of type district within parent region LK-1",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/d90cf6de.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![AgeGroup/2024/LK-1:district/Map](images/readme/d90cf6de.png)

#### 8.02) `Communication/2012/LK-2:dsd/Map`

```json
{
    "result": {
        "what_description": "Households classified by ownership of communication items such as telephone, radio, and television",
        "when_description": "2012",
        "where_description": "Regions of type dsd within parent region LK-2",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/78ee004c.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Communication/2012/LK-2:dsd/Map](images/readme/78ee004c.png)

#### 8.03) `ConstructionYear/2012/LK-3:dsd/Map`

```json
{
    "result": {
        "what_description": "Housing units classified by the decade or period in which they were constructed",
        "when_description": "2012",
        "where_description": "Regions of type dsd within parent region LK-3",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/7277a9e9.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![ConstructionYear/2012/LK-3:dsd/Map](images/readme/7277a9e9.png)

#### 8.04) `CookingFuel/2024/LK-4:dsd/Map`

```json
{
    "result": {
        "what_description": "Main energy or fuel type used for cooking (e.g. gas, firewood, electricity) by households",
        "when_description": "2024",
        "where_description": "Regions of type dsd within parent region LK-4",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/93fd117c.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![CookingFuel/2024/LK-4:dsd/Map](images/readme/93fd117c.png)

#### 8.05) `Economy/2012/LK-5:dsd/Map`

```json
{
    "result": {
        "what_description": "Population classified by economic activity status including employed, unemployed, and economically inactive",
        "when_description": "2012",
        "where_description": "Regions of type dsd within parent region LK-5",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/c6bc2ae2.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Economy/2012/LK-5:dsd/Map](images/readme/c6bc2ae2.png)

#### 8.06) `Education/2012/LK-6:dsd/Map`

```json
{
    "result": {
        "what_description": "Population classified by the highest level of educational qualification attained",
        "when_description": "2012",
        "where_description": "Regions of type dsd within parent region LK-6",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/9ab3f647.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Education/2012/LK-6:dsd/Map](images/readme/9ab3f647.png)

#### 8.07) `EnergyFuel/2024/LK-7:dsd/Map`

```json
{
    "result": {
        "what_description": "Households classified by the main energy fuel used for any purpose",
        "when_description": "2024",
        "where_description": "Regions of type dsd within parent region LK-7",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/754e8b73.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![EnergyFuel/2024/LK-7:dsd/Map](images/readme/754e8b73.png)

#### 8.08) `Floor/2024/LK-2,LK-3/Map`

```json
{
    "result": {
        "what_description": "Households classified by the main material used for floor construction (e.g. cement, tile, mud)",
        "when_description": "2024",
        "where_description": "Regions of type province with IDs LK-2,LK-3",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/96c12447.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Floor/2024/LK-2,LK-3/Map](images/readme/96c12447.png)

#### 8.09) `Fuel/2012/LK-5...LK-8/Map`

```json
{
    "result": {
        "what_description": "Households classified by the primary fuel used for cooking",
        "when_description": "2012",
        "where_description": "Regions of type province, from LK-5 to LK-8",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/20259bb8.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Fuel/2012/LK-5...LK-8/Map](images/readme/20259bb8.png)

#### 8.10) `Gender/2024/LK-1127025@20/Map`

```json
{
    "result": {
        "what_description": "Population broken down by male and female",
        "when_description": "2024",
        "where_description": "Regions of type gnd within 20 km of LK-1127025",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/5083c4fc.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Gender/2024/LK-1127025@20/Map](images/readme/5083c4fc.png)

#### 8.11) `GenderAndAgeGroup/2024/LK-1103/Map`

```json
{
    "result": {
        "what_description": "Population cross-tabulated by sex and age group",
        "when_description": "2024",
        "where_description": "Regions of type dsd with IDs LK-1103",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/99b92321.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![GenderAndAgeGroup/2024/LK-1103/Map](images/readme/99b92321.png)

#### 8.12) `HouseholdsCookingFuel/2024/LK-81:pd/Map`

```json
{
    "result": {
        "what_description": "Number of households classified by the main fuel or energy source used for cooking",
        "when_description": "2024",
        "where_description": "Regions of type pd within parent region LK-81",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/fd0c9ed2.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![HouseholdsCookingFuel/2024/LK-81:pd/Map](images/readme/fd0c9ed2.png)

#### 8.13) `HouseholdsLighting/2024/LK-23:dsd/Map`

```json
{
    "result": {
        "what_description": "Number of households classified by their primary source of lighting",
        "when_description": "2024",
        "where_description": "Regions of type dsd within parent region LK-23",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/a5f622e5.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![HouseholdsLighting/2024/LK-23:dsd/Map](images/readme/a5f622e5.png)

#### 8.14) `HouseholdsToilet/2024/LK-41:dsd/Map`

```json
{
    "result": {
        "what_description": "Number of households classified by the type of toilet facility used",
        "when_description": "2024",
        "where_description": "Regions of type dsd within parent region LK-41",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/6474ea5e.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![HouseholdsToilet/2024/LK-41:dsd/Map](images/readme/6474ea5e.png)

#### 8.15) `HouseholdsWater/2024/LK-51:dsd/Map`

```json
{
    "result": {
        "what_description": "Number of households classified by their main source of drinking water",
        "when_description": "2024",
        "where_description": "Regions of type dsd within parent region LK-51",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/703f367d.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![HouseholdsWater/2024/LK-51:dsd/Map](images/readme/703f367d.png)

#### 8.16) `Lighting/2024/LK-61:dsd/Map`

```json
{
    "result": {
        "what_description": "Primary source of lighting (e.g. electricity, kerosene) used by households",
        "when_description": "2024",
        "where_description": "Regions of type dsd within parent region LK-61",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/cca555a6.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Lighting/2024/LK-61:dsd/Map](images/readme/cca555a6.png)

#### 8.17) `Local/2025/LK:district/Map`

```json
{
    "result": {
        "what_description": "Results of the 2025 Sri Lankan Local Election",
        "when_description": "2025",
        "where_description": "Regions of type district within parent region LK",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/f37e1230.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Local/2025/LK:district/Map](images/readme/f37e1230.png)

#### 8.18) `Marital/2012/LK-3,LK-9,LK-8/Map`

```json
{
    "result": {
        "what_description": "Population classified by marital status such as never married, married, widowed, and divorced",
        "when_description": "2012",
        "where_description": "Regions of type province with IDs LK-3,LK-9,LK-8",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/f76234db.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Marital/2012/LK-3,LK-9,LK-8/Map](images/readme/f76234db.png)

#### 8.19) `Occupancy/2012/LK-1127:gnd/Map`

```json
{
    "result": {
        "what_description": "Housing units classified by occupancy status, distinguishing occupied from vacant units",
        "when_description": "2012",
        "where_description": "Regions of type gnd within parent region LK-1127",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/afe92b98.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Occupancy/2012/LK-1127:gnd/Map](images/readme/afe92b98.png)

#### 8.20) `OccupiedUnits/2024/LK-2:district/Map`

```json
{
    "result": {
        "what_description": "Count of occupied housing units",
        "when_description": "2024",
        "where_description": "Regions of type district within parent region LK-2",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/72d5dc0a.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![OccupiedUnits/2024/LK-2:district/Map](images/readme/72d5dc0a.png)

#### 8.21) `Ownership/2012/LK-53:dsd/Map`

```json
{
    "result": {
        "what_description": "Households classified by the ownership status of their dwelling, such as owned or rented",
        "when_description": "2012",
        "where_description": "Regions of type dsd within parent region LK-53",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/6f90ddff.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Ownership/2012/LK-53:dsd/Map](images/readme/6f90ddff.png)

#### 8.22) `Persons/2012/LK-1103:gnd/Map`

```json
{
    "result": {
        "what_description": "Households classified by the number of persons living in the dwelling",
        "when_description": "2012",
        "where_description": "Regions of type gnd within parent region LK-1103",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/f6f65c56.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Persons/2012/LK-1103:gnd/Map](images/readme/f6f65c56.png)

#### 8.23) `Quarters/2012/LK-4...LK-6/Map`

```json
{
    "result": {
        "what_description": "Households classified by the type of living quarters such as housing units, collective living quarters, and makeshift housing",
        "when_description": "2012",
        "where_description": "Regions of type province, from LK-4 to LK-6",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/7d0daf3f.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Quarters/2012/LK-4...LK-6/Map](images/readme/7d0daf3f.png)

#### 8.24) `RelationshipToHead/2012/LK-23:dsd/Map`

```json
{
    "result": {
        "what_description": "Population classified by their relationship to the head of the household",
        "when_description": "2012",
        "where_description": "Regions of type dsd within parent region LK-23",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/7b902a0d.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![RelationshipToHead/2012/LK-23:dsd/Map](images/readme/7b902a0d.png)

#### 8.25) `Roof/2024/LK-9:dsd/Map`

```json
{
    "result": {
        "what_description": "Housing units classified by the main material used for roof construction (e.g. tile, sheet, concrete)",
        "when_description": "2024",
        "where_description": "Regions of type dsd within parent region LK-9",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/46043a13.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Roof/2024/LK-9:dsd/Map](images/readme/46043a13.png)

#### 8.26) `Rooms/2012/LK-2:pd/Map`

```json
{
    "result": {
        "what_description": "Households classified by the number of rooms in the dwelling",
        "when_description": "2012",
        "where_description": "Regions of type pd within parent region LK-2",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/361b3f9d.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Rooms/2012/LK-2:pd/Map](images/readme/361b3f9d.png)

#### 8.27) `Structure/2024/LK-3:district/Map`

```json
{
    "result": {
        "what_description": "Housing units classified by structural type (e.g. permanent, semi-permanent, temporary)",
        "when_description": "2024",
        "where_description": "Regions of type district within parent region LK-3",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/dea30d59.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Structure/2024/LK-3:district/Map](images/readme/dea30d59.png)

#### 8.28) `Toilet/2024/LK-5:dsd/Map`

```json
{
    "result": {
        "what_description": "Distribution of households by type of toilet facility (e.g. flush, pit latrine, none)",
        "when_description": "2024",
        "where_description": "Regions of type dsd within parent region LK-5",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/7090dbe0.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Toilet/2024/LK-5:dsd/Map](images/readme/7090dbe0.png)

#### 8.29) `Unit/2012/LK-6:dsd/Map`

```json
{
    "result": {
        "what_description": "Housing units classified by type such as house, flat, annexe, or room",
        "when_description": "2012",
        "where_description": "Regions of type dsd within parent region LK-6",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/7c107e7d.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Unit/2012/LK-6:dsd/Map](images/readme/7c107e7d.png)

#### 8.30) `Wall/2012/LK-7:dsd/Map`

```json
{
    "result": {
        "what_description": "Housing units classified by the main material used for wall construction",
        "when_description": "2012",
        "where_description": "Regions of type dsd within parent region LK-7",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/10599a7f.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Wall/2012/LK-7:dsd/Map](images/readme/10599a7f.png)

#### 8.31) `Walls/2024/LK-8:dsd/Map`

```json
{
    "result": {
        "what_description": "Housing units classified by the main material used for wall construction (e.g. brick, cabook, cadjan)",
        "when_description": "2024",
        "where_description": "Regions of type dsd within parent region LK-8",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/90480a91.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Walls/2024/LK-8:dsd/Map](images/readme/90480a91.png)

#### 8.32) `Waste/2012/LK-1:dsd/Map`

```json
{
    "result": {
        "what_description": "Households classified by the method used for solid waste disposal",
        "when_description": "2012",
        "where_description": "Regions of type dsd within parent region LK-1",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/51d83e79.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Waste/2012/LK-1:dsd/Map](images/readme/51d83e79.png)

#### 8.33) `Water/2024/LK-2,LK-4/Map`

```json
{
    "result": {
        "what_description": "Primary source of drinking water (e.g. pipe-borne, well, river) used by households",
        "when_description": "2024",
        "where_description": "Regions of type province with IDs LK-2,LK-4",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/d34c5c1b.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Water/2024/LK-2,LK-4/Map](images/readme/d34c5c1b.png)

![Maintainer](https://img.shields.io/badge/maintainer-nuuuwan-red)
![MadeWith](https://img.shields.io/badge/made_with-python-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
