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
            "Economy": [
                "2012"
            ],
            "Education": [
            ... // 101 lines ...
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
        "where_description": "Countrys with IDs LK",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/f578857e.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Basic/2024/LK/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Basic/2024/LK/Map](images/readme/f578857e.png)

#### 2.02) `Basic/2024/LK-1:district/Map`

```json
{
    "result": {
        "what_description": "Basic information",
        "when_description": "2024",
        "where_description": "Districts within LK-1",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/06554961.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Basic/2024/LK-1:district/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Basic/2024/LK-1:district/Map](images/readme/06554961.png)

#### 2.03) `Basic/2024/LK-2,LK-3/Map`

```json
{
    "result": {
        "what_description": "Basic information",
        "when_description": "2024",
        "where_description": "Provinces with IDs LK-2,LK-3",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/67a36359.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Basic/2024/LK-2,LK-3/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Basic/2024/LK-2,LK-3/Map](images/readme/67a36359.png)

#### 2.04) `Basic/2024/LK-3,LK-9,LK-8/Map`

```json
{
    "result": {
        "what_description": "Basic information",
        "when_description": "2024",
        "where_description": "Provinces with IDs LK-3,LK-9,LK-8",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/c235c32c.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Basic/2024/LK-3,LK-9,LK-8/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Basic/2024/LK-3,LK-9,LK-8/Map](images/readme/c235c32c.png)

#### 2.05) `Basic/2024/LK-5...LK-8/Map`

```json
{
    "result": {
        "what_description": "Basic information",
        "when_description": "2024",
        "where_description": "Provinces, from LK-5 to LK-8",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/5accfe2f.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Basic/2024/LK-5...LK-8/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Basic/2024/LK-5...LK-8/Map](images/readme/5accfe2f.png)

#### 2.06) `Basic/2024/LK-1127025@20/Map`

```json
{
    "result": {
        "what_description": "Basic information",
        "when_description": "2024",
        "where_description": "Gnds within 20 km of LK-1127025",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/7394e15c.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Basic/2024/LK-1127025@20/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Basic/2024/LK-1127025@20/Map](images/readme/7394e15c.png)

#### 2.07) `Basic/2024/LK-1103&EC-01B/Map`

```json
{
    "result": {
        "what_description": "Basic information",
        "when_description": "2024",
        "where_description": "Intersection of Dsd LK-1103 and Pd EC-01B",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/0816017b.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Basic/2024/LK-1103&EC-01B/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Basic/2024/LK-1103&EC-01B/Map](images/readme/0816017b.png)

### 3) Sub-Regions

#### 3.01) `Basic/2024/LK-61/Map`

```json
{
    "result": {
        "what_description": "Basic information",
        "when_description": "2024",
        "where_description": "Districts with IDs LK-61",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/49033146.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Basic/2024/LK-61/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Basic/2024/LK-61/Map](images/readme/49033146.png)

#### 3.02) `Basic/2024/LK-71:dsd/Map`

```json
{
    "result": {
        "what_description": "Basic information",
        "when_description": "2024",
        "where_description": "Dsds within LK-71",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/7ae293c9.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Basic/2024/LK-71:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Basic/2024/LK-71:dsd/Map](images/readme/7ae293c9.png)

#### 3.03) `Basic/2024/LK-81:pd/Map`

```json
{
    "result": {
        "what_description": "Basic information",
        "when_description": "2024",
        "where_description": "Pds within LK-81",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/771bc5cb.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Basic/2024/LK-81:pd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Basic/2024/LK-81:pd/Map](images/readme/771bc5cb.png)

### 4) Religion

#### 4.01) `Religion/2012/LK-23:dsd/Map`

```json
{
    "result": {
        "what_description": "Population distributed by religious affiliation such as Buddhist, Hindu, Islam, and Christian",
        "when_description": "2012",
        "where_description": "Dsds within LK-23",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/b3b4de30.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Religion/2012/LK-23:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Religion/2012/LK-23:dsd/Map](images/readme/b3b4de30.png)

#### 4.02) `Religion/2024/LK-23:dsd/Map`

```json
{
    "result": {
        "what_description": "Population distributed by religious affiliation (e.g. Buddhist, Hindu, Islam)",
        "when_description": "2024",
        "where_description": "Dsds within LK-23",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/9d4ac0b7.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Religion/2024/LK-23:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Religion/2024/LK-23:dsd/Map](images/readme/9d4ac0b7.png)

#### 4.03) `Religion/2012/LK-1103:gnd/JSON`

```json
{
    "result": {
        "what_description": "Population distributed by religious affiliation such as Buddhist, Hindu, Islam, and Christian",
        "when_description": "2012",
        "where_description": "Gnds within LK-1103",
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
                ... // 1185 lines ...
                "Other": 251
            },
            "total_value": 323223,
            "pct_values": {
                "Islam": 0.4177,
                "Hindu": 0.227,
                "Buddhist": 0.1901,
                "RomanCatholic": 0.1313,
                "OtherChristian": 0.0332,
                "Other": 0.0008
            }
        },
        "source": "Census of Population and Housing 2012",
        "source_url": "https://www.statistics.gov.lk/Resource/en/Population/CPH_2011/CPH_2012_5Per_Rpt.pdf",
        "description": "Population distributed by religious affiliation such as Buddhist, Hindu, Islam, and Christian",
        "cmd": "Religion/2012/LK-1103:gnd/JSON"
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
        "where_description": "Gnds within LK-1127",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/7e388621.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Religion/2012/LK-1127:gnd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Religion/2012/LK-1127:gnd/Map](images/readme/7e388621.png)

#### 4.05) `Religion/2012/LK-41:dsd/Map:2nd`

```json
{
    "result": {
        "what_description": "Population distributed by religious affiliation such as Buddhist, Hindu, Islam, and Christian",
        "when_description": "2012",
        "where_description": "Dsds within LK-41",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/6a015ea6.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Religion/2012/LK-41:dsd/Map:2nd"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Religion/2012/LK-41:dsd/Map:2nd](images/readme/6a015ea6.png)

#### 4.06) `Religion/2012/LK-51:dsd/Map:Buddhist`

```json
{
    "result": {
        "what_description": "Population distributed by religious affiliation such as Buddhist, Hindu, Islam, and Christian",
        "when_description": "2012",
        "where_description": "Dsds within LK-51",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/aaaf7c50.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Religion/2012/LK-51:dsd/Map:Buddhist"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Religion/2012/LK-51:dsd/Map:Buddhist](images/readme/aaaf7c50.png)

### 5) Ethnicity

#### 5.01) `Ethnicity/2024/LK-53:dsd/Map`

```json
{
    "result": {
        "what_description": "Population distributed by ethnic group (e.g. Sinhalese, Sri Lanka Tamil, Moor)",
        "when_description": "2024",
        "where_description": "Dsds within LK-53",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/823b017d.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Ethnicity/2024/LK-53:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Ethnicity/2024/LK-53:dsd/Map](images/readme/823b017d.png)

#### 5.02) `Ethnicity/Latest/LK-2:dsd/Map`

```json
{
    "result": {
        "what_description": "Population distributed by ethnic group (e.g. Sinhalese, Sri Lanka Tamil, Moor)",
        "when_description": "Latest",
        "where_description": "Dsds within LK-2",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/1abd1338.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Ethnicity/Latest/LK-2:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Ethnicity/Latest/LK-2:dsd/Map](images/readme/1abd1338.png)

#### 5.03) `Ethnicity/2024/LK-23:dsd/Map`

```json
{
    "result": {
        "what_description": "Population distributed by ethnic group (e.g. Sinhalese, Sri Lanka Tamil, Moor)",
        "when_description": "2024",
        "where_description": "Dsds within LK-23",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/5348caed.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Ethnicity/2024/LK-23:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Ethnicity/2024/LK-23:dsd/Map](images/readme/5348caed.png)

### 6) Elections

#### 6.01) `Parliamentary/2024/LK/JSON`

```json
{
    "result": {
        "what_description": "Results of the 2024 Sri Lankan Parliamentary Election",
        "when_description": "2024",
        "where_description": "Countrys with IDs LK",
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
                    ... // 1336 lines ...
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
        "source_url": "https://www.elections.gov.lk",
        "cmd": "Parliamentary/2024/LK/JSON"
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
        "where_description": "Pds within LK-2",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/3dae5731.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Presidential/Latest/LK-2:pd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Presidential/Latest/LK-2:pd/Map](images/readme/3dae5731.png)

### 7) History

#### 7.01) `Basic/2012/LK-pre1984:district/JSON`

```json
{
    "result": {
        "what_description": "Basic information",
        "when_description": "2012",
        "where_description": "Districts within LK",
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
            ... // 264 lines ...
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
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Basic/2012/LK-pre1984:district/JSON"
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
        "where_description": "Districts within LK",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/488036dc.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Religion/2012/LK-pre1959:district/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Religion/2012/LK-pre1959:district/Map](images/readme/488036dc.png)

#### 7.03) `Basic/2012/LK-pre1845:province/Map`

```json
{
    "result": {
        "what_description": "Basic information",
        "when_description": "2012",
        "where_description": "Provinces within LK",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/9c727ec1.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Basic/2012/LK-pre1845:province/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Basic/2012/LK-pre1845:province/Map](images/readme/9c727ec1.png)

#### 7.04) `Ethnicity/2012/LK-23-pre2019:dsd/Map`

```json
{
    "result": {
        "what_description": "Population distributed by ethnic group such as Sinhalese, Sri Lanka Tamil, Moor, and others",
        "when_description": "2012",
        "where_description": "Dsds within LK-23",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/2ec9cf9f.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Ethnicity/2012/LK-23-pre2019:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Ethnicity/2012/LK-23-pre2019:dsd/Map](images/readme/2ec9cf9f.png)

#### 7.05) `Ethnicity/2024/LK-23-pre2019:dsd/Map`

```json
{
    "result": {
        "what_description": "Population distributed by ethnic group (e.g. Sinhalese, Sri Lanka Tamil, Moor)",
        "when_description": "2024",
        "where_description": "Dsds within LK-23",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/5348caed.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Ethnicity/2024/LK-23-pre2019:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Ethnicity/2024/LK-23-pre2019:dsd/Map](images/readme/5348caed.png)

### 8) Other-Whats

#### 8.01) `AgeGroup/2024/LK-1:district/Map`

```json
{
    "result": {
        "what_description": "Population distributed across standard age bands",
        "when_description": "2024",
        "where_description": "Districts within LK-1",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/c035ac55.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "AgeGroup/2024/LK-1:district/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![AgeGroup/2024/LK-1:district/Map](images/readme/c035ac55.png)

#### 8.02) `Communication/2012/LK-2:dsd/Map`

```json
{
    "result": {
        "what_description": "Households classified by ownership of communication items such as telephone, radio, and television",
        "when_description": "2012",
        "where_description": "Dsds within LK-2",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/7dfd94e9.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Communication/2012/LK-2:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Communication/2012/LK-2:dsd/Map](images/readme/7dfd94e9.png)

#### 8.03) `ConstructionYear/2012/LK-3:dsd/Map`

```json
{
    "result": {
        "what_description": "Housing units classified by the decade or period in which they were constructed",
        "when_description": "2012",
        "where_description": "Dsds within LK-3",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/30cf6f6d.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "ConstructionYear/2012/LK-3:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![ConstructionYear/2012/LK-3:dsd/Map](images/readme/30cf6f6d.png)

#### 8.04) `Fuel/2024/LK-4:dsd/Map`

```json
{
    "result": {
        "what_description": "Number of households classified by the main fuel or energy source used for cooking",
        "when_description": "2024",
        "where_description": "Dsds within LK-4",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/e28cd398.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Fuel/2024/LK-4:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Fuel/2024/LK-4:dsd/Map](images/readme/e28cd398.png)

#### 8.05) `Economy/2012/LK-5:dsd/Map`

```json
{
    "result": {
        "what_description": "Population classified by economic activity status including employed, unemployed, and economically inactive",
        "when_description": "2012",
        "where_description": "Dsds within LK-5",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/25b52575.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Economy/2012/LK-5:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Economy/2012/LK-5:dsd/Map](images/readme/25b52575.png)

#### 8.06) `Education/2012/LK-6:dsd/Map`

```json
{
    "result": {
        "what_description": "Population classified by the highest level of educational qualification attained",
        "when_description": "2012",
        "where_description": "Dsds within LK-6",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/d6f800e3.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Education/2012/LK-6:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Education/2012/LK-6:dsd/Map](images/readme/d6f800e3.png)

#### 8.07) `Energy/2024/LK-7:dsd/Map`

```json
{
    "result": {
        "what_description": "Households classified by the main energy fuel used for any purpose",
        "when_description": "2024",
        "where_description": "Dsds within LK-7",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/7cbe1400.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Energy/2024/LK-7:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Energy/2024/LK-7:dsd/Map](images/readme/7cbe1400.png)

#### 8.08) `Floor/2024/LK-2,LK-3/Map`

```json
{
    "result": {
        "what_description": "Households classified by the main material used for floor construction (e.g. cement, tile, mud)",
        "when_description": "2024",
        "where_description": "Provinces with IDs LK-2,LK-3",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/30a639ed.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Floor/2024/LK-2,LK-3/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Floor/2024/LK-2,LK-3/Map](images/readme/30a639ed.png)

#### 8.09) `Gender/2024/LK-1127025@20/Map`

```json
{
    "result": {
        "what_description": "Population broken down by male and female",
        "when_description": "2024",
        "where_description": "Gnds within 20 km of LK-1127025",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/4f5ded21.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Gender/2024/LK-1127025@20/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Gender/2024/LK-1127025@20/Map](images/readme/4f5ded21.png)

#### 8.10) `GenderAndAgeGroup/2024/LK-1103/Map`

```json
{
    "result": {
        "what_description": "Population cross-tabulated by sex and age group",
        "when_description": "2024",
        "where_description": "Dsds with IDs LK-1103",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/d3efd811.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "GenderAndAgeGroup/2024/LK-1103/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![GenderAndAgeGroup/2024/LK-1103/Map](images/readme/d3efd811.png)

#### 8.11) `Lighting/2024/LK-61:dsd/Map`

```json
{
    "result": {
        "what_description": "Number of households classified by their primary source of lighting",
        "when_description": "2024",
        "where_description": "Dsds within LK-61",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/fb1cd9f5.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Lighting/2024/LK-61:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Lighting/2024/LK-61:dsd/Map](images/readme/fb1cd9f5.png)

#### 8.12) `Local/2025/LK:district/Map`

```json
{
    "result": {
        "what_description": "Results of the 2025 Sri Lankan Local Election",
        "when_description": "2025",
        "where_description": "Districts within LK",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/2210f68e.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Local/2025/LK:district/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Local/2025/LK:district/Map](images/readme/2210f68e.png)

#### 8.13) `Marital/2012/LK-3,LK-9,LK-8/Map`

```json
{
    "result": {
        "what_description": "Population classified by marital status such as never married, married, widowed, and divorced",
        "when_description": "2012",
        "where_description": "Provinces with IDs LK-3,LK-9,LK-8",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/d1dd8a40.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Marital/2012/LK-3,LK-9,LK-8/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Marital/2012/LK-3,LK-9,LK-8/Map](images/readme/d1dd8a40.png)

#### 8.14) `Occupancy/2012/LK-1127:gnd/Map`

```json
{
    "result": {
        "what_description": "Housing units classified by occupancy status, distinguishing occupied from vacant units",
        "when_description": "2012",
        "where_description": "Gnds within LK-1127",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/315aa0b9.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Occupancy/2012/LK-1127:gnd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Occupancy/2012/LK-1127:gnd/Map](images/readme/315aa0b9.png)

#### 8.15) `Ownership/2012/LK-53:dsd/Map`

```json
{
    "result": {
        "what_description": "Households classified by the ownership status of their dwelling, such as owned or rented",
        "when_description": "2012",
        "where_description": "Dsds within LK-53",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/3f30ebc0.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Ownership/2012/LK-53:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Ownership/2012/LK-53:dsd/Map](images/readme/3f30ebc0.png)

#### 8.16) `Quarters/2012/LK-4...LK-6/Map`

```json
{
    "result": {
        "what_description": "Households classified by the type of living quarters such as housing units, collective living quarters, and makeshift housing",
        "when_description": "2012",
        "where_description": "Provinces, from LK-4 to LK-6",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/57f7cedf.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Quarters/2012/LK-4...LK-6/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Quarters/2012/LK-4...LK-6/Map](images/readme/57f7cedf.png)

#### 8.17) `RelationshipToHead/2012/LK-23:dsd/Map`

```json
{
    "result": {
        "what_description": "Population classified by their relationship to the head of the household",
        "when_description": "2012",
        "where_description": "Dsds within LK-23",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/a748a3ff.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "RelationshipToHead/2012/LK-23:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![RelationshipToHead/2012/LK-23:dsd/Map](images/readme/a748a3ff.png)

#### 8.18) `Roof/2024/LK-9:dsd/Map`

```json
{
    "result": {
        "what_description": "Housing units classified by the main material used for roof construction (e.g. tile, sheet, concrete)",
        "when_description": "2024",
        "where_description": "Dsds within LK-9",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/6a16ce08.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Roof/2024/LK-9:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Roof/2024/LK-9:dsd/Map](images/readme/6a16ce08.png)

#### 8.19) `Rooms/2012/LK-2:pd/Map`

```json
{
    "result": {
        "what_description": "Households classified by the number of rooms in the dwelling",
        "when_description": "2012",
        "where_description": "Pds within LK-2",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/7ee47c5c.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Rooms/2012/LK-2:pd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Rooms/2012/LK-2:pd/Map](images/readme/7ee47c5c.png)

#### 8.20) `Structure/2024/LK-3:district/Map`

```json
{
    "result": {
        "what_description": "Housing units classified by structural type (e.g. permanent, semi-permanent, temporary)",
        "when_description": "2024",
        "where_description": "Districts within LK-3",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/7dc46400.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Structure/2024/LK-3:district/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Structure/2024/LK-3:district/Map](images/readme/7dc46400.png)

#### 8.21) `Toilet/2024/LK-5:dsd/Map`

```json
{
    "result": {
        "what_description": "Number of households classified by the type of toilet facility used",
        "when_description": "2024",
        "where_description": "Dsds within LK-5",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/bc70199f.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Toilet/2024/LK-5:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Toilet/2024/LK-5:dsd/Map](images/readme/bc70199f.png)

#### 8.22) `Walls/2024/LK-8:dsd/Map`

```json
{
    "result": {
        "what_description": "Housing units classified by the main material used for wall construction (e.g. brick, cabook, cadjan)",
        "when_description": "2024",
        "where_description": "Dsds within LK-8",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/0e5b4f40.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Walls/2024/LK-8:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Walls/2024/LK-8:dsd/Map](images/readme/0e5b4f40.png)

#### 8.23) `Waste/2012/LK-1:dsd/Map`

```json
{
    "result": {
        "what_description": "Households classified by the method used for solid waste disposal",
        "when_description": "2012",
        "where_description": "Dsds within LK-1",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/dcf9c1e0.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Waste/2012/LK-1:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Waste/2012/LK-1:dsd/Map](images/readme/dcf9c1e0.png)

#### 8.24) `Water/2024/LK-2,LK-4/Map`

```json
{
    "result": {
        "what_description": "Number of households classified by their main source of drinking water",
        "when_description": "2024",
        "where_description": "Provinces with IDs LK-2,LK-4",
        "how_description": "Geographical map visualization",
        "image_path": "/tmp/lanka_data/images/2d05ab1d.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Water/2024/LK-2,LK-4/Map"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Water/2024/LK-2,LK-4/Map](images/readme/2d05ab1d.png)

![Maintainer](https://img.shields.io/badge/maintainer-nuuuwan-red)
![MadeWith](https://img.shields.io/badge/made_with-python-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
