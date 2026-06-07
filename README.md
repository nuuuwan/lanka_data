# Lanka Data

This repo implements a simple interface to query data about Sri Lanka.

## Data Sources

- [Census of Population and Housing 2012](https://www.statistics.gov.lk/Resource/en/Population/CPH_2011/CPH_2012_5Per_Rpt.pdf)
- [Department of Census and Statistics, Sri Lanka](https://www.statistics.gov.lk/)
- [Election Commission of Sri Lanka](https://www.elections.gov.lk)

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

### 1) Selection

#### 1.01) `Basic/2024/LK/Map`

```json
{
    "result": {
        "title_items": [
            "Sri Lanka Country",
            "Basic Information",
            "2024",
            "Map"
        ],
        "image_path": "/tmp/lanka_data/images/48853a44.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Basic/2024/LK/Map](images/readme/48853a44.png)

#### 1.02) `Basic/2024/LK-1:district/Map`

```json
{
    "result": {
        "title_items": [
            "Colombo, Gampaha, Kalutara Districts",
            "Basic Information",
            "2024",
            "Map"
        ],
        "image_path": "/tmp/lanka_data/images/d81caf73.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Basic/2024/LK-1:district/Map](images/readme/d81caf73.png)

#### 1.03) `Basic/2024/LK-2,LK-3/Map`

```json
{
    "result": {
        "title_items": [
            "Central, Southern Provinces",
            "Basic Information",
            "2024",
            "Map"
        ],
        "image_path": "/tmp/lanka_data/images/0233d3fd.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Basic/2024/LK-2,LK-3/Map](images/readme/0233d3fd.png)

#### 1.04) `Basic/2024/LK-3,LK-9,LK-8/Map`

```json
{
    "result": {
        "title_items": [
            "Southern, Uva, Sabaragamuwa Provinces",
            "Basic Information",
            "2024",
            "Map"
        ],
        "image_path": "/tmp/lanka_data/images/4399d293.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Basic/2024/LK-3,LK-9,LK-8/Map](images/readme/4399d293.png)

#### 1.05) `Basic/2024/LK-5...LK-8/Map`

```json
{
    "result": {
        "title_items": [
            "Eastern, North Western, North Central, Uva Provinces",
            "Basic Information",
            "2024",
            "Map"
        ],
        "image_path": "/tmp/lanka_data/images/58e0fbfb.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Basic/2024/LK-5...LK-8/Map](images/readme/58e0fbfb.png)

#### 1.06) `Basic/2024/LK-1127025@20/Map`

```json
{
    "result": {
        "title_items": [
            "718 Gnds",
            "Basic Information",
            "2024",
            "Map"
        ],
        "image_path": "/tmp/lanka_data/images/f57d8427.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Basic/2024/LK-1127025@20/Map](images/readme/f57d8427.png)

#### 1.07) `Basic/2024/LK-1103&EC-01B/Map`

```json
{
    "result": {
        "title_items": [
            "24 Gnds",
            "Basic Information",
            "2024",
            "Map"
        ],
        "image_path": "/tmp/lanka_data/images/15de4034.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Basic/2024/LK-1103&EC-01B/Map](images/readme/15de4034.png)

### 2) Sub-Regions

#### 2.01) `Basic/2024/LK-61/Map`

```json
{
    "result": {
        "title_items": [
            "Kurunegala District",
            "Basic Information",
            "2024",
            "Map"
        ],
        "image_path": "/tmp/lanka_data/images/f3dd60f4.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Basic/2024/LK-61/Map](images/readme/f3dd60f4.png)

#### 2.02) `Basic/2024/LK-71:dsd/Map`

```json
{
    "result": {
        "title_items": [
            "22 Dsds",
            "Basic Information",
            "2024",
            "Map"
        ],
        "image_path": "/tmp/lanka_data/images/98aa28e2.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Basic/2024/LK-71:dsd/Map](images/readme/98aa28e2.png)

#### 2.03) `Basic/2024/LK-81:pd/Map`

```json
{
    "result": {
        "title_items": [
            "9 Pds",
            "Basic Information",
            "2024",
            "Map"
        ],
        "image_path": "/tmp/lanka_data/images/bfa7d721.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Basic/2024/LK-81:pd/Map](images/readme/bfa7d721.png)

### 3) Religion

#### 3.01) `Religion/2012/LK-23:dsd/Map`

```json
{
    "result": {
        "title_items": [
            "Kotmale, Hanguranketa (pre 2019), Walapane (pre 2019), Nuwara Eliya (pre 2019), Ambagamuwa Dsds",
            "Religion",
            "2012",
            "Map"
        ],
        "image_path": "/tmp/lanka_data/images/a2b164d4.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Religion/2012/LK-23:dsd/Map](images/readme/a2b164d4.png)

#### 3.02) `Religion/2024/LK-23:dsd/Map`

```json
{
    "result": {
        "title_items": [
            "10 Dsds",
            "Religion",
            "2024",
            "Map"
        ],
        "image_path": "/tmp/lanka_data/images/3147011c.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Religion/2024/LK-23:dsd/Map](images/readme/3147011c.png)

#### 3.03) `Religion/2012/LK-1103:gnd/JSON`

```json
{
    "result": {
        "title_items": [
            "35 Gnds",
            "Religion",
            "2012",
            "JSON"
        ],
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
                ... // 1185 lines ...
                "Roman Catholic": 42435,
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
        "source_url": "https://www.statistics.gov.lk/Resource/en/Population/CPH_2011/CPH_2012_5Per_Rpt.pdf"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

#### 3.04) `Religion/2012/LK-1127:gnd/Map`

```json
{
    "result": {
        "title_items": [
            "20 Gnds",
            "Religion",
            "2012",
            "Map"
        ],
        "image_path": "/tmp/lanka_data/images/0d265f1a.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Religion/2012/LK-1127:gnd/Map](images/readme/0d265f1a.png)

#### 3.05) `Religion/2012/LK-41:dsd/Map:2nd`

```json
{
    "result": {
        "title_items": [
            "15 Dsds",
            "Religion",
            "2012",
            "Map (2nd)"
        ],
        "image_path": "/tmp/lanka_data/images/09cd4316.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Religion/2012/LK-41:dsd/Map:2nd](images/readme/09cd4316.png)

#### 3.06) `Religion/2012/LK-51:dsd/Map:Buddhist`

```json
{
    "result": {
        "title_items": [
            "14 Dsds",
            "Religion",
            "2012",
            "Map (Buddhist)"
        ],
        "image_path": "/tmp/lanka_data/images/6b7887b6.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Religion/2012/LK-51:dsd/Map:Buddhist](images/readme/6b7887b6.png)

### 4) Ethnicity

#### 4.01) `Ethnicity/2024/LK-53:dsd/Map`

```json
{
    "result": {
        "title_items": [
            "11 Dsds",
            "Ethnicity",
            "2024",
            "Map"
        ],
        "image_path": "/tmp/lanka_data/images/8069c8aa.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Ethnicity/2024/LK-53:dsd/Map](images/readme/8069c8aa.png)

#### 4.02) `Ethnicity/Latest/LK-2:dsd/Map`

```json
{
    "result": {
        "title_items": [
            "41 Dsds",
            "Ethnicity",
            "Latest",
            "Map"
        ],
        "image_path": "/tmp/lanka_data/images/bade8195.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Ethnicity/Latest/LK-2:dsd/Map](images/readme/bade8195.png)

#### 4.03) `Ethnicity/2024/LK-23:dsd/Map`

```json
{
    "result": {
        "title_items": [
            "10 Dsds",
            "Ethnicity",
            "2024",
            "Map"
        ],
        "image_path": "/tmp/lanka_data/images/d6f9c6e3.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Ethnicity/2024/LK-23:dsd/Map](images/readme/d6f9c6e3.png)

### 5) Elections

#### 5.01) `Parliamentary/2024/LK/JSON`

```json
{
    "result": {
        "title_items": [
            "Sri Lanka Country",
            "Parliamentary",
            "2024",
            "JSON"
        ],
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
                    ... // 1337 lines ...
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

#### 5.02) `Presidential/Latest/LK-2:pd/Map`

```json
{
    "result": {
        "title_items": [
            "21 Pds",
            "Presidential",
            "Latest",
            "Map"
        ],
        "image_path": "/tmp/lanka_data/images/9aa59ecf.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Presidential/Latest/LK-2:pd/Map](images/readme/9aa59ecf.png)

### 6) History

#### 6.01) `Basic/2012/LK-pre1984:district/JSON`

```json
{
    "result": {
        "title_items": [
            "24 Districts",
            "Basic Information",
            "2012",
            "JSON"
        ],
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
                ... // 265 lines ...
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

#### 6.02) `Religion/2012/LK-pre1959:district/Map`

```json
{
    "result": {
        "title_items": [
            "20 Districts",
            "Religion",
            "2012",
            "Map"
        ],
        "image_path": "/tmp/lanka_data/images/4e535cda.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Religion/2012/LK-pre1959:district/Map](images/readme/4e535cda.png)

#### 6.03) `Basic/2012/LK-pre1845:province/Map`

```json
{
    "result": {
        "title_items": [
            "Western (pre 1845), Central (pre 1886), Southern (pre 1886), Northern (pre 1873), Eastern (pre 1873) Provinces",
            "Basic Information",
            "2012",
            "Map"
        ],
        "image_path": "/tmp/lanka_data/images/a85c9f82.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Basic/2012/LK-pre1845:province/Map](images/readme/a85c9f82.png)

#### 6.04) `Ethnicity/2012/LK-23-pre2019:dsd/Map`

```json
{
    "result": {
        "title_items": [
            "Kotmale, Hanguranketa (pre 2019), Walapane (pre 2019), Nuwara Eliya (pre 2019), Ambagamuwa Dsds",
            "Ethnicity",
            "2012",
            "Map"
        ],
        "image_path": "/tmp/lanka_data/images/d9a29dc7.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Ethnicity/2012/LK-23-pre2019:dsd/Map](images/readme/d9a29dc7.png)

#### 6.05) `Ethnicity/2024/LK-23-pre2019:dsd/Map`

```json
{
    "result": {
        "title_items": [
            "Kotmale, Hanguranketa (pre 2019), Walapane (pre 2019), Nuwara Eliya (pre 2019), Ambagamuwa Dsds",
            "Ethnicity",
            "2024",
            "Map"
        ],
        "image_path": "/tmp/lanka_data/images/a7f89f48.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": false
}
```

![Ethnicity/2024/LK-23-pre2019:dsd/Map](images/readme/a7f89f48.png)

![Maintainer](https://img.shields.io/badge/maintainer-nuuuwan-red)
![MadeWith](https://img.shields.io/badge/made_with-python-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
