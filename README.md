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

### workflows/single.py

Runs single command.

```bash
python workflows/single.py <cmd>
```

### workflows/console.py

Console tool for running commands.

```bash
python workflows/console.py <cmd>

/Where/What/When/How

> /<cmd>
```

## Example cmds (`<cmd>`)

### 1) Help

#### 1.01) Help

```bash
Help
```

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
            ... // 92 lines ...
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
    "cache_hit": true
}
```

Source: [examples/outputs/Help/Output.json](examples/outputs/Help/Output.json)

### 2) Selection

#### 2.01) Map of Basic Information (2024) for Provinces in LK.

```bash
Basic/2024/LK:province/Map
```

```json
{
    "result": {
        "what_description": "Basic Information",
        "when_description": "2024",
        "where_description": "Provinces in LK",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Basic/2024/LK:province/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Basic/2024/LK:province/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Basic/2024/LK:province/Map/Output.json](examples/outputs/Basic/2024/LK:province/Map/Output.json)

![Basic/2024/LK:province/Map](examples/outputs/Basic/2024/LK:province/Map/Image.png)

Source: [examples/outputs/Basic/2024/LK:province/Map/Image.png](examples/outputs/Basic/2024/LK:province/Map/Image.png)

#### 2.02) Map of Basic Information (2024) for Districts in LK-1.

```bash
Basic/2024/LK-1:district/Map
```

```json
{
    "result": {
        "what_description": "Basic Information",
        "when_description": "2024",
        "where_description": "Districts in LK-1",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Basic/2024/LK-1:district/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Basic/2024/LK-1:district/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Basic/2024/LK-1:district/Map/Output.json](examples/outputs/Basic/2024/LK-1:district/Map/Output.json)

![Basic/2024/LK-1:district/Map](examples/outputs/Basic/2024/LK-1:district/Map/Image.png)

Source: [examples/outputs/Basic/2024/LK-1:district/Map/Image.png](examples/outputs/Basic/2024/LK-1:district/Map/Image.png)

#### 2.03) Map of Basic Information (2024) for LK-1, LK-2, LK-3, LK-9, LK-8.

```bash
Basic/2024/LK-1,LK-2,LK-3,LK-9,LK-8/Map
```

```json
{
    "result": {
        "what_description": "Basic Information",
        "when_description": "2024",
        "where_description": "LK-1, LK-2, LK-3, LK-9, LK-8",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Basic/2024/LK-1,LK-2,LK-3,LK-9,LK-8/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Basic/2024/LK-1,LK-2,LK-3,LK-9,LK-8/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Basic/2024/LK-1,LK-2,LK-3,LK-9,LK-8/Map/Output.json](examples/outputs/Basic/2024/LK-1,LK-2,LK-3,LK-9,LK-8/Map/Output.json)

![Basic/2024/LK-1,LK-2,LK-3,LK-9,LK-8/Map](examples/outputs/Basic/2024/LK-1,LK-2,LK-3,LK-9,LK-8/Map/Image.png)

Source: [examples/outputs/Basic/2024/LK-1,LK-2,LK-3,LK-9,LK-8/Map/Image.png](examples/outputs/Basic/2024/LK-1,LK-2,LK-3,LK-9,LK-8/Map/Image.png)

#### 2.04) Map of Basic Information (2024) for LK-5 to LK-8.

```bash
Basic/2024/LK-5...LK-8/Map
```

```json
{
    "result": {
        "what_description": "Basic Information",
        "when_description": "2024",
        "where_description": "LK-5 to LK-8",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Basic/2024/LK-5...LK-8/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Basic/2024/LK-5...LK-8/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Basic/2024/LK-5...LK-8/Map/Output.json](examples/outputs/Basic/2024/LK-5...LK-8/Map/Output.json)

![Basic/2024/LK-5...LK-8/Map](examples/outputs/Basic/2024/LK-5...LK-8/Map/Image.png)

Source: [examples/outputs/Basic/2024/LK-5...LK-8/Map/Image.png](examples/outputs/Basic/2024/LK-5...LK-8/Map/Image.png)

#### 2.05) Map of Basic Information (2024) for Regions within 20.0 km of LK-1127025.

```bash
Basic/2024/LK-1127025@20/Map
```

```json
{
    "result": {
        "what_description": "Basic Information",
        "when_description": "2024",
        "where_description": "Regions within 20.0 km of LK-1127025",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Basic/2024/LK-1127025@20/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Basic/2024/LK-1127025@20/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Basic/2024/LK-1127025@20/Map/Output.json](examples/outputs/Basic/2024/LK-1127025@20/Map/Output.json)

![Basic/2024/LK-1127025@20/Map](examples/outputs/Basic/2024/LK-1127025@20/Map/Image.png)

Source: [examples/outputs/Basic/2024/LK-1127025@20/Map/Image.png](examples/outputs/Basic/2024/LK-1127025@20/Map/Image.png)

### 3) Religion

#### 3.01) Map of Population distributed by religious affiliation such as Buddhist, Hindu, Islam, and Christian (2012) for Divisional Secretariat Divisions in LK-23-pre2019.

```bash
Religion/2012/LK-23:dsd/Map
```

```json
{
    "result": {
        "what_description": "Population distributed by religious affiliation such as Buddhist, Hindu, Islam, and Christian",
        "when_description": "2012",
        "where_description": "Divisional Secretariat Divisions in LK-23-pre2019",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Religion/2012/LK-23:dsd/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Religion/2012/LK-23:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Religion/2012/LK-23:dsd/Map/Output.json](examples/outputs/Religion/2012/LK-23:dsd/Map/Output.json)

![Religion/2012/LK-23:dsd/Map](examples/outputs/Religion/2012/LK-23:dsd/Map/Image.png)

Source: [examples/outputs/Religion/2012/LK-23:dsd/Map/Image.png](examples/outputs/Religion/2012/LK-23:dsd/Map/Image.png)

#### 3.02) Map of GN-level population by religion (Buddhist, Hindu, Islam, Roman Catholic, other Christian, other). (2024) for Divisional Secretariat Divisions in LK-23.

```bash
Religion/2024/LK-23:dsd/Map
```

```json
{
    "result": {
        "what_description": "GN-level population by religion (Buddhist, Hindu, Islam, Roman Catholic, other Christian, other).",
        "when_description": "2024",
        "where_description": "Divisional Secretariat Divisions in LK-23",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Religion/2024/LK-23:dsd/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Religion/2024/LK-23:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Religion/2024/LK-23:dsd/Map/Output.json](examples/outputs/Religion/2024/LK-23:dsd/Map/Output.json)

![Religion/2024/LK-23:dsd/Map](examples/outputs/Religion/2024/LK-23:dsd/Map/Image.png)

Source: [examples/outputs/Religion/2024/LK-23:dsd/Map/Image.png](examples/outputs/Religion/2024/LK-23:dsd/Map/Image.png)

#### 3.03) JSON of Population distributed by religious affiliation such as Buddhist, Hindu, Islam, and Christian (2012) for Grama Niladhari Divisions in LK-1103.

```bash
Religion/2012/LK-1103:gnd/JSON
```

```json
{
    "result": {
        "what_description": "Population distributed by religious affiliation such as Buddhist, Hindu, Islam, and Christian",
        "when_description": "2012",
        "where_description": "Grama Niladhari Divisions in LK-1103",
        "how_description": "JSON",
        "data_list": [
            {
                "region_id": "LK-1103005",
                "region_name": "Sammanthranapura",
                "region_type": "gnd",
                "history_year": "Current",
                "area_sqkm": 0.18,
                "center_lat": 6.977933,
                "center_lng": 79.878128,
                "other_names": "\u0dc3\u0db8\u0dca\u0db8\u0db1\u0dca\u0dad\u0dca\u200d\u0dbb\u0dab\u0db4\u0dd4\u0dbb",
                "num": " ",
                "country_id": "LK",
                "province_id": "LK-1",
                "district_id": "LK-11",
                ... // 1220 lines ...
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
    "cache_hit": true
}
```

Source: [examples/outputs/Religion/2012/LK-1103:gnd/JSON/Output.json](examples/outputs/Religion/2012/LK-1103:gnd/JSON/Output.json)

#### 3.04) Map of Population distributed by religious affiliation such as Buddhist, Hindu, Islam, and Christian (2012) for Grama Niladhari Divisions in LK-1127.

```bash
Religion/2012/LK-1127:gnd/Map
```

```json
{
    "result": {
        "what_description": "Population distributed by religious affiliation such as Buddhist, Hindu, Islam, and Christian",
        "when_description": "2012",
        "where_description": "Grama Niladhari Divisions in LK-1127",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Religion/2012/LK-1127:gnd/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Religion/2012/LK-1127:gnd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Religion/2012/LK-1127:gnd/Map/Output.json](examples/outputs/Religion/2012/LK-1127:gnd/Map/Output.json)

![Religion/2012/LK-1127:gnd/Map](examples/outputs/Religion/2012/LK-1127:gnd/Map/Image.png)

Source: [examples/outputs/Religion/2012/LK-1127:gnd/Map/Image.png](examples/outputs/Religion/2012/LK-1127:gnd/Map/Image.png)

#### 3.05) Map (2nd largest value) of Population distributed by religious affiliation such as Buddhist, Hindu, Islam, and Christian (2012) for Divisional Secretariat Divisions in LK-41-pre2019.

```bash
Religion/2012/LK-41:dsd/Map:2nd
```

```json
{
    "result": {
        "what_description": "Population distributed by religious affiliation such as Buddhist, Hindu, Islam, and Christian",
        "when_description": "2012",
        "where_description": "Divisional Secretariat Divisions in LK-41-pre2019",
        "how_description": "Map (2nd largest value)",
        "image_path": "/tmp/lanka_data/output/Religion/2012/LK-41:dsd/Map:2nd/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Religion/2012/LK-41:dsd/Map:2nd"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Religion/2012/LK-41:dsd/Map:2nd/Output.json](examples/outputs/Religion/2012/LK-41:dsd/Map:2nd/Output.json)

![Religion/2012/LK-41:dsd/Map:2nd](examples/outputs/Religion/2012/LK-41:dsd/Map:2nd/Image.png)

Source: [examples/outputs/Religion/2012/LK-41:dsd/Map:2nd/Image.png](examples/outputs/Religion/2012/LK-41:dsd/Map:2nd/Image.png)

#### 3.06) Map (Diversity) of Population distributed by religious affiliation such as Buddhist, Hindu, Islam, and Christian (2012) for Divisional Secretariat Divisions in LK-11-pre2019.

```bash
Religion/2012/LK-11:dsd/Map:Diversity
```

```json
{
    "result": {
        "what_description": "Population distributed by religious affiliation such as Buddhist, Hindu, Islam, and Christian",
        "when_description": "2012",
        "where_description": "Divisional Secretariat Divisions in LK-11-pre2019",
        "how_description": "Map (Diversity)",
        "image_path": "/tmp/lanka_data/output/Religion/2012/LK-11:dsd/Map:Diversity/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Religion/2012/LK-11:dsd/Map:Diversity"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Religion/2012/LK-11:dsd/Map:Diversity/Output.json](examples/outputs/Religion/2012/LK-11:dsd/Map:Diversity/Output.json)

![Religion/2012/LK-11:dsd/Map:Diversity](examples/outputs/Religion/2012/LK-11:dsd/Map:Diversity/Image.png)

Source: [examples/outputs/Religion/2012/LK-11:dsd/Map:Diversity/Image.png](examples/outputs/Religion/2012/LK-11:dsd/Map:Diversity/Image.png)

#### 3.07) Map (Buddhist) of Population distributed by religious affiliation such as Buddhist, Hindu, Islam, and Christian (2012) for Divisional Secretariat Divisions in LK-51-pre2019.

```bash
Religion/2012/LK-51:dsd/Map:Buddhist
```

```json
{
    "result": {
        "what_description": "Population distributed by religious affiliation such as Buddhist, Hindu, Islam, and Christian",
        "when_description": "2012",
        "where_description": "Divisional Secretariat Divisions in LK-51-pre2019",
        "how_description": "Map (Buddhist)",
        "image_path": "/tmp/lanka_data/output/Religion/2012/LK-51:dsd/Map:Buddhist/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Religion/2012/LK-51:dsd/Map:Buddhist"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Religion/2012/LK-51:dsd/Map:Buddhist/Output.json](examples/outputs/Religion/2012/LK-51:dsd/Map:Buddhist/Output.json)

![Religion/2012/LK-51:dsd/Map:Buddhist](examples/outputs/Religion/2012/LK-51:dsd/Map:Buddhist/Image.png)

Source: [examples/outputs/Religion/2012/LK-51:dsd/Map:Buddhist/Image.png](examples/outputs/Religion/2012/LK-51:dsd/Map:Buddhist/Image.png)

### 4) Ethnicity

#### 4.01) Map of GN-level population by ethnic group (Sinhalese, Sri Lanka Tamil, Indian Tamil, Moor, Burgher, Malay, Chetty, Bharatha, Veddha, other). (2024) for Divisional Secretariat Divisions in LK-53.

```bash
Ethnicity/2024/LK-53:dsd/Map
```

```json
{
    "result": {
        "what_description": "GN-level population by ethnic group (Sinhalese, Sri Lanka Tamil, Indian Tamil, Moor, Burgher, Malay, Chetty, Bharatha, Veddha, other).",
        "when_description": "2024",
        "where_description": "Divisional Secretariat Divisions in LK-53",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Ethnicity/2024/LK-53:dsd/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Ethnicity/2024/LK-53:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Ethnicity/2024/LK-53:dsd/Map/Output.json](examples/outputs/Ethnicity/2024/LK-53:dsd/Map/Output.json)

![Ethnicity/2024/LK-53:dsd/Map](examples/outputs/Ethnicity/2024/LK-53:dsd/Map/Image.png)

Source: [examples/outputs/Ethnicity/2024/LK-53:dsd/Map/Image.png](examples/outputs/Ethnicity/2024/LK-53:dsd/Map/Image.png)

#### 4.02) Cartogram of GN-level population by ethnic group (Sinhalese, Sri Lanka Tamil, Indian Tamil, Moor, Burgher, Malay, Chetty, Bharatha, Veddha, other). (2024) for Divisional Secretariat Divisions in LK-53.

```bash
Ethnicity/2024/LK-53:dsd/Cartogram
```

```json
{
    "result": {
        "what_description": "GN-level population by ethnic group (Sinhalese, Sri Lanka Tamil, Indian Tamil, Moor, Burgher, Malay, Chetty, Bharatha, Veddha, other).",
        "when_description": "2024",
        "where_description": "Divisional Secretariat Divisions in LK-53",
        "how_description": "Cartogram",
        "image_path": "/tmp/lanka_data/output/Ethnicity/2024/LK-53:dsd/Cartogram/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Ethnicity/2024/LK-53:dsd/Cartogram"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Ethnicity/2024/LK-53:dsd/Cartogram/Output.json](examples/outputs/Ethnicity/2024/LK-53:dsd/Cartogram/Output.json)

![Ethnicity/2024/LK-53:dsd/Cartogram](examples/outputs/Ethnicity/2024/LK-53:dsd/Cartogram/Image.png)

Source: [examples/outputs/Ethnicity/2024/LK-53:dsd/Cartogram/Image.png](examples/outputs/Ethnicity/2024/LK-53:dsd/Cartogram/Image.png)

#### 4.03) Map of GN-level population by ethnic group (Sinhalese, Sri Lanka Tamil, Indian Tamil, Moor, Burgher, Malay, Chetty, Bharatha, Veddha, other). (Latest) for Divisional Secretariat Divisions in LK-2.

```bash
Ethnicity/Latest/LK-2:dsd/Map
```

```json
{
    "result": {
        "what_description": "GN-level population by ethnic group (Sinhalese, Sri Lanka Tamil, Indian Tamil, Moor, Burgher, Malay, Chetty, Bharatha, Veddha, other).",
        "when_description": "Latest",
        "where_description": "Divisional Secretariat Divisions in LK-2",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Ethnicity/Latest/LK-2:dsd/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Ethnicity/Latest/LK-2:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Ethnicity/Latest/LK-2:dsd/Map/Output.json](examples/outputs/Ethnicity/Latest/LK-2:dsd/Map/Output.json)

![Ethnicity/Latest/LK-2:dsd/Map](examples/outputs/Ethnicity/Latest/LK-2:dsd/Map/Image.png)

Source: [examples/outputs/Ethnicity/Latest/LK-2:dsd/Map/Image.png](examples/outputs/Ethnicity/Latest/LK-2:dsd/Map/Image.png)

#### 4.04) Map of GN-level population by ethnic group (Sinhalese, Sri Lanka Tamil, Indian Tamil, Moor, Burgher, Malay, Chetty, Bharatha, Veddha, other). (2024) for Divisional Secretariat Divisions in LK-23.

```bash
Ethnicity/2024/LK-23:dsd/Map
```

```json
{
    "result": {
        "what_description": "GN-level population by ethnic group (Sinhalese, Sri Lanka Tamil, Indian Tamil, Moor, Burgher, Malay, Chetty, Bharatha, Veddha, other).",
        "when_description": "2024",
        "where_description": "Divisional Secretariat Divisions in LK-23",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Ethnicity/2024/LK-23:dsd/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Ethnicity/2024/LK-23:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Ethnicity/2024/LK-23:dsd/Map/Output.json](examples/outputs/Ethnicity/2024/LK-23:dsd/Map/Output.json)

![Ethnicity/2024/LK-23:dsd/Map](examples/outputs/Ethnicity/2024/LK-23:dsd/Map/Image.png)

Source: [examples/outputs/Ethnicity/2024/LK-23:dsd/Map/Image.png](examples/outputs/Ethnicity/2024/LK-23:dsd/Map/Image.png)

### 5) Elections

#### 5.01) JSON of Results of the 2024 Sri Lankan Parliamentary Election (2024) for LK.

```bash
Parliamentary/2024/LK/JSON
```

```json
{
    "result": {
        "what_description": "Results of the 2024 Sri Lankan Parliamentary Election",
        "when_description": "2024",
        "where_description": "LK",
        "how_description": "JSON",
        "data_list": [
            {
                "region_id": "LK",
                "region_name": "Sri Lanka",
                "region_type": "country",
                "history_year": "Current",
                "area_sqkm": 65983.58,
                "center_lat": 7.621863,
                "center_lng": 80.698448,
                "other_names": "\u0b87\u0bb2\u0b99\u0bcd\u0b95\u0bc8,\u0dc1\u0dca\u200d\u0dbb\u0dd3 \u0dbd\u0d82\u0d9a\u0dcf",
                "summary": {
                    "electors": 17140354,
                    "polled": 11815246,
                    "valid": 11148006,
                    ... // 1339 lines ...
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
            },
            "total_value": 11815246
        },
        "source": "Election Commission of Sri Lanka",
        "source_url": "https://www.elections.gov.lk",
        "cmd": "Parliamentary/2024/LK/JSON"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Parliamentary/2024/LK/JSON/Output.json](examples/outputs/Parliamentary/2024/LK/JSON/Output.json)

#### 5.02) Map of Results of the 2024 Sri Lankan Presidential Election (Latest) for Polling Divisions in LK-11.

```bash
Presidential/Latest/LK-11:pd/Map
```

```json
{
    "result": {
        "what_description": "Results of the 2024 Sri Lankan Presidential Election",
        "when_description": "Latest",
        "where_description": "Polling Divisions in LK-11",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Presidential/Latest/LK-11:pd/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Presidential/Latest/LK-11:pd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Presidential/Latest/LK-11:pd/Map/Output.json](examples/outputs/Presidential/Latest/LK-11:pd/Map/Output.json)

![Presidential/Latest/LK-11:pd/Map](examples/outputs/Presidential/Latest/LK-11:pd/Map/Image.png)

Source: [examples/outputs/Presidential/Latest/LK-11:pd/Map/Image.png](examples/outputs/Presidential/Latest/LK-11:pd/Map/Image.png)

#### 5.03) Map of Results of the 2025 Sri Lankan Local Election (2025) for Districts in LK.

```bash
Local/2025/LK:district/Map
```

```json
{
    "result": {
        "what_description": "Results of the 2025 Sri Lankan Local Election",
        "when_description": "2025",
        "where_description": "Districts in LK",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Local/2025/LK:district/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Local/2025/LK:district/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Local/2025/LK:district/Map/Output.json](examples/outputs/Local/2025/LK:district/Map/Output.json)

![Local/2025/LK:district/Map](examples/outputs/Local/2025/LK:district/Map/Image.png)

Source: [examples/outputs/Local/2025/LK:district/Map/Image.png](examples/outputs/Local/2025/LK:district/Map/Image.png)

### 6) History

#### 6.01) Map of Basic Information (2012) for Provinces in LK-pre1845.

```bash
Basic/2012/LK-pre1845:province/Map
```

```json
{
    "result": {
        "what_description": "Basic Information",
        "when_description": "2012",
        "where_description": "Provinces in LK-pre1845",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Basic/2012/LK-pre1845:province/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Basic/2012/LK-pre1845:province/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Basic/2012/LK-pre1845:province/Map/Output.json](examples/outputs/Basic/2012/LK-pre1845:province/Map/Output.json)

![Basic/2012/LK-pre1845:province/Map](examples/outputs/Basic/2012/LK-pre1845:province/Map/Image.png)

Source: [examples/outputs/Basic/2012/LK-pre1845:province/Map/Image.png](examples/outputs/Basic/2012/LK-pre1845:province/Map/Image.png)

#### 6.02) Map of Basic Information (2012) for Provinces in LK-pre1873.

```bash
Basic/2012/LK-pre1873:province/Map
```

```json
{
    "result": {
        "what_description": "Basic Information",
        "when_description": "2012",
        "where_description": "Provinces in LK-pre1873",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Basic/2012/LK-pre1873:province/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Basic/2012/LK-pre1873:province/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Basic/2012/LK-pre1873:province/Map/Output.json](examples/outputs/Basic/2012/LK-pre1873:province/Map/Output.json)

![Basic/2012/LK-pre1873:province/Map](examples/outputs/Basic/2012/LK-pre1873:province/Map/Image.png)

Source: [examples/outputs/Basic/2012/LK-pre1873:province/Map/Image.png](examples/outputs/Basic/2012/LK-pre1873:province/Map/Image.png)

#### 6.03) Map of Basic Information (2012) for Provinces in LK-pre1886.

```bash
Basic/2012/LK-pre1886:province/Map
```

```json
{
    "result": {
        "what_description": "Basic Information",
        "when_description": "2012",
        "where_description": "Provinces in LK-pre1886",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Basic/2012/LK-pre1886:province/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Basic/2012/LK-pre1886:province/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Basic/2012/LK-pre1886:province/Map/Output.json](examples/outputs/Basic/2012/LK-pre1886:province/Map/Output.json)

![Basic/2012/LK-pre1886:province/Map](examples/outputs/Basic/2012/LK-pre1886:province/Map/Image.png)

Source: [examples/outputs/Basic/2012/LK-pre1886:province/Map/Image.png](examples/outputs/Basic/2012/LK-pre1886:province/Map/Image.png)

#### 6.04) Map of Basic Information (2012) for Provinces in LK-pre1889.

```bash
Basic/2012/LK-pre1889:province/Map
```

```json
{
    "result": {
        "what_description": "Basic Information",
        "when_description": "2012",
        "where_description": "Provinces in LK-pre1889",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Basic/2012/LK-pre1889:province/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Basic/2012/LK-pre1889:province/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Basic/2012/LK-pre1889:province/Map/Output.json](examples/outputs/Basic/2012/LK-pre1889:province/Map/Output.json)

![Basic/2012/LK-pre1889:province/Map](examples/outputs/Basic/2012/LK-pre1889:province/Map/Image.png)

Source: [examples/outputs/Basic/2012/LK-pre1889:province/Map/Image.png](examples/outputs/Basic/2012/LK-pre1889:province/Map/Image.png)

#### 6.05) Map of Basic Information (2012) for Provinces in LK.

```bash
Basic/2012/LK:province/Map
```

```json
{
    "result": {
        "what_description": "Basic Information",
        "when_description": "2012",
        "where_description": "Provinces in LK",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Basic/2012/LK:province/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Basic/2012/LK:province/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Basic/2012/LK:province/Map/Output.json](examples/outputs/Basic/2012/LK:province/Map/Output.json)

![Basic/2012/LK:province/Map](examples/outputs/Basic/2012/LK:province/Map/Image.png)

Source: [examples/outputs/Basic/2012/LK:province/Map/Image.png](examples/outputs/Basic/2012/LK:province/Map/Image.png)

#### 6.06) Map of Basic Information (2012) for Districts in LK-pre1959.

```bash
Basic/2012/LK-pre1959:district/Map
```

```json
{
    "result": {
        "what_description": "Basic Information",
        "when_description": "2012",
        "where_description": "Districts in LK-pre1959",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Basic/2012/LK-pre1959:district/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Basic/2012/LK-pre1959:district/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Basic/2012/LK-pre1959:district/Map/Output.json](examples/outputs/Basic/2012/LK-pre1959:district/Map/Output.json)

![Basic/2012/LK-pre1959:district/Map](examples/outputs/Basic/2012/LK-pre1959:district/Map/Image.png)

Source: [examples/outputs/Basic/2012/LK-pre1959:district/Map/Image.png](examples/outputs/Basic/2012/LK-pre1959:district/Map/Image.png)

#### 6.07) Map of Basic Information (2012) for Districts in LK-pre1961.

```bash
Basic/2012/LK-pre1961:district/Map
```

```json
{
    "result": {
        "what_description": "Basic Information",
        "when_description": "2012",
        "where_description": "Districts in LK-pre1961",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Basic/2012/LK-pre1961:district/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Basic/2012/LK-pre1961:district/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Basic/2012/LK-pre1961:district/Map/Output.json](examples/outputs/Basic/2012/LK-pre1961:district/Map/Output.json)

![Basic/2012/LK-pre1961:district/Map](examples/outputs/Basic/2012/LK-pre1961:district/Map/Image.png)

Source: [examples/outputs/Basic/2012/LK-pre1961:district/Map/Image.png](examples/outputs/Basic/2012/LK-pre1961:district/Map/Image.png)

#### 6.08) Map of Basic Information (2012) for Districts in LK-pre1978.

```bash
Basic/2012/LK-pre1978:district/Map
```

```json
{
    "result": {
        "what_description": "Basic Information",
        "when_description": "2012",
        "where_description": "Districts in LK-pre1978",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Basic/2012/LK-pre1978:district/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Basic/2012/LK-pre1978:district/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Basic/2012/LK-pre1978:district/Map/Output.json](examples/outputs/Basic/2012/LK-pre1978:district/Map/Output.json)

![Basic/2012/LK-pre1978:district/Map](examples/outputs/Basic/2012/LK-pre1978:district/Map/Image.png)

Source: [examples/outputs/Basic/2012/LK-pre1978:district/Map/Image.png](examples/outputs/Basic/2012/LK-pre1978:district/Map/Image.png)

#### 6.09) Map of Basic Information (2012) for Districts in LK-pre1984.

```bash
Basic/2012/LK-pre1984:district/Map
```

```json
{
    "result": {
        "what_description": "Basic Information",
        "when_description": "2012",
        "where_description": "Districts in LK-pre1984",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Basic/2012/LK-pre1984:district/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Basic/2012/LK-pre1984:district/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Basic/2012/LK-pre1984:district/Map/Output.json](examples/outputs/Basic/2012/LK-pre1984:district/Map/Output.json)

![Basic/2012/LK-pre1984:district/Map](examples/outputs/Basic/2012/LK-pre1984:district/Map/Image.png)

Source: [examples/outputs/Basic/2012/LK-pre1984:district/Map/Image.png](examples/outputs/Basic/2012/LK-pre1984:district/Map/Image.png)

#### 6.10) Map of Basic Information (2012) for Districts in LK.

```bash
Basic/2012/LK:district/Map
```

```json
{
    "result": {
        "what_description": "Basic Information",
        "when_description": "2012",
        "where_description": "Districts in LK",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Basic/2012/LK:district/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Basic/2012/LK:district/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Basic/2012/LK:district/Map/Output.json](examples/outputs/Basic/2012/LK:district/Map/Output.json)

![Basic/2012/LK:district/Map](examples/outputs/Basic/2012/LK:district/Map/Image.png)

Source: [examples/outputs/Basic/2012/LK:district/Map/Image.png](examples/outputs/Basic/2012/LK:district/Map/Image.png)

#### 6.11) Map of Population distributed by ethnic group such as Sinhalese, Sri Lanka Tamil, Moor, and others (2012) for Divisional Secretariat Divisions in LK-23-pre2019.

```bash
Ethnicity/2012/LK-23-pre2019:dsd/Map
```

```json
{
    "result": {
        "what_description": "Population distributed by ethnic group such as Sinhalese, Sri Lanka Tamil, Moor, and others",
        "when_description": "2012",
        "where_description": "Divisional Secretariat Divisions in LK-23-pre2019",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Ethnicity/2012/LK-23-pre2019:dsd/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Ethnicity/2012/LK-23-pre2019:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Ethnicity/2012/LK-23-pre2019:dsd/Map/Output.json](examples/outputs/Ethnicity/2012/LK-23-pre2019:dsd/Map/Output.json)

![Ethnicity/2012/LK-23-pre2019:dsd/Map](examples/outputs/Ethnicity/2012/LK-23-pre2019:dsd/Map/Image.png)

Source: [examples/outputs/Ethnicity/2012/LK-23-pre2019:dsd/Map/Image.png](examples/outputs/Ethnicity/2012/LK-23-pre2019:dsd/Map/Image.png)

#### 6.12) Map of GN-level population by ethnic group (Sinhalese, Sri Lanka Tamil, Indian Tamil, Moor, Burgher, Malay, Chetty, Bharatha, Veddha, other). (2024) for Divisional Secretariat Divisions in LK-23-pre2019.

```bash
Ethnicity/2024/LK-23-pre2019:dsd/Map
```

```json
{
    "result": {
        "what_description": "GN-level population by ethnic group (Sinhalese, Sri Lanka Tamil, Indian Tamil, Moor, Burgher, Malay, Chetty, Bharatha, Veddha, other).",
        "when_description": "2024",
        "where_description": "Divisional Secretariat Divisions in LK-23-pre2019",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Ethnicity/2024/LK-23-pre2019:dsd/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Ethnicity/2024/LK-23-pre2019:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Ethnicity/2024/LK-23-pre2019:dsd/Map/Output.json](examples/outputs/Ethnicity/2024/LK-23-pre2019:dsd/Map/Output.json)

![Ethnicity/2024/LK-23-pre2019:dsd/Map](examples/outputs/Ethnicity/2024/LK-23-pre2019:dsd/Map/Image.png)

Source: [examples/outputs/Ethnicity/2024/LK-23-pre2019:dsd/Map/Image.png](examples/outputs/Ethnicity/2024/LK-23-pre2019:dsd/Map/Image.png)

#### 6.13) Map of GN-level population by ethnic group (Sinhalese, Sri Lanka Tamil, Indian Tamil, Moor, Burgher, Malay, Chetty, Bharatha, Veddha, other). (2024) for Divisional Secretariat Divisions in LK-23.

```bash
Ethnicity/2024/LK-23:dsd/Map
```

```json
{
    "result": {
        "what_description": "GN-level population by ethnic group (Sinhalese, Sri Lanka Tamil, Indian Tamil, Moor, Burgher, Malay, Chetty, Bharatha, Veddha, other).",
        "when_description": "2024",
        "where_description": "Divisional Secretariat Divisions in LK-23",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Ethnicity/2024/LK-23:dsd/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Ethnicity/2024/LK-23:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Ethnicity/2024/LK-23:dsd/Map/Output.json](examples/outputs/Ethnicity/2024/LK-23:dsd/Map/Output.json)

![Ethnicity/2024/LK-23:dsd/Map](examples/outputs/Ethnicity/2024/LK-23:dsd/Map/Image.png)

Source: [examples/outputs/Ethnicity/2024/LK-23:dsd/Map/Image.png](examples/outputs/Ethnicity/2024/LK-23:dsd/Map/Image.png)

#### 6.14) Map of Basic Information (2024) for Divisional Secretariat Divisions in LK.

```bash
Basic/2024/LK:dsd/Map
```

```json
{
    "result": {
        "what_description": "Basic Information",
        "when_description": "2024",
        "where_description": "Divisional Secretariat Divisions in LK",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Basic/2024/LK:dsd/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Basic/2024/LK:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Basic/2024/LK:dsd/Map/Output.json](examples/outputs/Basic/2024/LK:dsd/Map/Output.json)

![Basic/2024/LK:dsd/Map](examples/outputs/Basic/2024/LK:dsd/Map/Image.png)

Source: [examples/outputs/Basic/2024/LK:dsd/Map/Image.png](examples/outputs/Basic/2024/LK:dsd/Map/Image.png)

#### 6.15) Map of Basic Information (2024) for Grama Niladhari Divisions in LK.

```bash
Basic/2024/LK:gnd/Map
```

```json
{
    "result": {
        "what_description": "Basic Information",
        "when_description": "2024",
        "where_description": "Grama Niladhari Divisions in LK",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Basic/2024/LK:gnd/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Basic/2024/LK:gnd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Basic/2024/LK:gnd/Map/Output.json](examples/outputs/Basic/2024/LK:gnd/Map/Output.json)

![Basic/2024/LK:gnd/Map](examples/outputs/Basic/2024/LK:gnd/Map/Image.png)

Source: [examples/outputs/Basic/2024/LK:gnd/Map/Image.png](examples/outputs/Basic/2024/LK:gnd/Map/Image.png)

### 7) Other-Whats

#### 7.01) Map of GN-level population by five-year age groups from 00–04 up to 95 and above. (2024) for Districts in LK-1.

```bash
AgeGroup/2024/LK-1:district/Map
```

```json
{
    "result": {
        "what_description": "GN-level population by five-year age groups from 00\u201304 up to 95 and above.",
        "when_description": "2024",
        "where_description": "Districts in LK-1",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/AgeGroup/2024/LK-1:district/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "AgeGroup/2024/LK-1:district/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/AgeGroup/2024/LK-1:district/Map/Output.json](examples/outputs/AgeGroup/2024/LK-1:district/Map/Output.json)

![AgeGroup/2024/LK-1:district/Map](examples/outputs/AgeGroup/2024/LK-1:district/Map/Image.png)

Source: [examples/outputs/AgeGroup/2024/LK-1:district/Map/Image.png](examples/outputs/AgeGroup/2024/LK-1:district/Map/Image.png)

#### 7.02) Map of Households classified by ownership of communication items such as telephone, radio, and television (2012) for Divisional Secretariat Divisions in LK-2-pre2019.

```bash
Communication/2012/LK-2:dsd/Map
```

```json
{
    "result": {
        "what_description": "Households classified by ownership of communication items such as telephone, radio, and television",
        "when_description": "2012",
        "where_description": "Divisional Secretariat Divisions in LK-2-pre2019",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Communication/2012/LK-2:dsd/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Communication/2012/LK-2:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Communication/2012/LK-2:dsd/Map/Output.json](examples/outputs/Communication/2012/LK-2:dsd/Map/Output.json)

![Communication/2012/LK-2:dsd/Map](examples/outputs/Communication/2012/LK-2:dsd/Map/Image.png)

Source: [examples/outputs/Communication/2012/LK-2:dsd/Map/Image.png](examples/outputs/Communication/2012/LK-2:dsd/Map/Image.png)

#### 7.03) Map of Housing units classified by the decade or period in which they were constructed (2012) for Divisional Secretariat Divisions in LK-3-pre2019.

```bash
ConstructionYear/2012/LK-3:dsd/Map
```

```json
{
    "result": {
        "what_description": "Housing units classified by the decade or period in which they were constructed",
        "when_description": "2012",
        "where_description": "Divisional Secretariat Divisions in LK-3-pre2019",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/ConstructionYear/2012/LK-3:dsd/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "ConstructionYear/2012/LK-3:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/ConstructionYear/2012/LK-3:dsd/Map/Output.json](examples/outputs/ConstructionYear/2012/LK-3:dsd/Map/Output.json)

![ConstructionYear/2012/LK-3:dsd/Map](examples/outputs/ConstructionYear/2012/LK-3:dsd/Map/Image.png)

Source: [examples/outputs/ConstructionYear/2012/LK-3:dsd/Map/Image.png](examples/outputs/ConstructionYear/2012/LK-3:dsd/Map/Image.png)

#### 7.04) Map of Households by principal cooking fuel (firewood, kerosene, gas, electricity, sawdust, bio-gas, other). (2024) for Divisional Secretariat Divisions in LK-4.

```bash
Fuel/2024/LK-4:dsd/Map
```

```json
{
    "result": {
        "what_description": "Households by principal cooking fuel (firewood, kerosene, gas, electricity, sawdust, bio-gas, other).",
        "when_description": "2024",
        "where_description": "Divisional Secretariat Divisions in LK-4",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Fuel/2024/LK-4:dsd/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Fuel/2024/LK-4:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Fuel/2024/LK-4:dsd/Map/Output.json](examples/outputs/Fuel/2024/LK-4:dsd/Map/Output.json)

![Fuel/2024/LK-4:dsd/Map](examples/outputs/Fuel/2024/LK-4:dsd/Map/Image.png)

Source: [examples/outputs/Fuel/2024/LK-4:dsd/Map/Image.png](examples/outputs/Fuel/2024/LK-4:dsd/Map/Image.png)

#### 7.05) Map of Population classified by economic activity status including employed, unemployed, and economically inactive (2012) for Divisional Secretariat Divisions in LK-5-pre2019.

```bash
Economy/2012/LK-5:dsd/Map
```

```json
{
    "result": {
        "what_description": "Population classified by economic activity status including employed, unemployed, and economically inactive",
        "when_description": "2012",
        "where_description": "Divisional Secretariat Divisions in LK-5-pre2019",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Economy/2012/LK-5:dsd/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Economy/2012/LK-5:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Economy/2012/LK-5:dsd/Map/Output.json](examples/outputs/Economy/2012/LK-5:dsd/Map/Output.json)

![Economy/2012/LK-5:dsd/Map](examples/outputs/Economy/2012/LK-5:dsd/Map/Image.png)

Source: [examples/outputs/Economy/2012/LK-5:dsd/Map/Image.png](examples/outputs/Economy/2012/LK-5:dsd/Map/Image.png)

#### 7.06) Map of Population classified by the highest level of educational qualification attained (2012) for Divisional Secretariat Divisions in LK-6-pre2019.

```bash
Education/2012/LK-6:dsd/Map
```

```json
{
    "result": {
        "what_description": "Population classified by the highest level of educational qualification attained",
        "when_description": "2012",
        "where_description": "Divisional Secretariat Divisions in LK-6-pre2019",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Education/2012/LK-6:dsd/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Education/2012/LK-6:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Education/2012/LK-6:dsd/Map/Output.json](examples/outputs/Education/2012/LK-6:dsd/Map/Output.json)

![Education/2012/LK-6:dsd/Map](examples/outputs/Education/2012/LK-6:dsd/Map/Image.png)

Source: [examples/outputs/Education/2012/LK-6:dsd/Map/Image.png](examples/outputs/Education/2012/LK-6:dsd/Map/Image.png)

#### 7.07) Map of Occupied housing units by floor material (cement, terrazzo/tile, concrete, mud, wood, sand, other). (2024) for LK-2, LK-3.

```bash
Floor/2024/LK-2,LK-3/Map
```

```json
{
    "result": {
        "what_description": "Occupied housing units by floor material (cement, terrazzo/tile, concrete, mud, wood, sand, other).",
        "when_description": "2024",
        "where_description": "LK-2, LK-3",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Floor/2024/LK-2,LK-3/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Floor/2024/LK-2,LK-3/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Floor/2024/LK-2,LK-3/Map/Output.json](examples/outputs/Floor/2024/LK-2,LK-3/Map/Output.json)

![Floor/2024/LK-2,LK-3/Map](examples/outputs/Floor/2024/LK-2,LK-3/Map/Image.png)

Source: [examples/outputs/Floor/2024/LK-2,LK-3/Map/Image.png](examples/outputs/Floor/2024/LK-2,LK-3/Map/Image.png)

#### 7.08) Map of GN-level population disaggregated by sex (male, female). (2024) for Regions within 20.0 km of LK-1127025.

```bash
Gender/2024/LK-1127025@20/Map
```

```json
{
    "result": {
        "what_description": "GN-level population disaggregated by sex (male, female).",
        "when_description": "2024",
        "where_description": "Regions within 20.0 km of LK-1127025",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Gender/2024/LK-1127025@20/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Gender/2024/LK-1127025@20/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Gender/2024/LK-1127025@20/Map/Output.json](examples/outputs/Gender/2024/LK-1127025@20/Map/Output.json)

![Gender/2024/LK-1127025@20/Map](examples/outputs/Gender/2024/LK-1127025@20/Map/Image.png)

Source: [examples/outputs/Gender/2024/LK-1127025@20/Map/Image.png](examples/outputs/Gender/2024/LK-1127025@20/Map/Image.png)

#### 7.09) Map of Households by principal lighting source (grid electricity, kerosene lamp, solar, bio-gas, generator, other). (2024) for Divisional Secretariat Divisions in LK-61.

```bash
Lighting/2024/LK-61:dsd/Map
```

```json
{
    "result": {
        "what_description": "Households by principal lighting source (grid electricity, kerosene lamp, solar, bio-gas, generator, other).",
        "when_description": "2024",
        "where_description": "Divisional Secretariat Divisions in LK-61",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Lighting/2024/LK-61:dsd/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Lighting/2024/LK-61:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Lighting/2024/LK-61:dsd/Map/Output.json](examples/outputs/Lighting/2024/LK-61:dsd/Map/Output.json)

![Lighting/2024/LK-61:dsd/Map](examples/outputs/Lighting/2024/LK-61:dsd/Map/Image.png)

Source: [examples/outputs/Lighting/2024/LK-61:dsd/Map/Image.png](examples/outputs/Lighting/2024/LK-61:dsd/Map/Image.png)

#### 7.10) Map of Population classified by marital status such as never married, married, widowed, and divorced (2012) for LK-3, LK-9, LK-8.

```bash
Marital/2012/LK-3,LK-9,LK-8/Map
```

```json
{
    "result": {
        "what_description": "Population classified by marital status such as never married, married, widowed, and divorced",
        "when_description": "2012",
        "where_description": "LK-3, LK-9, LK-8",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Marital/2012/LK-3,LK-9,LK-8/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Marital/2012/LK-3,LK-9,LK-8/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Marital/2012/LK-3,LK-9,LK-8/Map/Output.json](examples/outputs/Marital/2012/LK-3,LK-9,LK-8/Map/Output.json)

![Marital/2012/LK-3,LK-9,LK-8/Map](examples/outputs/Marital/2012/LK-3,LK-9,LK-8/Map/Image.png)

Source: [examples/outputs/Marital/2012/LK-3,LK-9,LK-8/Map/Image.png](examples/outputs/Marital/2012/LK-3,LK-9,LK-8/Map/Image.png)

#### 7.11) Map of Housing units classified by occupancy status, distinguishing occupied from vacant units (2012) for Grama Niladhari Divisions in LK-1127.

```bash
Occupancy/2012/LK-1127:gnd/Map
```

```json
{
    "result": {
        "what_description": "Housing units classified by occupancy status, distinguishing occupied from vacant units",
        "when_description": "2012",
        "where_description": "Grama Niladhari Divisions in LK-1127",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Occupancy/2012/LK-1127:gnd/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Occupancy/2012/LK-1127:gnd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Occupancy/2012/LK-1127:gnd/Map/Output.json](examples/outputs/Occupancy/2012/LK-1127:gnd/Map/Output.json)

![Occupancy/2012/LK-1127:gnd/Map](examples/outputs/Occupancy/2012/LK-1127:gnd/Map/Image.png)

Source: [examples/outputs/Occupancy/2012/LK-1127:gnd/Map/Image.png](examples/outputs/Occupancy/2012/LK-1127:gnd/Map/Image.png)

#### 7.12) Map of Households classified by the ownership status of their dwelling, such as owned or rented (2012) for Divisional Secretariat Divisions in LK-53-pre2019.

```bash
Ownership/2012/LK-53:dsd/Map
```

```json
{
    "result": {
        "what_description": "Households classified by the ownership status of their dwelling, such as owned or rented",
        "when_description": "2012",
        "where_description": "Divisional Secretariat Divisions in LK-53-pre2019",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Ownership/2012/LK-53:dsd/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Ownership/2012/LK-53:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Ownership/2012/LK-53:dsd/Map/Output.json](examples/outputs/Ownership/2012/LK-53:dsd/Map/Output.json)

![Ownership/2012/LK-53:dsd/Map](examples/outputs/Ownership/2012/LK-53:dsd/Map/Image.png)

Source: [examples/outputs/Ownership/2012/LK-53:dsd/Map/Image.png](examples/outputs/Ownership/2012/LK-53:dsd/Map/Image.png)

#### 7.13) Map of Households classified by the type of living quarters such as housing units, collective living quarters, and makeshift housing (2012) for LK-4 to LK-6.

```bash
Quarters/2012/LK-4...LK-6/Map
```

```json
{
    "result": {
        "what_description": "Households classified by the type of living quarters such as housing units, collective living quarters, and makeshift housing",
        "when_description": "2012",
        "where_description": "LK-4 to LK-6",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Quarters/2012/LK-4...LK-6/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Quarters/2012/LK-4...LK-6/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Quarters/2012/LK-4...LK-6/Map/Output.json](examples/outputs/Quarters/2012/LK-4...LK-6/Map/Output.json)

![Quarters/2012/LK-4...LK-6/Map](examples/outputs/Quarters/2012/LK-4...LK-6/Map/Image.png)

Source: [examples/outputs/Quarters/2012/LK-4...LK-6/Map/Image.png](examples/outputs/Quarters/2012/LK-4...LK-6/Map/Image.png)

#### 7.14) Map of Population classified by their relationship to the head of the household (2012) for Divisional Secretariat Divisions in LK-23-pre2019.

```bash
RelationshipToHead/2012/LK-23:dsd/Map
```

```json
{
    "result": {
        "what_description": "Population classified by their relationship to the head of the household",
        "when_description": "2012",
        "where_description": "Divisional Secretariat Divisions in LK-23-pre2019",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/RelationshipToHead/2012/LK-23:dsd/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "RelationshipToHead/2012/LK-23:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/RelationshipToHead/2012/LK-23:dsd/Map/Output.json](examples/outputs/RelationshipToHead/2012/LK-23:dsd/Map/Output.json)

![RelationshipToHead/2012/LK-23:dsd/Map](examples/outputs/RelationshipToHead/2012/LK-23:dsd/Map/Image.png)

Source: [examples/outputs/RelationshipToHead/2012/LK-23:dsd/Map/Image.png](examples/outputs/RelationshipToHead/2012/LK-23:dsd/Map/Image.png)

#### 7.15) Map of Occupied housing units by roof material (tile, asbestos, concrete, metal sheet, cadjan/straw, other). (2024) for Divisional Secretariat Divisions in LK-9.

```bash
Roof/2024/LK-9:dsd/Map
```

```json
{
    "result": {
        "what_description": "Occupied housing units by roof material (tile, asbestos, concrete, metal sheet, cadjan/straw, other).",
        "when_description": "2024",
        "where_description": "Divisional Secretariat Divisions in LK-9",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Roof/2024/LK-9:dsd/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Roof/2024/LK-9:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Roof/2024/LK-9:dsd/Map/Output.json](examples/outputs/Roof/2024/LK-9:dsd/Map/Output.json)

![Roof/2024/LK-9:dsd/Map](examples/outputs/Roof/2024/LK-9:dsd/Map/Image.png)

Source: [examples/outputs/Roof/2024/LK-9:dsd/Map/Image.png](examples/outputs/Roof/2024/LK-9:dsd/Map/Image.png)

#### 7.16) Map of Households classified by the number of rooms in the dwelling (2012) for Polling Divisions in LK-2.

```bash
Rooms/2012/LK-2:pd/Map
```

```json
{
    "result": {
        "what_description": "Households classified by the number of rooms in the dwelling",
        "when_description": "2012",
        "where_description": "Polling Divisions in LK-2",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Rooms/2012/LK-2:pd/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Rooms/2012/LK-2:pd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Rooms/2012/LK-2:pd/Map/Output.json](examples/outputs/Rooms/2012/LK-2:pd/Map/Output.json)

![Rooms/2012/LK-2:pd/Map](examples/outputs/Rooms/2012/LK-2:pd/Map/Image.png)

Source: [examples/outputs/Rooms/2012/LK-2:pd/Map/Image.png](examples/outputs/Rooms/2012/LK-2:pd/Map/Image.png)

#### 7.17) Map of Occupied housing units by building structure type (single/attached houses by storey count, other). (2024) for Districts in LK-3.

```bash
Structure/2024/LK-3:district/Map
```

```json
{
    "result": {
        "what_description": "Occupied housing units by building structure type (single/attached houses by storey count, other).",
        "when_description": "2024",
        "where_description": "Districts in LK-3",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Structure/2024/LK-3:district/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Structure/2024/LK-3:district/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Structure/2024/LK-3:district/Map/Output.json](examples/outputs/Structure/2024/LK-3:district/Map/Output.json)

![Structure/2024/LK-3:district/Map](examples/outputs/Structure/2024/LK-3:district/Map/Image.png)

Source: [examples/outputs/Structure/2024/LK-3:district/Map/Image.png](examples/outputs/Structure/2024/LK-3:district/Map/Image.png)

#### 7.18) Map of Households by toilet facility type and exclusivity (within unit/premises, shared, common/public, none). (2024) for Divisional Secretariat Divisions in LK-53.

```bash
Toilet/2024/LK-53:dsd/Map
```

```json
{
    "result": {
        "what_description": "Households by toilet facility type and exclusivity (within unit/premises, shared, common/public, none).",
        "when_description": "2024",
        "where_description": "Divisional Secretariat Divisions in LK-53",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Toilet/2024/LK-53:dsd/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Toilet/2024/LK-53:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Toilet/2024/LK-53:dsd/Map/Output.json](examples/outputs/Toilet/2024/LK-53:dsd/Map/Output.json)

![Toilet/2024/LK-53:dsd/Map](examples/outputs/Toilet/2024/LK-53:dsd/Map/Image.png)

Source: [examples/outputs/Toilet/2024/LK-53:dsd/Map/Image.png](examples/outputs/Toilet/2024/LK-53:dsd/Map/Image.png)

#### 7.19) Map of Occupied housing units by wall construction material (bricks, cement block, cabook, mud, cadjan, sheets, other). (2024) for Divisional Secretariat Divisions in LK-8.

```bash
Walls/2024/LK-8:dsd/Map
```

```json
{
    "result": {
        "what_description": "Occupied housing units by wall construction material (bricks, cement block, cabook, mud, cadjan, sheets, other).",
        "when_description": "2024",
        "where_description": "Divisional Secretariat Divisions in LK-8",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Walls/2024/LK-8:dsd/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Walls/2024/LK-8:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Walls/2024/LK-8:dsd/Map/Output.json](examples/outputs/Walls/2024/LK-8:dsd/Map/Output.json)

![Walls/2024/LK-8:dsd/Map](examples/outputs/Walls/2024/LK-8:dsd/Map/Image.png)

Source: [examples/outputs/Walls/2024/LK-8:dsd/Map/Image.png](examples/outputs/Walls/2024/LK-8:dsd/Map/Image.png)

#### 7.20) Map of Households classified by the method used for solid waste disposal (2012) for Divisional Secretariat Divisions in LK-1-pre2019.

```bash
Waste/2012/LK-1:dsd/Map
```

```json
{
    "result": {
        "what_description": "Households classified by the method used for solid waste disposal",
        "when_description": "2012",
        "where_description": "Divisional Secretariat Divisions in LK-1-pre2019",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Waste/2012/LK-1:dsd/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Waste/2012/LK-1:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Waste/2012/LK-1:dsd/Map/Output.json](examples/outputs/Waste/2012/LK-1:dsd/Map/Output.json)

![Waste/2012/LK-1:dsd/Map](examples/outputs/Waste/2012/LK-1:dsd/Map/Image.png)

Source: [examples/outputs/Waste/2012/LK-1:dsd/Map/Image.png](examples/outputs/Waste/2012/LK-1:dsd/Map/Image.png)

#### 7.21) Map of Households by principal drinking water source (wells, tube well, pipe-borne, bottled, RO filter, bowser, other). (2024) for Divisional Secretariat Divisions in LK-4.

```bash
Water/2024/LK-4:dsd/Map
```

```json
{
    "result": {
        "what_description": "Households by principal drinking water source (wells, tube well, pipe-borne, bottled, RO filter, bowser, other).",
        "when_description": "2024",
        "where_description": "Divisional Secretariat Divisions in LK-4",
        "how_description": "Map",
        "image_path": "/tmp/lanka_data/output/Water/2024/LK-4:dsd/Map/Image.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/",
        "cmd": "Water/2024/LK-4:dsd/Map"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

Source: [examples/outputs/Water/2024/LK-4:dsd/Map/Output.json](examples/outputs/Water/2024/LK-4:dsd/Map/Output.json)

![Water/2024/LK-4:dsd/Map](examples/outputs/Water/2024/LK-4:dsd/Map/Image.png)

Source: [examples/outputs/Water/2024/LK-4:dsd/Map/Image.png](examples/outputs/Water/2024/LK-4:dsd/Map/Image.png)

![Maintainer](https://img.shields.io/badge/maintainer-nuuuwan-red)
![MadeWith](https://img.shields.io/badge/made_with-python-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
