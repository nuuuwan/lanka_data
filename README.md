# Lanka Data

This repo implements a simple interface
to query data about Sri Lanka.

## Data Sources

- [Census of Population and Housing 2012](https://www.statistics.gov.lk/Resource/en/Population/CPH_2011/CPH_2012_5Per_Rpt.pdf)
- [Census of Population and Housing 2024](https://www.statistics.gov.lk/Population/StaticalInformation/CPH2024)
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

## Example Commands (`<cmd>`)

### 01. `LK`

```json
{
    "result": {
        "regions": [
            {
                "id": "LK",
                "name": "Sri Lanka",
                "area_sqkm": 65983.58,
                "center_lat": 7.621863,
                "center_lng": 80.698448
            }
        ],
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

### 02. `LK-99`

```json
{
    "error": "Region ID not found: LK-99"
}
```

### 03. `LK-1:district`

```json
{
    "result": {
        "regions": [
            {
                "id": "LK-11",
                "name": "Colombo",
                "area_sqkm": 688.17,
                "center_lat": 6.869822,
                "center_lng": 80.018487,
                "province_id": "LK-1",
                "ed_id": "EC-01",
                "pd_id": null
            },
            {
                "id": "LK-12",
                "name": "Gampaha",
                "area_sqkm": 1385.23,
                "center_lat": 7.123406,
                "center_lng": 80.018206,
                "province_id": "LK-1",
                "ed_id": "EC-02",
                "pd_id": null
            },
            {
                "id": "LK-13",
                "name": "Kalutara",
                "area_sqkm": 1646.99,
                "center_lat": 6.577185,
                "center_lng": 80.127744,
                "province_id": "LK-1",
                "ed_id": "EC-03",
                "pd_id": null
            }
        ],
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

### 04. `LK/Map`

```json
{
    "result": {
        "image_path": "/tmp/lanka_data/cache/lk-map.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

![LK/Map](images/readme/lk-map.png)

### 05. `LK-11/Map`

```json
{
    "result": {
        "image_path": "/tmp/lanka_data/cache/lk-11-map.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

![LK-11/Map](images/readme/lk-11-map.png)

### 06. `LK-1:district/Map`

```json
{
    "result": {
        "image_path": "/tmp/lanka_data/cache/lk-1:district-map.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

![LK-1:district/Map](images/readme/lk-1:district-map.png)

### 07. `LK:pd/Map`

```json
{
    "result": {
        "image_path": "/tmp/lanka_data/cache/lk:pd-map.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

![LK:pd/Map](images/readme/lk:pd-map.png)

### 08. `LK-1:pd/Map`

```json
{
    "result": {
        "image_path": "/tmp/lanka_data/cache/lk-1:pd-map.png",
        "source": "Department of Census and Statistics, Sri Lanka",
        "source_url": "https://www.statistics.gov.lk/"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

![LK-1:pd/Map](images/readme/lk-1:pd-map.png)

### 09. `LK-11/Religion/2012`

```json
{
    "result": {
        "data_list": [
            {
                "region_id": "LK-11",
                "region_name": "Colombo",
                "values": {
                    "buddhist": 1632125,
                    "islam": 274067,
                    "hindu": 186303,
                    "roman_catholic": 162260,
                    "other_christian": 66947,
                    "other": 2262
                },
                "total_value": 2323964,
                "pct_values": {
                    "buddhist": 0.7023,
                    "islam": 0.1179,
                    "hindu": 0.0802,
                    "roman_catholic": 0.0698,
                    "other_christian": 0.0288,
                    "other": 0.001
                }
            }
        ],
        "source": "Census of Population and Housing 2012",
        "source_url": "https://www.statistics.gov.lk/Resource/en/Population/CPH_2011/CPH_2012_5Per_Rpt.pdf"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

### 10. `LK-9:district/Ethnicity/2012`

```json
{
    "result": {
        "data_list": [
            {
                "region_id": "LK-91",
                "region_name": "Ratnapura",
                "values": {
                    "sinhalese": 947811,
                    "ind_tamil": 62124,
                    "sl_tamil": 54437,
                    "sl_moor": 22346,
                    "other_eth": 549,
                    "burgher": 405,
                    "malay": 288,
                    "sl_chetty": 35,
                    "bharatha": 12
                },
                "total_value": 1088007,
                "pct_values": {
                    "sinhalese": 0.8711,
                    ... // 24 lines ...
                "total_value": 840648,
                "pct_values": {
                    "sinhalese": 0.8545,
                    "sl_moor": 0.0714,
                    "ind_tamil": 0.052,
                    "sl_tamil": 0.0212,
                    "burgher": 0.0003,
                    "other_eth": 0.0002,
                    "malay": 0.0002,
                    "sl_chetty": 0.0001,
                    "bharatha": 0.0
                }
            }
        ],
        "source": "Census of Population and Housing 2012",
        "source_url": "https://www.statistics.gov.lk/Resource/en/Population/CPH_2011/CPH_2012_5Per_Rpt.pdf"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

### 11. `LK/PresidentialElection/2024`

```json
{
    "result": {
        "data_list": [
            {
                "region_id": "LK",
                "region_name": "Sri Lanka",
                "summary": {
                    "electors": 17140354,
                    "polled": 13619916,
                    "valid": 13319616,
                    "rejected": 300300,
                    "p_turnout": 0.7946,
                    "p_valid": 0.978,
                    "p_rejected": 0.022
                },
                "by_party": {
                    "NPP": 5634915,
                    "SJB": 4363035,
                    "IND16": 2299767,
                    "SLPP": 342781,
                    ... // 63 lines ...
                    "SEP": 0.0003,
                    "NIF": 0.0003,
                    "IND15": 0.0003,
                    "NDF": 0.0003,
                    "IND6": 0.0003,
                    "UNFF": 0.0002,
                    "IND7": 0.0002,
                    "ELPP": 0.0002,
                    "IND8": 0.0002,
                    "NSU": 0.0001,
                    "SLLP": 0.0001
                }
            }
        ],
        "source": "Election Commission of Sri Lanka",
        "source_url": "https://www.elections.gov.lk"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

### 12. `LK/ParliamentaryElection/2024`

```json
{
    "result": {
        "data_list": [
            {
                "region_id": "LK",
                "region_name": "Sri Lanka",
                "summary": {
                    "electors": 17140354,
                    "polled": 11815246,
                    "valid": 11148006,
                    "rejected": 667240,
                    "p_turnout": 0.6893,
                    "p_valid": 0.9435,
                    "p_rejected": 0.0565
                },
                "by_party": {
                    "NPP": 6863186,
                    "SJB": 1968716,
                    "NDF": 500835,
                    "SLPP": 350429,
                    ... // 649 lines ...
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
            }
        ],
        "source": "Election Commission of Sri Lanka",
        "source_url": "https://www.elections.gov.lk"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

### 13. `LK-1103/Religion/2024`

```json
{
    "result": {
        "data_list": [
            {
                "region_id": "LK-1103",
                "region_name": "Colombo",
                "values": {
                    "islam": 125890,
                    "hindu": 71811,
                    "buddhist": 47726,
                    "roman_catholic": 36117,
                    "other_christian": 10381,
                    "other": 164
                },
                "total_value": 292089,
                "pct_values": {
                    "islam": 0.431,
                    "hindu": 0.2459,
                    "buddhist": 0.1634,
                    "roman_catholic": 0.1237,
                    "other_christian": 0.0355,
                    "other": 0.0006
                }
            }
        ],
        "source": "Census of Population and Housing 2024",
        "source_url": "https://www.statistics.gov.lk/Population/StaticalInformation/CPH2024"
    },
    "query_time_ms": 0,
    "cache_hit": true
}
```

![Maintainer](https://img.shields.io/badge/maintainer-nuuuwan-red)
![MadeWith](https://img.shields.io/badge/made_with-python-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
