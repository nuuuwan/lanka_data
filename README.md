# Lanka Data

This repo implements a simple interface to query data about Sri Lanka.

## Data Sources

- [Census of Population and Housing 2012](https://www.statistics.gov.lk/Resource/en/Population/CPH_2011/CPH_2012_5Per_Rpt.pdf)
- [Census of Population and Housing 2024](https://www.statistics.gov.lk/Population/StaticalInformation/CPH2024)
- [Election Commission of Sri lanka](https://www.elections.gov.lk)
- [Lanka Data](https://github.com/nuuuwan/lanka_data/blob/main/README.md)
- [Survey Department of Sri Lanka](https://survey.gov.lk/)

## Usage

### Run Code

```python
from lanka_data import Command


db = Command("<cmd>")
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
    "command_str": "Help",
    "result": {
        "what_to_whens": [
            "TODO"
        ],
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
    "sources": [
        {
            "name": "Lanka Data",
            "url": "https://github.com/nuuuwan/lanka_data/blob/main/README.md"
        }
    ],
    "query_time_ms": 0
}
```

Source: [examples/outputs/Help/Output.json](examples/outputs/Help/Output.json)

### 2) Selection

#### 2.01) Empty/2024/LK:province/Map

```bash
Empty/2024/LK:province/Map
```

```json
{
    "command_str": "Empty/2024/LK:province/Map",
    "result": {
        "image_path": "/tmp/lanka_data/output/Empty/2024/LK:province/Map/Image.png"
    },
    "sources": [
        {
            "name": "Survey Department of Sri Lanka",
            "url": "https://survey.gov.lk/"
        }
    ],
    "query_time_ms": 0
}
```

Source: [examples/outputs/Empty/2024/LK:province/Map/Output.json](examples/outputs/Empty/2024/LK:province/Map/Output.json)

![Empty/2024/LK:province/Map](examples/outputs/Empty/2024/LK:province/Map/Image.png)

Source: [examples/outputs/Empty/2024/LK:province/Map/Image.png](examples/outputs/Empty/2024/LK:province/Map/Image.png)

#### 2.02) Empty/2024/LK-1,LK-2,LK-3,LK-9,LK-8/Map

```bash
Empty/2024/LK-1,LK-2,LK-3,LK-9,LK-8/Map
```

```json
{
    "command_str": "Empty/2024/LK-1,LK-2,LK-3,LK-9,LK-8/Map",
    "result": {
        "image_path": "/tmp/lanka_data/output/Empty/2024/LK-1,LK-2,LK-3,LK-9,LK-8/Map/Image.png"
    },
    "sources": [
        {
            "name": "Survey Department of Sri Lanka",
            "url": "https://survey.gov.lk/"
        }
    ],
    "query_time_ms": 0
}
```

Source: [examples/outputs/Empty/2024/LK-1,LK-2,LK-3,LK-9,LK-8/Map/Output.json](examples/outputs/Empty/2024/LK-1,LK-2,LK-3,LK-9,LK-8/Map/Output.json)

![Empty/2024/LK-1,LK-2,LK-3,LK-9,LK-8/Map](examples/outputs/Empty/2024/LK-1,LK-2,LK-3,LK-9,LK-8/Map/Image.png)

Source: [examples/outputs/Empty/2024/LK-1,LK-2,LK-3,LK-9,LK-8/Map/Image.png](examples/outputs/Empty/2024/LK-1,LK-2,LK-3,LK-9,LK-8/Map/Image.png)

#### 2.03) Empty/2024/LK-5...LK-8/Map

```bash
Empty/2024/LK-5...LK-8/Map
```

```json
{
    "command_str": "Empty/2024/LK-5...LK-8/Map",
    "result": {
        "image_path": "/tmp/lanka_data/output/Empty/2024/LK-5...LK-8/Map/Image.png"
    },
    "sources": [
        {
            "name": "Survey Department of Sri Lanka",
            "url": "https://survey.gov.lk/"
        }
    ],
    "query_time_ms": 0
}
```

Source: [examples/outputs/Empty/2024/LK-5...LK-8/Map/Output.json](examples/outputs/Empty/2024/LK-5...LK-8/Map/Output.json)

![Empty/2024/LK-5...LK-8/Map](examples/outputs/Empty/2024/LK-5...LK-8/Map/Image.png)

Source: [examples/outputs/Empty/2024/LK-5...LK-8/Map/Image.png](examples/outputs/Empty/2024/LK-5...LK-8/Map/Image.png)

#### 2.04) Empty/2024/LK-1127025@20/Map

```bash
Empty/2024/LK-1127025@20/Map
```

```json
{
    "command_str": "Empty/2024/LK-1127025@20/Map",
    "result": {
        "image_path": "/tmp/lanka_data/output/Empty/2024/LK-1127025@20/Map/Image.png"
    },
    "sources": [
        {
            "name": "Survey Department of Sri Lanka",
            "url": "https://survey.gov.lk/"
        }
    ],
    "query_time_ms": 0
}
```

Source: [examples/outputs/Empty/2024/LK-1127025@20/Map/Output.json](examples/outputs/Empty/2024/LK-1127025@20/Map/Output.json)

![Empty/2024/LK-1127025@20/Map](examples/outputs/Empty/2024/LK-1127025@20/Map/Image.png)

Source: [examples/outputs/Empty/2024/LK-1127025@20/Map/Image.png](examples/outputs/Empty/2024/LK-1127025@20/Map/Image.png)

### 3) Religion

#### 3.01) Religion/2012-2024/LK:district/Map

```bash
Religion/2012-2024/LK:district/Map
```

```json
{
    "command_str": "Religion/2012-2024/LK:district/Map",
    "result": {
        "image_path": "/tmp/lanka_data/output/Religion/2012-2024/LK:district/Map/Image.png"
    },
    "sources": [
        {
            "name": "Census of Population and Housing 2012",
            "url": "https://www.statistics.gov.lk/Resource/en/Population/CPH_2011/CPH_2012_5Per_Rpt.pdf"
        },
        {
            "name": "Census of Population and Housing 2024",
            "url": "https://www.statistics.gov.lk/Population/StaticalInformation/CPH2024"
        }
    ],
    "query_time_ms": 0
}
```

Source: [examples/outputs/Religion/2012-2024/LK:district/Map/Output.json](examples/outputs/Religion/2012-2024/LK:district/Map/Output.json)

![Religion/2012-2024/LK:district/Map](examples/outputs/Religion/2012-2024/LK:district/Map/Image.png)

Source: [examples/outputs/Religion/2012-2024/LK:district/Map/Image.png](examples/outputs/Religion/2012-2024/LK:district/Map/Image.png)

#### 3.02) Religion/2012-2024/LK:district/Cartogram

```bash
Religion/2012-2024/LK:district/Cartogram
```

```json
{
    "command_str": "Religion/2012-2024/LK:district/Cartogram",
    "result": {
        "image_path": "/tmp/lanka_data/output/Religion/2012-2024/LK:district/Cartogram/Image.png"
    },
    "sources": [
        {
            "name": "Census of Population and Housing 2012",
            "url": "https://www.statistics.gov.lk/Resource/en/Population/CPH_2011/CPH_2012_5Per_Rpt.pdf"
        },
        {
            "name": "Census of Population and Housing 2024",
            "url": "https://www.statistics.gov.lk/Population/StaticalInformation/CPH2024"
        }
    ],
    "query_time_ms": 0
}
```

Source: [examples/outputs/Religion/2012-2024/LK:district/Cartogram/Output.json](examples/outputs/Religion/2012-2024/LK:district/Cartogram/Output.json)

![Religion/2012-2024/LK:district/Cartogram](examples/outputs/Religion/2012-2024/LK:district/Cartogram/Image.png)

Source: [examples/outputs/Religion/2012-2024/LK:district/Cartogram/Image.png](examples/outputs/Religion/2012-2024/LK:district/Cartogram/Image.png)

#### 3.03) Religion/2012-2024/LK:district/Map:DiversityPew

```bash
Religion/2012-2024/LK:district/Map:DiversityPew
```

```json
{
    "command_str": "Religion/2012-2024/LK:district/Map:DiversityPew",
    "result": {
        "image_path": "/tmp/lanka_data/output/Religion/2012-2024/LK:district/Map:DiversityPew/Image.png"
    },
    "sources": [
        {
            "name": "Census of Population and Housing 2012",
            "url": "https://www.statistics.gov.lk/Resource/en/Population/CPH_2011/CPH_2012_5Per_Rpt.pdf"
        },
        {
            "name": "Census of Population and Housing 2024",
            "url": "https://www.statistics.gov.lk/Population/StaticalInformation/CPH2024"
        }
    ],
    "query_time_ms": 0
}
```

Source: [examples/outputs/Religion/2012-2024/LK:district/Map:DiversityPew/Output.json](examples/outputs/Religion/2012-2024/LK:district/Map:DiversityPew/Output.json)

![Religion/2012-2024/LK:district/Map:DiversityPew](examples/outputs/Religion/2012-2024/LK:district/Map:DiversityPew/Image.png)

Source: [examples/outputs/Religion/2012-2024/LK:district/Map:DiversityPew/Image.png](examples/outputs/Religion/2012-2024/LK:district/Map:DiversityPew/Image.png)

#### 3.04) Religion/2012-2024/LK:district/Map:Segregation

```bash
Religion/2012-2024/LK:district/Map:Segregation
```

```json
{
    "command_str": "Religion/2012-2024/LK:district/Map:Segregation",
    "result": {
        "image_path": "/tmp/lanka_data/output/Religion/2012-2024/LK:district/Map:Segregation/Image.png"
    },
    "sources": [
        {
            "name": "Census of Population and Housing 2012",
            "url": "https://www.statistics.gov.lk/Resource/en/Population/CPH_2011/CPH_2012_5Per_Rpt.pdf"
        },
        {
            "name": "Census of Population and Housing 2024",
            "url": "https://www.statistics.gov.lk/Population/StaticalInformation/CPH2024"
        }
    ],
    "query_time_ms": 0
}
```

Source: [examples/outputs/Religion/2012-2024/LK:district/Map:Segregation/Output.json](examples/outputs/Religion/2012-2024/LK:district/Map:Segregation/Output.json)

![Religion/2012-2024/LK:district/Map:Segregation](examples/outputs/Religion/2012-2024/LK:district/Map:Segregation/Image.png)

Source: [examples/outputs/Religion/2012-2024/LK:district/Map:Segregation/Image.png](examples/outputs/Religion/2012-2024/LK:district/Map:Segregation/Image.png)

### 4) Elections

#### 4.01) Parliamentary/2024/LK/JSON

```bash
Parliamentary/2024/LK/JSON
```

```json
{
    "command_str": "Parliamentary/2024/LK/JSON",
    "result": [
        {
            "region_id": "LK",
            "region_name": "Sri Lanka",
            "center_lat": 7.621863,
            "center_lng": 80.698448,
            "current_ids": [
                "LK"
            ],
            "values": {
                "NPP": 6863186,
                "SJB": 1968716,
                "NDF": 500835,
                "SLPP": 350429,
                "ITAK": 257813,
                "SB": 178006,
                "SLMC": 87038,
                "UDV": 83488,
                ... // 648 lines ...
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
    "sources": [
        {
            "name": "Election Commission of Sri lanka",
            "url": "https://www.elections.gov.lk"
        }
    ],
    "query_time_ms": 0
}
```

Source: [examples/outputs/Parliamentary/2024/LK/JSON/Output.json](examples/outputs/Parliamentary/2024/LK/JSON/Output.json)

#### 4.02) Presidential/2015/LK-11:pd/Map

```bash
Presidential/2015/LK-11:pd/Map
```

```json
{
    "command_str": "Presidential/2015/LK-11:pd/Map",
    "result": {
        "image_path": "/tmp/lanka_data/output/Presidential/2015/LK-11:pd/Map/Image.png"
    },
    "sources": [
        {
            "name": "Election Commission of Sri lanka",
            "url": "https://www.elections.gov.lk"
        }
    ],
    "query_time_ms": 0
}
```

Source: [examples/outputs/Presidential/2015/LK-11:pd/Map/Output.json](examples/outputs/Presidential/2015/LK-11:pd/Map/Output.json)

![Presidential/2015/LK-11:pd/Map](examples/outputs/Presidential/2015/LK-11:pd/Map/Image.png)

Source: [examples/outputs/Presidential/2015/LK-11:pd/Map/Image.png](examples/outputs/Presidential/2015/LK-11:pd/Map/Image.png)

#### 4.03) Local/2025/LK:district/Map

```bash
Local/2025/LK:district/Map
```

```json
{
    "command_str": "Local/2025/LK:district/Map",
    "result": {
        "image_path": "/tmp/lanka_data/output/Local/2025/LK:district/Map/Image.png"
    },
    "sources": [
        {
            "name": "Election Commission of Sri lanka",
            "url": "https://www.elections.gov.lk"
        }
    ],
    "query_time_ms": 0
}
```

Source: [examples/outputs/Local/2025/LK:district/Map/Output.json](examples/outputs/Local/2025/LK:district/Map/Output.json)

![Local/2025/LK:district/Map](examples/outputs/Local/2025/LK:district/Map/Image.png)

Source: [examples/outputs/Local/2025/LK:district/Map/Image.png](examples/outputs/Local/2025/LK:district/Map/Image.png)

### 5) History

#### 5.01) Empty/2012/LK-pre1845:province/Map

```bash
Empty/2012/LK-pre1845:province/Map
```

```json
{
    "command_str": "Empty/2012/LK-pre1845:province/Map",
    "result": {
        "image_path": "/tmp/lanka_data/output/Empty/2012/LK-pre1845:province/Map/Image.png"
    },
    "sources": [
        {
            "name": "Survey Department of Sri Lanka",
            "url": "https://survey.gov.lk/"
        }
    ],
    "query_time_ms": 0
}
```

Source: [examples/outputs/Empty/2012/LK-pre1845:province/Map/Output.json](examples/outputs/Empty/2012/LK-pre1845:province/Map/Output.json)

![Empty/2012/LK-pre1845:province/Map](examples/outputs/Empty/2012/LK-pre1845:province/Map/Image.png)

Source: [examples/outputs/Empty/2012/LK-pre1845:province/Map/Image.png](examples/outputs/Empty/2012/LK-pre1845:province/Map/Image.png)

#### 5.02) Empty/2012/LK-pre1873:province/Map

```bash
Empty/2012/LK-pre1873:province/Map
```

```json
{
    "command_str": "Empty/2012/LK-pre1873:province/Map",
    "result": {
        "image_path": "/tmp/lanka_data/output/Empty/2012/LK-pre1873:province/Map/Image.png"
    },
    "sources": [
        {
            "name": "Survey Department of Sri Lanka",
            "url": "https://survey.gov.lk/"
        }
    ],
    "query_time_ms": 0
}
```

Source: [examples/outputs/Empty/2012/LK-pre1873:province/Map/Output.json](examples/outputs/Empty/2012/LK-pre1873:province/Map/Output.json)

![Empty/2012/LK-pre1873:province/Map](examples/outputs/Empty/2012/LK-pre1873:province/Map/Image.png)

Source: [examples/outputs/Empty/2012/LK-pre1873:province/Map/Image.png](examples/outputs/Empty/2012/LK-pre1873:province/Map/Image.png)

#### 5.03) Empty/2012/LK-pre1886:province/Map

```bash
Empty/2012/LK-pre1886:province/Map
```

```json
{
    "command_str": "Empty/2012/LK-pre1886:province/Map",
    "result": {
        "image_path": "/tmp/lanka_data/output/Empty/2012/LK-pre1886:province/Map/Image.png"
    },
    "sources": [
        {
            "name": "Survey Department of Sri Lanka",
            "url": "https://survey.gov.lk/"
        }
    ],
    "query_time_ms": 0
}
```

Source: [examples/outputs/Empty/2012/LK-pre1886:province/Map/Output.json](examples/outputs/Empty/2012/LK-pre1886:province/Map/Output.json)

![Empty/2012/LK-pre1886:province/Map](examples/outputs/Empty/2012/LK-pre1886:province/Map/Image.png)

Source: [examples/outputs/Empty/2012/LK-pre1886:province/Map/Image.png](examples/outputs/Empty/2012/LK-pre1886:province/Map/Image.png)

#### 5.04) Empty/2012/LK-pre1889:province/Map

```bash
Empty/2012/LK-pre1889:province/Map
```

```json
{
    "command_str": "Empty/2012/LK-pre1889:province/Map",
    "result": {
        "image_path": "/tmp/lanka_data/output/Empty/2012/LK-pre1889:province/Map/Image.png"
    },
    "sources": [
        {
            "name": "Survey Department of Sri Lanka",
            "url": "https://survey.gov.lk/"
        }
    ],
    "query_time_ms": 0
}
```

Source: [examples/outputs/Empty/2012/LK-pre1889:province/Map/Output.json](examples/outputs/Empty/2012/LK-pre1889:province/Map/Output.json)

![Empty/2012/LK-pre1889:province/Map](examples/outputs/Empty/2012/LK-pre1889:province/Map/Image.png)

Source: [examples/outputs/Empty/2012/LK-pre1889:province/Map/Image.png](examples/outputs/Empty/2012/LK-pre1889:province/Map/Image.png)

#### 5.05) Empty/2012/LK:province/Map

```bash
Empty/2012/LK:province/Map
```

```json
{
    "command_str": "Empty/2012/LK:province/Map",
    "result": {
        "image_path": "/tmp/lanka_data/output/Empty/2012/LK:province/Map/Image.png"
    },
    "sources": [
        {
            "name": "Survey Department of Sri Lanka",
            "url": "https://survey.gov.lk/"
        }
    ],
    "query_time_ms": 0
}
```

Source: [examples/outputs/Empty/2012/LK:province/Map/Output.json](examples/outputs/Empty/2012/LK:province/Map/Output.json)

![Empty/2012/LK:province/Map](examples/outputs/Empty/2012/LK:province/Map/Image.png)

Source: [examples/outputs/Empty/2012/LK:province/Map/Image.png](examples/outputs/Empty/2012/LK:province/Map/Image.png)

#### 5.06) Empty/2012/LK-pre1959:district/Map

```bash
Empty/2012/LK-pre1959:district/Map
```

```json
{
    "command_str": "Empty/2012/LK-pre1959:district/Map",
    "result": {
        "image_path": "/tmp/lanka_data/output/Empty/2012/LK-pre1959:district/Map/Image.png"
    },
    "sources": [
        {
            "name": "Survey Department of Sri Lanka",
            "url": "https://survey.gov.lk/"
        }
    ],
    "query_time_ms": 0
}
```

Source: [examples/outputs/Empty/2012/LK-pre1959:district/Map/Output.json](examples/outputs/Empty/2012/LK-pre1959:district/Map/Output.json)

![Empty/2012/LK-pre1959:district/Map](examples/outputs/Empty/2012/LK-pre1959:district/Map/Image.png)

Source: [examples/outputs/Empty/2012/LK-pre1959:district/Map/Image.png](examples/outputs/Empty/2012/LK-pre1959:district/Map/Image.png)

#### 5.07) Empty/2012/LK-pre1961:district/Map

```bash
Empty/2012/LK-pre1961:district/Map
```

```json
{
    "command_str": "Empty/2012/LK-pre1961:district/Map",
    "result": {
        "image_path": "/tmp/lanka_data/output/Empty/2012/LK-pre1961:district/Map/Image.png"
    },
    "sources": [
        {
            "name": "Survey Department of Sri Lanka",
            "url": "https://survey.gov.lk/"
        }
    ],
    "query_time_ms": 0
}
```

Source: [examples/outputs/Empty/2012/LK-pre1961:district/Map/Output.json](examples/outputs/Empty/2012/LK-pre1961:district/Map/Output.json)

![Empty/2012/LK-pre1961:district/Map](examples/outputs/Empty/2012/LK-pre1961:district/Map/Image.png)

Source: [examples/outputs/Empty/2012/LK-pre1961:district/Map/Image.png](examples/outputs/Empty/2012/LK-pre1961:district/Map/Image.png)

#### 5.08) Empty/2012/LK-pre1978:district/Map

```bash
Empty/2012/LK-pre1978:district/Map
```

```json
{
    "command_str": "Empty/2012/LK-pre1978:district/Map",
    "result": {
        "image_path": "/tmp/lanka_data/output/Empty/2012/LK-pre1978:district/Map/Image.png"
    },
    "sources": [
        {
            "name": "Survey Department of Sri Lanka",
            "url": "https://survey.gov.lk/"
        }
    ],
    "query_time_ms": 0
}
```

Source: [examples/outputs/Empty/2012/LK-pre1978:district/Map/Output.json](examples/outputs/Empty/2012/LK-pre1978:district/Map/Output.json)

![Empty/2012/LK-pre1978:district/Map](examples/outputs/Empty/2012/LK-pre1978:district/Map/Image.png)

Source: [examples/outputs/Empty/2012/LK-pre1978:district/Map/Image.png](examples/outputs/Empty/2012/LK-pre1978:district/Map/Image.png)

#### 5.09) Empty/2012/LK-pre1984:district/Map

```bash
Empty/2012/LK-pre1984:district/Map
```

```json
{
    "command_str": "Empty/2012/LK-pre1984:district/Map",
    "result": {
        "image_path": "/tmp/lanka_data/output/Empty/2012/LK-pre1984:district/Map/Image.png"
    },
    "sources": [
        {
            "name": "Survey Department of Sri Lanka",
            "url": "https://survey.gov.lk/"
        }
    ],
    "query_time_ms": 0
}
```

Source: [examples/outputs/Empty/2012/LK-pre1984:district/Map/Output.json](examples/outputs/Empty/2012/LK-pre1984:district/Map/Output.json)

![Empty/2012/LK-pre1984:district/Map](examples/outputs/Empty/2012/LK-pre1984:district/Map/Image.png)

Source: [examples/outputs/Empty/2012/LK-pre1984:district/Map/Image.png](examples/outputs/Empty/2012/LK-pre1984:district/Map/Image.png)

#### 5.10) Empty/2012/LK:district/Map

```bash
Empty/2012/LK:district/Map
```

```json
{
    "command_str": "Empty/2012/LK:district/Map",
    "result": {
        "image_path": "/tmp/lanka_data/output/Empty/2012/LK:district/Map/Image.png"
    },
    "sources": [
        {
            "name": "Survey Department of Sri Lanka",
            "url": "https://survey.gov.lk/"
        }
    ],
    "query_time_ms": 0
}
```

Source: [examples/outputs/Empty/2012/LK:district/Map/Output.json](examples/outputs/Empty/2012/LK:district/Map/Output.json)

![Empty/2012/LK:district/Map](examples/outputs/Empty/2012/LK:district/Map/Image.png)

Source: [examples/outputs/Empty/2012/LK:district/Map/Image.png](examples/outputs/Empty/2012/LK:district/Map/Image.png)

#### 5.11) Ethnicity/2012/LK-23-pre2019:dsd/Map

```bash
Ethnicity/2012/LK-23-pre2019:dsd/Map
```

```json
{
    "command_str": "Ethnicity/2012/LK-23-pre2019:dsd/Map",
    "result": {
        "image_path": "/tmp/lanka_data/output/Ethnicity/2012/LK-23-pre2019:dsd/Map/Image.png"
    },
    "sources": [
        {
            "name": "Census of Population and Housing 2012",
            "url": "https://www.statistics.gov.lk/Resource/en/Population/CPH_2011/CPH_2012_5Per_Rpt.pdf"
        }
    ],
    "query_time_ms": 0
}
```

Source: [examples/outputs/Ethnicity/2012/LK-23-pre2019:dsd/Map/Output.json](examples/outputs/Ethnicity/2012/LK-23-pre2019:dsd/Map/Output.json)

![Ethnicity/2012/LK-23-pre2019:dsd/Map](examples/outputs/Ethnicity/2012/LK-23-pre2019:dsd/Map/Image.png)

Source: [examples/outputs/Ethnicity/2012/LK-23-pre2019:dsd/Map/Image.png](examples/outputs/Ethnicity/2012/LK-23-pre2019:dsd/Map/Image.png)

#### 5.12) Ethnicity/2024/LK-23-pre2019:dsd/Map

```bash
Ethnicity/2024/LK-23-pre2019:dsd/Map
```

```json
{
    "command_str": "Ethnicity/2024/LK-23-pre2019:dsd/Map",
    "result": {
        "image_path": "/tmp/lanka_data/output/Ethnicity/2024/LK-23-pre2019:dsd/Map/Image.png"
    },
    "sources": [
        {
            "name": "Census of Population and Housing 2024",
            "url": "https://www.statistics.gov.lk/Population/StaticalInformation/CPH2024"
        }
    ],
    "query_time_ms": 0
}
```

Source: [examples/outputs/Ethnicity/2024/LK-23-pre2019:dsd/Map/Output.json](examples/outputs/Ethnicity/2024/LK-23-pre2019:dsd/Map/Output.json)

![Ethnicity/2024/LK-23-pre2019:dsd/Map](examples/outputs/Ethnicity/2024/LK-23-pre2019:dsd/Map/Image.png)

Source: [examples/outputs/Ethnicity/2024/LK-23-pre2019:dsd/Map/Image.png](examples/outputs/Ethnicity/2024/LK-23-pre2019:dsd/Map/Image.png)

#### 5.13) Ethnicity/2024/LK-23:dsd/Map

```bash
Ethnicity/2024/LK-23:dsd/Map
```

```json
{
    "command_str": "Ethnicity/2024/LK-23:dsd/Map",
    "result": {
        "image_path": "/tmp/lanka_data/output/Ethnicity/2024/LK-23:dsd/Map/Image.png"
    },
    "sources": [
        {
            "name": "Census of Population and Housing 2024",
            "url": "https://www.statistics.gov.lk/Population/StaticalInformation/CPH2024"
        }
    ],
    "query_time_ms": 0
}
```

Source: [examples/outputs/Ethnicity/2024/LK-23:dsd/Map/Output.json](examples/outputs/Ethnicity/2024/LK-23:dsd/Map/Output.json)

![Ethnicity/2024/LK-23:dsd/Map](examples/outputs/Ethnicity/2024/LK-23:dsd/Map/Image.png)

Source: [examples/outputs/Ethnicity/2024/LK-23:dsd/Map/Image.png](examples/outputs/Ethnicity/2024/LK-23:dsd/Map/Image.png)

#### 5.14) Empty/2024/LK:dsd/Map

```bash
Empty/2024/LK:dsd/Map
```

```json
{
    "command_str": "Empty/2024/LK:dsd/Map",
    "result": {
        "image_path": "/tmp/lanka_data/output/Empty/2024/LK:dsd/Map/Image.png"
    },
    "sources": [
        {
            "name": "Survey Department of Sri Lanka",
            "url": "https://survey.gov.lk/"
        }
    ],
    "query_time_ms": 0
}
```

Source: [examples/outputs/Empty/2024/LK:dsd/Map/Output.json](examples/outputs/Empty/2024/LK:dsd/Map/Output.json)

![Empty/2024/LK:dsd/Map](examples/outputs/Empty/2024/LK:dsd/Map/Image.png)

Source: [examples/outputs/Empty/2024/LK:dsd/Map/Image.png](examples/outputs/Empty/2024/LK:dsd/Map/Image.png)

#### 5.15) Empty/2024/LK:gnd/Map

```bash
Empty/2024/LK:gnd/Map
```

```json
{
    "command_str": "Empty/2024/LK:gnd/Map",
    "result": {
        "image_path": "/tmp/lanka_data/output/Empty/2024/LK:gnd/Map/Image.png"
    },
    "sources": [
        {
            "name": "Survey Department of Sri Lanka",
            "url": "https://survey.gov.lk/"
        }
    ],
    "query_time_ms": 0
}
```

Source: [examples/outputs/Empty/2024/LK:gnd/Map/Output.json](examples/outputs/Empty/2024/LK:gnd/Map/Output.json)

![Empty/2024/LK:gnd/Map](examples/outputs/Empty/2024/LK:gnd/Map/Image.png)

Source: [examples/outputs/Empty/2024/LK:gnd/Map/Image.png](examples/outputs/Empty/2024/LK:gnd/Map/Image.png)

![Maintainer](https://img.shields.io/badge/maintainer-nuuuwan-red)
![MadeWith](https://img.shields.io/badge/made_with-python-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
