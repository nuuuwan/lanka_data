# Lanka Data

*A collection of utility functions for interacting with Public Data about Sri Lanka.*

## Install

```bash
pip install lanka-data-nuuuwan
```

## Example Usage

### Find (Search for) Dataset

```python
from lanka_data import DataSet
dataset = Dataset.find('GDP', limit=1)[0]
print(dataset)
```

```bash
Dataset(
    source_id='adb', 
    category='BALANCE OF PAYMENTSw calendar year ($ million)', 
    sub_category='BALANCE International investment position Other...', 
    scale='', 
    unit='% of GDP at current market prices', 
    frequency_name='Annual', 
    i_subject=-1, 
    footnotes={}, 
    summary_statistics={
        'n': 22, 
        'min_t': 2000, 'max_t': 2021, 
        'min_value': 33.03425, 'max_value': 14.144962364124
    }
)
```

### View Data

```python
print(dataset.data)
```

```bash
{'2000': 33.03425, '2001': 30.01958, '2002': 27.4765, '2003': 27.18631, ...}
```

### Plot Graph

```python
x, y = dataset.xy
print((x, y))
```

```bash
(
    ['2000', '2001', '2002', '2003', '2004', '2005', ...], 
    [33.03425, 30.01958, 27.4765, 27.18631, 27.8641, 26.00561, ...]
)
```

```python
import matplotlib.pyplot as plt

plt.title(first_dataset.sub_category)
plt.plot(x, y)
plt.show()
```

![README.example.png](README.example.png)

## Version History

* 1.0.0 - Basic Dataset with find capabilities
