# Lanka Data

*A collection of utility functions for interacting with Public Data about Sri Lanka.*

## Install

```bash
pip install lanka_data-nuuuwan
```

## Example Usage

```python
from lanka_data import DataSet
dataset = Dataset.find('GDP', limit=1)[0]
print(dataset)
```

```bash
Dataset(
    source_id='adb', 
    category='BALANCE OF PAYMENTSw calendar year ($ million)', 
    sub_category='BALANCE International investment position Other investment Debit Balance of Payments (Pct. of GDP at current market prices) Changes in inventories', 
    scale='', 
    unit='% of GDP at current market prices', 
    frequency_name='Annual', 
    i_subject=-1, 
    footnotes={}, 
    summary_statistics={'n': 22, 'min_t': 2000, 'max_t': 2021, 'min_value': 33.03425, 'max_value': 14.144962364124})
)
```

```python
print(dataset.data)
```

```bash
{'2000': 33.03425, '2001': 30.01958, '2002': 27.4765, '2003': 27.18631, '2004': 27.8641, '2005': 26.00561, '2006': 24.3, '2007': 23.6, '2008': 19.91753, '2009': 16.84135, '2010': 14.710765089295, '2011': 15.584218779657, '2012': 13.873549567794, '2013': 13.498926569782, '2014': 13.486329136145, '2015': 12.387104249851, '2016': 11.713979295222, '2017': 12.037371716173, '2018': 12.582439411597, '2019': 13.413486947557, '2020': 11.772149264932, '2021': 14.144962364124}
```


```python
print(dataset.xy)
```

```bash
(
    ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021'], 
    [33.03425, 30.01958, 27.4765, 27.18631, 27.8641, 26.00561, 24.3, 23.6, 19.91753, 16.84135, 14.710765089295, 15.584218779657, 13.873549567794, 13.498926569782, 13.486329136145, 12.387104249851, 11.713979295222, 12.037371716173, 12.582439411597, 13.413486947557, 11.772149264932, 14.144962364124]
)
```

```python
x, y = dataset.xy

import matplotlib.pyplot as plt

plt.title(first_dataset.sub_category)
plt.plot(x, y)
plt.show()
```

![README.example.png](README.example.png)

