# <p align="center"> tsclassification
<div align="center">
<p>
<img alt="GitHub" src="https://img.shields.io/github/license/gellison321/tsclassification">
</p>
</div>

## <p align="center"> A shapelet classifier for time series data.

### <p align="center">[Install From pip](https://pypi.org/project/tsclassification/)
```
$ pip install tsclassification
```

### Link to paper here

#### Dependencies
- numpy
- tsshapelet
- sklearn

##  <p align="center"> IMPLEMENTATION
### <p align="center"> [Full Implementation](https://github.com/gellison321/tsclassification/blob/main/implementation.ipynb)

```python
import pandas as pd
from tsclassification import ShapeletClassifier

# Loading Example Data 
with open('./data/sample_data/001_labeled.csv','rb') as file:
    df = pd.read_csv(file)

X = df['waist_vm']
y = df['activity']

# Shapelet classification with MLP classifier
clf = ShapeletClassifier(metric = 'dtw', # for comparing incoming data to shapelets
                         window = 300, # size of window to be classified
                         smoothing_period = 1, # for incoming data, and shapelet extraction
                         thres = 0.8, # for all peak extraction (phase-sync & shapelet extraction)
                         min_dist = 50 # for all peak extraction (phase-sync & shapelet extraction)
                         )

clf.fit(X, y, 
        classifier = 'MLP', max_iter = 500, hidden_layers_sizes = (100,), function = 'relu', # MLP parameters
        extraction = 'peak', barycenter = 'interpolated', # shapelet extraction parameters
        sample = 100, sample_method = 'sample_slide', sample_step = 30 # sampling parameters for compiling MLP training data
        )

# Score each shapelet against a given time series and return a multi-label output
km.predict([[]])

```
### <p align="center"> [Full Shapelet Class Shapelet Extraction](https://github.com/gellison321/tsshapelet/blob/main/implementation.ipynb)



