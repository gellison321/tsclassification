# <p align="center"> tsclassification
<div align="center">
<p>
<img alt="GitHub" src="https://img.shields.io/github/license/gellison321/tsclassification">
</p>
</div>

## <p align="center"> A shapelet classifier for time series data.


## [Link to Paper on IEEE](https://ieeexplore.ieee.org/document/10459802)
## References
> "Real-Time Human Activity Classification Using Gait Cycle Averaging and Biometric Heuristics"<br/>
> Grant Ellison and M.P. Markovic and Delaram Yazdansepas<br/>
> 22nd International Conference on Machine Learning and Applications (ICMLA)<br/>

``` bibtex
@inproceedings{gellison23,
  title={Real-Time Human Activity Classification Using Gait Cycle Averaging and Biometric Heuristics},
  author={Grant Ellison and M.P. Markovic and Delaram Yazdansepas},
  journal={22nd International Conference on Machine Learning and Applications (ICMLA)},
  year={2023},
  publisher={IEEE},
  DOI = {10.1109/ICMLA58977.2023.00056}
}
```

## Install From pip
```
$ pip install tsclassification
```

## Quick Documentation

```python
clf = ShapeletClassifier(metric = 'dtw', # 'dtw' or 'euclidean'
                         classification_window = 300, # the discrete time steps to classify
                         w = 0.5 # warping constraint for the dtw measure
                         )
```

```python

'''
Fits the model to the provided dataset by extracting shapelets, compiling the data for training, 
and then training a neural network classifier based on the extracted shapelets and compiled data.

Parameters

        - X (array-like): A 1D time series dataset.

        - y (array-like): The target values (class labels) for the samples in X.

        - shapelet_method (str): The method used to extract shapelets from the dataset. 

                                - 'barycenter' : Performs peak-analysis to extract subsequences.
                                        Averages subsequences to a barycenter shapelet.

                                - 'random' : Chooses a random qty of subsequences. Chooses the one
                                        with the minimum cumulative distance to each other subsequence.

                                - 'exhaustive : Chooses subsequences using a stepped, sliding
                                        window. Chooses the one with the minimum cumulative distance to
                                        each other subsequence.

        - compiler_method (str): The method used to compile the dataset for training the neural network after 
                                        shapelet extraction. 

                                - 'bootstrapped' : slides a stepped window,  randomly placed, for
                                        set number of samples. Repeats until the total qty of samples have been extracted. 

                                - 'slide' : A simple, stepped, sliding window.

                                - 'random' : Randomly positioned, fixed windows are extracted until 
                                        the total qty of samples have been extracted.

        - **kwargs: Additional keyword arguments that are passed to the shapelet extraction method and 
                neural network training procedure.

Note
        The choice of `shapelet_method` and `compiler_method`, along with specific parameters passed 
        through `**kwargs`, significantly affects the model's performance and training time. 
        It's essential to choose these parameters carefully based on the dataset characteristics and the
        analysis goals.
'''

clf.fit(X, # 1d time series array
        y, # 1d list of classes
        
        shapelet_method = 'barycenter',
        sampling_method = 'bootstrapped',

        qty = 100, # number of samples to extract for training the MLP
        window = 100, # window size of 'slide' method
        step = 30, # the step size for the sliding window technique
        min_dist = 70, # the minimum peak distance for barycenter shapelet method
        max_dist = 120, # the maximum peak distance for barycenter shapelet method
        verbose = False, 
        
        max_iter = 2000, # **kwargs pass through to SKLearn's MLP class
        hidden_layer_sizes = (10,)
        )
```

```python
clf.predict([[]]) # takes in a list of time series arrays and returns a list of predicted classes
```

```python
clf.save('path/to/file') # saves the whole classifier object to a pkl file
```