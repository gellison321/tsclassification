from .sample_methods import compiler
from tsshapelet import metrics, Shapelet, query, score, utils
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from concurrent.futures import ThreadPoolExecutor
import numpy as np, os, inspect, multiprocessing
from sklearn.pipeline import Pipeline
import pickle


def get_params(func):
    return inspect.signature(func).parameters.keys()


class ShapeletClassifier():

    def __init__ (self, metric = 'dtw', classification_window = 300, w = 0.9):
        self.window = classification_window
        self.metric = metric
        self.w = w
        self.shapelets = {}


    def extract_shapelets(self, X, y, shapelet_method, *args, **kwargs):
        ''' Implements the shapelet extraction process using tsshapelet library. '''

        X, y = np.array(X), np.array(y)

        kwargs['parallel_cores'] = os.cpu_count() - 1

        for label in np.unique(y):

            index = np.where(y == label)[0]

            if len(index) <= self.window:
                pass

            shape = Shapelet(X[index])
            shape.quantile_normalization()

            if 'smoothing_period' in kwargs:
                shape.smooth(kwargs['smoothing_period'])
            
            if 'thres' not in kwargs:
                kwargs['thres'] = 0.8
            
            if shapelet_method == 'random':
                shape.random_shapelet(*args, **{kw : arg for kw, arg in kwargs.items() if kw in get_params(shape.random_shapelet)})

                # Random shapelets are not necessarily phase synchronized
                shapelet = Shapelet(utils['reinterpolate'](shape.shapelet, self.window))
                shapelet.phase_sync(thres = kwargs['thres'])
                self.shapelets[label] = utils['reinterpolate'](shapelet.series, self.window)

            elif shapelet_method == 'exhaustive':
                shape.exhaustive_shapelet(**{kw : arg for kw, arg in kwargs.items() if kw in get_params(shape.exhaustive_shapelet)})

                # Exhaustive shapelets are not necessarily phase synchronized
                shapelet = Shapelet(utils['reinterpolate'](shape.shapelet, self.window))
                shapelet.phase_sync(thres = kwargs['thres'])
                self.shapelets[label] = utils['reinterpolate'](shapelet.series, self.window)

            elif shapelet_method == 'barycenter':
                shape.barycenter_shapelet(**{kw : arg for kw, arg in kwargs.items() if kw in get_params(shape.barycenter_shapelet)})

                # Barycenter shapelets are already phase synchronized
                self.shapelets[label] = utils['reinterpolate'](shape.shapelet, self.window)

            else:
                raise ValueError('Only random, exhaustive and barycenter methods are supported')


    def score_x(self, args):
        ''' Processes the time series and returns the score of the time series against the shapelets. '''

        x, thres, smoothing_period = args

        shape = Shapelet(x)
        shape.phase_sync(thres = thres)
        shape.series = shape.series[:self.window]
        shape.quantile_normalization()
        shape.smooth(smoothing_period)
        x = shape.series

        scores = []
        for shapelet in self.shapelets.values():
            scores.append(metrics[self.metric](x, shapelet, self.w))
        return scores

    
    def fit_nn(self,**kwargs):
        ''' Scores the compiled data, and fits a neural network classifier scored data.'''

        if 'thres' not in kwargs:
            kwargs['thres'] = 0.8
            
        if 'smoothing_period' not in kwargs:
            kwargs['smoothing_period'] = 1

        # Scoring the data with multiprocessing
        with multiprocessing.Pool(os.cpu_count() - 1) as p:
            train_X = list(p.map(self.score_x, [(x, kwargs['thres'], kwargs['smoothing_period']) for x in self.train_X]))

        self.train_X = np.array(train_X) 

        # Infinite values can arise from the scoring process
        finite = np.isfinite(self.train_X)

        if finite.size > 0:
            max_non_inf = np.max(self.train_X[finite])
            
        else:
            raise ValueError('No finite values found in the scores. Check the shapelet extraction method and the input data')
        
        self.train_X[np.isinf(self.train_X)] = np.nan 

        self.pipeline = Pipeline([('imputer', SimpleImputer(missing_values = np.nan, strategy = 'constant', fill_value = max_non_inf)),
                                  ('scaler', StandardScaler())
                                ])

        self.pipeline.fit(self.train_X)
        self.train_X = self.pipeline.transform(self.train_X)
        
        # fitting the neural network
        self.clf = MLPClassifier(**{kw : arg for kw, arg in kwargs.items() if kw in get_params(MLPClassifier) and kw != 'verbose'})
        self.clf.fit(self.train_X, self.train_y)


    def fit(self, X, y, shapelet_method = 'random', sample_method = 'bootstrapped', **kwargs):
        '''
        Fits the model to the provided dataset by extracting shapelets, compiling the data for training, 
        and then training a neural network classifier based on the extracted shapelets and compiled data.

        Parameters
            X (array-like): A 1D time series dataset.
            y (array-like): The target values (class labels) for the samples in X.
            shapelet_method (str): The method used to extract shapelets from the dataset. Supported
                                   methods include 'random', 'exhaustive', and others depending on the implementation. 
                                   Defaults to 'random'.
            compiler_method (str): The method used to compile the dataset for training the neural network 
                                   after shapelet extraction. Supported methods include 'bootstrapped', 
                                   'direct', and others depending on the implementation. 
                                   Defaults to 'bootstrapped'.
            **kwargs: Additional keyword arguments that are passed to the shapelet extraction method and 
                      neural network training procedure. This includes parameters like 'verbose' to control 
                      verbosity, 'window' to set the window size for shapelet extraction, and other 
                      method-specific parameters.

        Returns
            None: This method does not return any value. It internally updates the model instance with the 
                  extracted shapelets, compiled training data, and a trained neural network classifier.

        Note
            The choice of `shapelet_method` and `compiler_method`, along with specific parameters passed 
            through `**kwargs`, significantly affects the model's performance and training time. 
            It's essential to choose these parameters carefully based on the dataset characteristics and the
            analysis goals.
        '''

        verbose, kwargs['window'] = kwargs['verbose'], self.window*2

        X, y = np.array(X), np.array(y)

        if verbose:
            print('Extracting shapelets')

        self.extract_shapelets(X, y, shapelet_method, **kwargs)

        if verbose:
            print('Compiling data for training')

        self.train_X, self.train_y = compiler(X, y, sample_method, **{kw : arg for kw, arg in kwargs.items() if kw in get_params(compiler)})
        
        if verbose:
            print('Training the neural network')

        self.fit_nn(**kwargs)

        self.predict(self.train_X)# warm up the model


    def predict(self, X, **kwargs):
        '''
        Predicts the class labels for the provided input dataset X using the trained model. 
        This method improves inference time by utilizing multithreading for scoring, followed by imputation 
        and scaling of the scores before making predictions with the trained MLPClassifier.

        Parameters
            X (Sequence[array-like]): The input features dataset for which to predict class labels.
                                      A sequence of time series data arrays to be passed for inference.

        Returns
            array-like: Predicted class labels for each sample in X.

        Note
            Before calling this method, ensure that the model is properly trained using the `fit` method. 
            The performance and accuracy of the predictions depend on the quality of the training process, 
            including shapelet extraction, data compilation, and neural network training.
        '''
        if 'thres' not in kwargs:
            kwargs['thres'] = 0.8

        if 'smoothing_period' not in kwargs:
            kwargs['smoothing_period'] = 1

        # The score_x method contains the preprocessing steps for the input data
        with ThreadPoolExecutor() as executor:
            scores = list(executor.map(self.score_x,[(x, kwargs['thres'], kwargs['smoothing_period']) for x in X]))

        # imputing infinite values with the maximum non-infinite value
        scores = np.array(scores)
        scores[np.isinf(scores)] = np.nan
        scores = self.pipeline.transform(scores)

        # calling the MLPClassifier predict method
        return self.clf.predict(scores)
    
    def save(self, path):
        ''' Saves the model to a binary file. '''

        if '.' in path:
            new_path = path.split('.')[0]
            end = path.split('.')[-1]

            if end == 'pkl':
                with open(path, 'wb') as f:
                    pickle.dump(self, f)

            else:
                new_path += '.pkl'
                with open(new_path, 'wb') as f:
                    pickle.dump(self, f)
        else:
            path += '.pkl'
            with open(path, 'wb') as f:
                pickle.dump(self, f)