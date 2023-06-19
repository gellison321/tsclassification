from tsshapelet import metrics, Shapelet
from tsclassification.sample_methods import compilers
import numpy as np
from sklearn import svm
from sklearn.neural_network import MLPClassifier

class ShapeletClassifier():

    def __init__ (self, metric = 'dtw', window = 300, smoothing_period = 1, thres = 0.8, min_dist = 50):
        self.thres = thres
        self.min_dist = min_dist
        self.smoothing_period = smoothing_period
        self.window = window
        self.metric = metric

    # Implements Shapelet object to extract time series shapelets
    def extract_shapelets(self, extraction = 'peak', barycenter = 'interpolated',  
                          min_dist = 60, thres = 0.8, max_dist = 120, qty = 1, 
                          sample = 100, comparison = 'candidates'):
        self.shapelets = {}
        for label in np.unique(self.y):
            index = np.where(self.y == label)[0]
            if len(index) <= 1:
                pass
            x = self.X[index]
            shape = Shapelet(x, metric = self.metric)
            shape.smooth(self.smoothing_period)
            shape.quantile_normalization()
            shape.candidate_extraction(extraction = extraction, min_dist = min_dist, 
                                    thres = thres, max_dist = max_dist, sample = sample)
            print(label, 'has', len(shape.candidates), 'candidates')
            shape.shapelet_selection(barycenter = barycenter, qty = qty, comparison = comparison)
            shape.reinterpolate_shapelets(self.window)
            self.shapelets[label] = shape.shapelets[0]

    # Helper which scores the shapelet for each label against a given series
    def score_shapelets(self, x):
        x = Shapelet(x)
        x.quantile_normalization()
        x.smooth(self.smoothing_period)
        x.phase_sync(thres = self.thres, min_dist = self.min_dist)
        x = x.series[:self.window]
        scores = []
        for label in self.shapelets:
            scores.append(metrics[self.metric](self.shapelets[label], x))
        return np.array(scores)
    
    # Impelements helper sample methods to compile and score training data for classifier
    def compile_classifier_training(self, sample = 100, method = 'sample_slide', step = 30):
        self.train_X, self.train_y = compilers[method](self.X, self.y, self.window, 
                                                       step = step, sample = sample)
        train_X = []
        for x in self.train_X:
            train_X.append(np.array(self.score_shapelets(x)))
        return train_X, self.train_y

    # Fits a support vector machine to the scored data
    def fit_SVM(self, X, y, decision_function_shape = 'ovo'):
        self.clf = svm.SVC(decision_function_shape = decision_function_shape).fit(X, y)

    # Fits a multi layer perceptron to the scored data
    def fit_MLP(self, X, y, max_iter = 500, hidden_layers_sizes = (100,), function = 'relu'):
        self.clf = MLPClassifier(random_state=1, max_iter = max_iter, hidden_layer_sizes = hidden_layers_sizes,
                                  activation = function).fit(X, y)

    # Control flow for fitting classifier to data
    def fit(self, X, y, classifier = 'MLP', max_iter = 500, hidden_layers_sizes = (100,),function = 'relu', 
            decision_function_shape = 'ovo', extraction = 'peak', barycenter = 'interpolated', sample = 100,
            sample_method = 'sample_slide', sample_step = 30):
        if type(X) != np.array:
            self.X = np.array(X)
        if type(y) != np.array:
            self.y = np.array(y)
        self.extract_shapelets(extraction = extraction, barycenter = barycenter)
        classifier_train_X, classifier_train_y = self.compile_classifier_training(sample = 100, method = 'sample_slide', step = 30)
        if classifier == 'MLP':
            self.fit_MLP(classifier_train_X, classifier_train_y, max_iter = max_iter, 
                         hidden_layers_sizes = hidden_layers_sizes, function = function)
        elif classifier == 'SVM':
            self.fit_SVM(classifier_train_X, classifier_train_y, 
                         decision_function_shape = decision_function_shape)

    # Given a window of incoming activity data, returns the predicted label
    def predict(self, X):
        scores = []
        for x in X:
            x = Shapelet(x)
            x.quantile_normalization()
            x.smooth(self.smoothing_period)
            x.phase_sync(thres = self.thres, min_dist = self.min_dist)
            scores.append(self.score_shapelets(x.series[:self.window]))
        return self.clf.predict(scores)