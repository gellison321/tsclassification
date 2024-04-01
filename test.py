from tsclassification import *
from tsclassification.sample_methods import compiler
from tsshapelet import utils
import numpy as np
from scipy.stats import mode
import itertools as iter
import time
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt

def load_csv(filename, delimiter=',', skip_header=0, dtype = object):
    try:
        data = np.genfromtxt(filename, delimiter=delimiter, skip_header=skip_header, dtype=dtype)
        return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def main():

    # Loading data
    data = load_csv('data/sample_data/001_labeled.csv', delimiter=',', skip_header=0, dtype = object)
    X = np.array(load_csv('data/sample_data/001_labeled.csv')[1:,1], dtype=float)
    y = np.array(load_csv('data/sample_data/001_labeled.csv')[1:,-1], dtype=object)

    # Test the compiler

    data, labels = compiler(X, y, method = 'slide', window = 100, step = 10, qty = 500)

    assert len(data) == len(labels)
    assert len(data[0]) == 100

    print('Slide Compiler test passed')

    data, labels = compiler(X, y, method = 'random', window = 100, step = 10, qty = 500)

    assert len(data) == len(labels)
    assert len(data[0]) == 100

    print('Random Compiler test passed')

    data, labels = compiler(X, y, method = 'bootstrapped', window = 100, step = 10, qty = 500)

    assert len(data) == len(labels)
    assert len(data[0]) == 100

    print('Bootstrapped Compiler test passed')

    # Test the shapelet extraction

    for shapelet_method, compiler_method, metric in iter.product(['barycenter', 'random', 'exhaustive'], ['slide', 'random', 'bootstrapped'], ['dtw']):

        print()
        print(f'Testing shapelet extraction with {shapelet_method} method, {compiler_method} compiler, and the {metric} metric')

        clf = ShapeletClassifier(metric = metric, classification_window = 300, w = 0.5)

        training_time_start = time.time()

        clf.fit(X, y, shapelet_method = shapelet_method, compiler_method = compiler_method, qty = 100, 
                window = 100, step = 30, verbose = False, max_iter = 2000,
                min_dist = 70, max_dist = 120, hidden_layer_sizes = (10,))
        
        training_time_end = time.time()
        print(f'Training time: {training_time_end - training_time_start:.4f} seconds')

        indexer = np.random.randint(0, len(X) - 300)

        inference_time_start = time.time()

        print(clf.predict([X[indexer:indexer+300]]), y[indexer + 150])

        inference_time_end = time.time()
        print(f'Inference time: {inference_time_end - inference_time_start:.4f} seconds')

        clf.save(f'data/sample_data/model{shapelet_method}_{compiler_method}_{metric}.pkl')

    print()
    print('Shapelet Classifier test passed')

if __name__ == '__main__':
    main()