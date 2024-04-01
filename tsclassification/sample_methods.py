import numpy as np
from numpy.lib.stride_tricks import as_strided

# Bootstrapped subsequence sampling from a stepped sliding window, with a random initialization
def bootstrapped(X, Y, window = 100, step = 1, qty = 1000):

    data = []
    labels = []
    for label in np.unique(Y):
        x = X[np.where(Y == label)[0]]
        y = Y[np.where(Y == label)[0]]
        temp_data = []
        temp_labels = []
        if len(x)>0:
            a = 0
            while len(temp_data)<qty:
                rano = np.random.randint(0, len(x)) if a != 0 else a
                a += 1
                for i in range(rano, len(x) - window, step):
                    if len(temp_data)>=qty:
                        break
                    temp_labels.append(y[i])
                    temp_data.append(np.array(x[i:i+window]))
            data.extend(temp_data[:qty])
            labels.extend(temp_labels[:qty])

    return data, labels

# Simple stepped sliding window for subsequence extraction
def slide(X, Y, window = 100, step = 1, qty = 1000):

    data = []
    labels = []
    for label in np.unique(Y):
        x = X[np.where(Y == label)[0]]
        y = Y[np.where(Y == label)[0]]
        temp_data = []
        temp_labels = []
        for i in range(0, len(x) - window, step):
            if len(temp_data) >= qty:
                break
            temp_data.append(np.array(x[i:i+window]))
            temp_labels.append(y[i])
        data.extend(temp_data[:qty])
        labels.extend(temp_labels[:qty])

    return data, labels

# Extracts randomly positioned subsequences
def random(X, Y, window = 100, qty = 1000):
    
    data = []
    labels = []
    for label in np.unique(Y):
        x = X[np.where(Y == label)[0]]
        y = Y[np.where(Y == label)[0]]
        if len(x) > window:
            for n in range(qty):
                indexer = np.random.randint(window//2, len(x) - window//2)
                series = x[indexer-window//2:indexer+window//2]
                data.append(np.array(series))
                labels.append(y[indexer])

    return data, labels

def compiler(X, y, method = 'slide', window = 100, step = 1, qty = 1000):

    if method == 'slide':
        return slide(X, y, window = window, step = step, qty = qty)
    
    elif method == 'random':
        return random(X, y, window = window, qty = qty)
    
    elif method == 'bootstrapped':
        return bootstrapped(X, y, window = window, step = step, qty = qty)
        
    else:
        raise ValueError('Only slide, random and bootstrapped methods are supported')
