import numpy as np

# Bootstrapped subsequence sampling from a stepped sliding window, with a random initialization
def sample_slide_comile(X, Y, window, step = 1, sample = 100):
    data = []
    labels = []
    for label in np.unique(Y):
        x = X[np.where(Y == label)[0]]
        y = Y[np.where(Y == label)[0]]
        temp_data = []
        temp_labels = []
        if len(x)>0:
            a = 0
            while len(temp_data)<sample:
                rano = np.random.randint(0, len(x)) if a != 0 else a
                a += 1
                for i in range(rano, len(x) - window, step):
                    if len(temp_data)>=sample:
                        break
                    temp_labels.append(y[i])
                    temp_data.append(np.array(x[i:i+window]))
            data.extend(temp_data[:sample])
            labels.extend(temp_labels[:sample])
    return data, labels

# Simple stepped sliding window for subsequence extraction
def slide_compile(X, Y, window, step = 1, sample = 100):
    data = []
    labels = []
    for label in np.unique(Y):
        x = X[np.where(Y == label)[0]]
        y = Y[np.where(Y == label)[0]]
        temp_data = []
        temp_labels = []
        for i in range(0, len(x) - window, step):
            if len(temp_data) >= sample:
                break
            temp_labels.append(y)
            temp_data.append(np.array(x[i:i+window]))
        data.extend(temp_data[:sample])
        labels.extend(temp_labels[:sample])
    return data, labels

# Extracts randomly positioned subsequences
def random_compile(X, Y, window, sample = 300, step = 1):
    data = []
    labels = []
    for label in np.unique(Y):
        x = X[np.where(Y == label)[0]]
        y = Y[np.where(Y == label)[0]]
        if len(x) > window:
            for n in range(sample):
                indexer = np.random.randint(window//2, len(x) - window//2)
                series = x[indexer-window//2:indexer+window//2]
                data.append(np.array(series))
                labels.append(y[indexer])

    return data, labels

compilers = {'random' : random_compile,
             'slide' : slide_compile,
             'sample_slide' : sample_slide_comile,
            }