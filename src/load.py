import numpy as np

def load():
    x_train = np.load('lfw/data/npy/x_train.npy')
    x_test = np.load('lfw/data/npy/x_test.npy')
    return x_train, x_test

