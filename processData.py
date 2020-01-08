import numpy as np
import pandas as pd

def processData(filepath, standardize = True):
    'Return standardized training data and labels from file'
    designM, labels = _readData(filepath)
    if standardize:
        designM = standardizeData(designM)
    designM = addIntercepts(designM)
    return designM, labels

def _readData(filepath):
    df = pd.read_csv(filepath)
    labels = _verticalize(df.iloc[:,1].values)
    designM = df.iloc[:,2:].values
    return designM, labels

def standardizeData(ndarray):
    return np.apply_along_axis(_standardize, 0, ndarray)

def addIntercepts(X):
    intercepts = np.ones((X.shape[0],1))
    return np.concatenate((intercepts, X), axis = 1)

def _standardize(column):
    return (column - np.average(column)) / np.std(column)
def _verticalize(ndarray):
    'Creates vertical array from 1-D numpy array'
    return np.expand_dims(ndarray, axis = 1)
