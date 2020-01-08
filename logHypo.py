import numpy as np
from scipy.special import expit
def logHypo(theta, X):
    "Calculate hypothesis vector for logistic regression"
    weightedSums = np.dot(X, theta)
    return expit(weightedSums) #apply sigmoid
