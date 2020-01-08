import numpy as np
from logHypo import logHypo
from gradientDescent import gradientDescent

def logisticRegression(X, y, alpha, regParam, iterations):
    'Runs logistic regression, returning trained theta'
    m, n = X.shape # 'n' includes intercept
    theta = _initTheta(n)
    return gradientDescent(theta, X, y, m, alpha, regParam, iterations)

def _initTheta(size):
    return np.zeros((size, 1))
