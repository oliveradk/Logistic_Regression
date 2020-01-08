import numpy as np
from logHypo import logHypo
from cost import cost
from plotCost import plotCost
def gradientDescent(theta, X, y, m, alpha, regParam, iterations):
    "Run Gradient Descent, returning updated theta"
    i = 0
    costRecord = []
    while i < iterations:
        hypo = logHypo(theta, X)
        theta = theta - alpha * _gradient(theta, hypo, X, y, m, regParam)
        costRecord.append(((np.asscalar(cost(hypo, theta, y, regParam, m))), i))
        i += 1
    plotCost(costRecord) #TODO remove
    return theta

def _gradient(theta, hypo, X, y, m, regParam):
    return (1 / m) * (_nonRegGradient(hypo, X, y) + _reg(theta, regParam))

def _nonRegGradient(hypo, X, y):
    return np.dot(X.T, (hypo - y))

def _reg(theta, regParam):
    return regParam * np.concatenate((np.zeros((1,1)), theta[1:]))
