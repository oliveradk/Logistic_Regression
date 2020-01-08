import numpy as np

def cost(hypo, theta, y, regParam, m):
    'Computes "cost" of current theta'
    return  _nonRegCost(hypo, theta, y, m) + _regCost(theta, regParam, m)

def _nonRegCost(hypo, theta, y, m):
    return (1 / m) * np.dot(y.T, np.log(hypo)) - np.dot((1 - y).T, np.log(1 - hypo))

def _regCost(theta, regParam, m):
    return (regParam / (2*m)) * _sumOfSquares(theta)

def _sumOfSquares(vector):
    return np.linalg.norm(vector)**2
