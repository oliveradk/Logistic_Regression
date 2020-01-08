import numpy as np
def accuracy(hypo, labels):
    'Returns accuracy of regression, rounding hypothesis to nearest int'
    return np.average(1 - abs(np.rint(hypo) - labels))
