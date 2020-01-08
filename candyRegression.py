import sys
from processData import processData
from logisticRegression import logisticRegression
from logHypo import logHypo
from accuracy import accuracy

def main(args):
    'Tests accuracy of logistic regression in predicting if candies are chocolate'

    alpha, regParam, iterations = _getParams(args)

    #Regression on full data set
    fullDesign, fullLabels = processData('data/candy-data.csv')
    fullTheta, fullAccuracy = _runLogRegression(fullDesign, fullLabels, alpha, regParam, iterations)

    #Regression on first half of data set
    halfDesign, halfLabels = processData('data/firstHalf-candy-data.csv')
    halfTheta, halfAccuracy = _runLogRegression(halfDesign, halfLabels, alpha, regParam, iterations)

    #Test accuracy of params trained on first half on second half
    secondHalfDesign, secondHalfLabels = processData('data/secondHalf-candy-data.csv')
    secondHalfAccuracy = accuracy(logHypo(halfTheta,secondHalfDesign), secondHalfLabels)

    _displayResults(fullAccuracy, halfAccuracy, secondHalfAccuracy)

def _getParams(args):
    if (len(args) < 4):
        alpha = .001
        regParam = 0
        iterations = 1000
    else:
        alpha, regParam, iterations = map(float, sys.argv[1:])
    return alpha, regParam, iterations

def _displayResults(fullAccuracy, halfAccuracy, secondHalfAccuracy):
    print('-----Accuracy-----')
    print('Full data: ', fullAccuracy)
    print('Fist half: ', halfAccuracy)
    print('Second half (trained on first half):', secondHalfAccuracy)

def _runLogRegression(designM, labels, alpha, regParam, iterations):
    theta = logisticRegression(designM, labels, alpha, regParam, iterations)
    a = accuracy(logHypo(theta, designM), labels)
    return theta, a

if __name__ == '__main__':
    main(sys.argv)
