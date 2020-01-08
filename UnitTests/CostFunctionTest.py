import unittest
import numpy as np
from  verticalize import verticalize
from costFunction import costFunction
class CostFunctionTest(unittest.TestCase):
    def test_costFunction_1(self):
        'Not regualarized, hypo matches data perfectly'
        m = 3
        regParam = 0
        hypo = verticalize(np.ones(m))
        labels = verticalize(np.ones(m))
        theta = verticalize(np.ones(m)) #TODO: make cost function response to params isn't actually used: regParam = 0
        self.assertEqual(0, costFunction(hypo,theta,labels,regParam,m))

    def test_costFunction_2(self):
        'Not regularized, hypo opposite of data'
        m = 3
        regParam = 0
        hypo = verticalize(np.zeros(m))
        labels = verticalize(np.ones(m))
        theta = verticalize(np.ones(m))
        self.assertEqual(m / (2 * m), costFunction(hypo,theta,labels,regParam, m))

    def test_costFunction_3(self):
        'TODO: add reguarlized test'

if __name__ == '__main__':
    unittest.main()
