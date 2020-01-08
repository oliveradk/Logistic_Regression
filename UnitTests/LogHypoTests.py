import sys
sys.path.append('..')
import unittest
import numpy as np
import logHypo as lh
from math import e
from verticalize import verticalize
class LogHypoTests(unittest.TestCase):

    def test_sigmoid(self):
        self.assertEqual(0.5, lh.sigmoid(0))
        self.assertEqual(1 / (1 + e), lh.sigmoid(-1))
        self.assertEqual(1 / (1 + (1/e)), lh.sigmoid(1))

    def test_LogHypo_1(self):
        'Random designM, theta == 0'
        m = 5
        n = 4
        designM = np.random.random((m,n+1))
        theta = np.zeros((n + 1, 1))
        hypo = lh.logHypo(designM, theta)
        self.assertTrue((hypo == (np.ones((m,1))) * .5).all())

    def test_LogHypo_2(self):
        'Identity design matrix, theta = [1,0,-1]'
        designM = np.identity(3)
        theta = verticalize(np.array([1,0,-1]))
        hypo = lh.logHypo(designM, theta)
        correctHypo = verticalize(np.array([1 / (1 + 1 / e), 1 / 2, 1 / (1 + e)]))
        self.assertTrue((hypo == correctHypo).all())

    def test_logHypo_3(self):
        'Rows of design matrix sum to 1,0,-1, theta = 1'
        n = 10
        designM = np.array([np.ones(n) * (1/n), np.zeros(n), np.ones(n) * (-1/n)])
        theta = verticalize(np.ones(n))
        hypo = lh.logHypo(designM, theta)
        correctHypo = verticalize(np.array([1 / (1 + 1 / e), 1 / 2, 1 / (1 + e)]))
        print(correctHypo)
        self.assertTrue((hypo == correctHypo).all())



if __name__ == '__main__':
    unittest.main()
