import sys
sys.path.append('..')
import unittest
import numpy as np
import matplotlib.pyplot as plt

from gradientDescent import gradientDescent

class GradientDescentTests(unittest.TestCase):

    def Xtest_1(self): #TODO: add back
        'Base case: everything 0'
        m = 5
        n = 5
        X = np.zeros((m,n))
        theta = np.zeros((n,1))
        alpha = 1
        regParam = 0
        iterations = 50
        y = np.zeros((m,1))

        theta = gradientDescent(theta, X, y, m, alpha, regParam, iterations)
        self.assertTrue(np.array_equal(theta, np.zeros((n,1))))
        self.assertEqual(type(theta), type(np.array([])))



if __name__ == '__main__':
    unittest.main()
