import sys
sys.path.append('..')
import unittest
import pandas as pd

from processData import processData
class ProcessDataTests(unittest.TestCase):
    smallCandyData = None

    def test_processData(self):
        'Test dimensions of output from process data'
        designM, labels = processData('../data/small-candy.csv')
        self.assertEqual(designM.shape, (15,12))
        self.assertEqual(labels.shape, (15,1))


if __name__ == '__main__':
    unittest.main()
