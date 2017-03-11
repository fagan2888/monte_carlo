import unittest
import numpy as np

from BayesianInference import *

class TestBayesianInference(unittest.TestCase):
    def test_prior(self):
        class MyPrior(Prior):
            def __init__(self, *args):
                super().__init__(*args)
                self.argin = args[0]

            def __call__(self, *args, **kwargs):
                return sum(self.argin) + sum(args[0])

        mp = MyPrior([1, 2, 3])
        self.assertEqual(mp([1, 2, 3]), 12)


    def test_sample_normal(self):
        class MyPrior(Prior):
            def __init__(self, *args):
                super().__init__(*args)

            def __call__(self, *args, **kwargs):
                mu = args[0][0]
                sigma = args[0][1]
                return np.exp(- 1.0 / 2 * mu ** 2) * np.exp(-1.0 / 2 * (sigma - 1) ** 2)

        class MyLikelihood(Likelihood):
            def __init__(self, *args):
                super().__init__(*args)

            def __call__(self, *args, **kwargs):
                mu = args[0][0]
                sigma = args[0][1]
                data = args[1]
                return np.exp(-1.0 / 2 * np.sum(np.square(data - mu)) / sigma)

        class MyProposal(Proposal):
            def __init__(self):
                super().__init__()

            def __call__(self, *args, **kwargs):


if __name__ == '__main__':
    unittest.main()
