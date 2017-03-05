import unittest
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

if __name__ == '__main__':
    unittest.main()
