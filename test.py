import unittest

import numpy as np

from rangenpdf import ProbabilityDensityFunction


class NewTest(unittest.TestCase):

    def test_triangular(self):

        triangx = np.linspace(0., 1., 100)
        triangy = 2 * triangx
        triangular = ProbabilityDensityFunction(triangx, triangy)
        self.assertAlmostEqual(triangular.prob(0., 1.), 1.)


if __name__ == '__main__':
    unittest.main
