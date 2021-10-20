"""A probability density function class."""
import numpy as np
from scipy import interpolate


class ProbabilityDensityFunction:
    """
    A docstring for this class
    """
    def __init__(self, x, y, **kwargs):
        # make sure that x, y are arrays
        xarr = np.array(x)
        yarr = np.array(y)
        self.pdf = interpolate.InterpolatedUnivariateSpline(xarr, yarr, **kwargs)
        ycdf = np.array([self.pdf.integral(xarr[0], t) for t in xarr])
        self.cdf = interpolate.InterpolatedUnivariateSpline(xarr, ycdf, **kwargs)
        xppf, ippf = np.unique(ycdf, return_index=True)
        yppf = x[ippf]
        self.ppf = interpolate.InterpolatedUnivariateSpline(xppf, yppf, **kwargs)

    def __call__(self, t):
        return self.pdf(t)

    def prob(self, x1, x2):
        """
        Gets the probability that a variable is between x1 and x2.
        :param x1: Sets the probability interval
        :param x2: Sets the probability interval
        :return: The probability that a variable is between x1 and x2.
        """
        if x1 > x2:
            return self.cdf(x1) - self.cdf(x2)

        return self.cdf(x2) - self.cdf(x1)

    def rnd(self, n=1000):
        """
        This generates a number of points, distributed with the pdf in the class attribute.
        :param n: The number of point generated.
        :return: an array of points distributed with the pdf
        """
        return self.ppf(np.random.uniform(0., 1., n))
