"""Thi docstring module"""
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


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


if __name__ == "__main__":

    xpdf = np.linspace(0, 1, 100)
    ypdf = 2 * xpdf
    pdf = ProbabilityDensityFunction(xpdf, ypdf)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    ax1.plot(xpdf, pdf.pdf(xpdf), label='pdf')
    ax2.plot(xpdf, pdf.cdf(xpdf), label='Cumulative')
    ax3.plot(xpdf, pdf.ppf(xpdf), label='ppf')

    rng = pdf.rnd(10000000)
    ax4.hist(rng, label='Random', bins=200)

    plt.show()
