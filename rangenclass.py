import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


class ProbabilityDensityFunction:

    def __init__(self, x, y, **kwargs):
        # x and y have to be np.array
        self._x = np.array(x)
        self._y = np.array(y)
        self.pdf = interpolate.InterpolatedUnivariateSpline(self._x, self._y, **kwargs)
        ycdf = np.array([self.pdf.integral(self._x[0], t) for t in self._x])
        self.cdf = interpolate.InterpolatedUnivariateSpline(self._x, ycdf, **kwargs)
        xppf, ippf = np.unique(ycdf, return_index=True)
        yppf = x[ippf]
        self.ppf = interpolate.InterpolatedUnivariateSpline(xppf, yppf, **kwargs)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def __call__(self, t):
        return self.pdf(t)

    def prob(self, x1, x2):
        return self.cdf(x2)-self.cdf(x1)

    def rnd(self, n=1000):
        return self.ppf(np.random.uniform(0., 1., n))


def triangular_test():
    triangx = np.linspace(0, 1, 100)
    triangy = 2 * triangx
    triangular = ProbabilityDensityFunction(triangx, triangy)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    ax1.plot(triangular.x, triangular.pdf(triangular.x), label='pdf')
    ax2.plot(triangular.x, triangular.cdf(triangular.x), label='Cumulative')
    ax3.plot(triangular.x, triangular.ppf(triangular.x), label='ppf')
    rng = triangular.rnd(10000000)
    ax4.hist(rng, label='Random', bins=200)


if __name__ == "__main__":
    triangular_test()
    plt.show()
