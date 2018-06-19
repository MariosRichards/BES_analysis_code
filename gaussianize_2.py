import numpy as np
from scipy.special import lambertw
from scipy.stats import kurtosis, boxcox
from scipy.optimize import fmin


class Gaussianize(object):
    """
    Gaussianization - the process to transform data to be approximately normal
    Used to transform skewed and fat-tailed distributions of search queries
    to approximately normal to determine the good and bad search query cut-offs
    Paper: https://arxiv.org/pdf/1010.2265v5.pdf
    Pieces lifted from https://github.com/gregversteeg/gaussianize/blob/master/gaussianize.py
    """

    def __init__(self, tol=.00001, max_iter=1000, method='lambert'):
        self.tol = tol
        self.max_iter = max_iter
        self.method = method
        self.trans_params = []

    def iterate_moments(self, y, tol=1.22e-4, max_iter=1000):
        delta0 = self.init_delta(y)
        tau1 = (np.median(y), np.std(y) * ((1. - 2. * delta0) ** 0.75), delta0)
        for k in range(max_iter):
            tau0 = tau1
            z = (y - tau1[0]) / tau1[1]
            delta1 = self.delta_gmm(z)
            x = tau0[0] + tau1[1] * self.w_d(z, delta1)
            mu1, sigma1 = np.mean(x), np.std(x)
            tau1 = (mu1, sigma1, delta1)

            if np.linalg.norm(np.array(tau1) - np.array(tau0)) < tol:
                break
            elif k > max_iter:
                break
            else:
                if (k % 100 == 0) and (k < max_iter):
                    print("No convergence after %d iterations" % k)
        return tau1

    def fit(self, x):
        x = np.asarray(x)
        if self.method == 'lambert':
            for x_i in x.T:
                self.trans_params.append(self.iterate_moments(x_i, tol=self.tol,
                                                              max_iter=self.max_iter))
        elif self.method == 'boxcox':
            for x_i in x.T:
                self.trans_params.append(boxcox(x_i)[1])
        else:
            raise NotImplementedError

    def transform(self, x):
        x = np.asarray(x)
        if self.method == 'lambert':
            return np.array([self.w_t(x_i, tp_i) for x_i, tp_i in zip(x.T, self.trans_params)]).T
        elif self.method == 'boxcox':
            return np.array([boxcox(x_i, tp_i) for x_i, tp_i in zip(x.T, self.trans_params)]).T
        else:
            raise NotImplementedError

    def observed_to_normal(self, x):
        self.fit(x)
        return self.transform(x)

    def normal_to_observed(self, y):
        if self.method == 'lambert':
            return np.array([self.inverse(y_i, tp_i) for y_i,
                                                         tp_i in zip(y.T, self.trans_params)]).T
        elif self.method == 'boxcox':
            return np.array([(1. + lmbda_i * y_i)**(1./lmbda_i)
                             for y_i, lmbda_i in zip(y.T, self.trans_params)]).T
        else:
            print('Inversion not supported for this gaussianization transform.')
            raise NotImplementedError

    def w_d(self, z, delta):
        if delta < 1e-3:
            return z
        return np.sign(z) * np.sqrt(np.real(lambertw(delta * z ** 2)) / delta)

    def w_t(self, y, tau):
        return tau[0] + tau[1] * self.w_d((y - tau[0]) / tau[1], tau[2])

    def inverse(self, x, tau):
        u = (x - tau[0]) / tau[1]
        return tau[0] + tau[1] * (u * np.exp(u * u * (tau[2] * 0.5)))

    def delta_gmm(self, z):
        delta0 = self.init_delta(z)

        def kurt(q):
            u = self.w_d(z, np.exp(q))
            if not np.all(np.isfinite(u)):
                return 0.
            else:
                k = kurtosis(u, fisher=True, bias=False)**2
                if not np.isfinite(k) or k > 1e10:
                    return 1e10
                else:
                    return k

        res = fmin(kurt, np.log(delta0), disp=0)
        return np.around(np.exp(res[-1]), 6)

    def init_delta(self, z):
        gamma = kurtosis(z, fisher=False, bias=False)
        with np.errstate(all='ignore'):
            delta0 = np.clip(1. / 66 * (np.sqrt(66 * gamma - 162.) - 6.), 0.01, 0.25)
        if not np.isfinite(delta0):
            delta0 = 0.01
        return delta0