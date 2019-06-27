 # -*- coding: utf-8 -*-
"""
KTD-Q is implemented on top of the original TDlearn code base.
The algorithm implemented in the original code is XKTD-V, but we wanted to use KTD-Q.

The original code base web address  : https://github.com/chrodan/tdlearn
"""

import numpy as np
import itertools
import copy
import time
import brl_util as util
import pdb

class ValueFunctionPredictor(object):

    """
        predicts the value function of a MDP for a given policy from given
        samples
    """

    def __init__(self, gamma=1, **kwargs):
        self.gamma = gamma
        self.time = 0
        if not hasattr(self, "init_vals"):
            self.init_vals = {}

    def update_V(self, s0, s1, r, V, **kwargs):
        raise NotImplementedError("Each predictor has to implement this")

    def reset(self):
        self.time = 0
        self.reset_trace()
        for k, v in self.init_vals.items():
            self.__setattr__(k, copy.copy(v))

    def reset_trace(self):
        if hasattr(self, "z"):
            if "z" in self.init_vals:
                self.z = self.init_vals["z"]
            else:
                del self.z
        if hasattr(self, "Z"):
            if "Z" in self.init_vals:
                self.Z = self.init_vals["Z"]
            else:
                del self.Z
        if hasattr(self, "last_rho"):
            if "last_rho" in self.init_vals:
                self.Z = self.init_vals["last_rho"]
            else:
                del self.Z

    def _assert_iterator(self, p):
        try:
            return iter(p)
        except TypeError:
            return ConstAlpha(p)

    def _tic(self):
        self._start = time.time()

    def _toc(self):
        self.time += (time.time() - self._start)


class LinearValueFunctionPredictor(ValueFunctionPredictor):

    """
        base class for value function predictors that predict V as a linear
        approximation, i.e.:
            V(x) = theta * phi(x)
    """
    def __init__(self, phi, theta0=None, **kwargs):

        ValueFunctionPredictor.__init__(self, **kwargs)

        self.phi = phi
        if theta0 is None:
            self.init_vals['theta'] = np.array([0])
        else:
            self.init_vals['theta'] = theta0

    def V(self, theta=None):
        """
        returns a the approximate value function for the given parameter
        """
        if theta is None:
            if not hasattr(self, "theta"):
                raise Exception("no theta available, has to be specified"
                                " by parameter")
            theta = self.theta

        return lambda x: np.dot(theta, self.phi(x))


class LambdaValueFunctionPredictor(ValueFunctionPredictor):

    """
        base class for predictors that have the lambda parameter as a tradeoff
        parameter for bootstrapping and sampling
    """
    def __init__(self, lam, z0=None, **kwargs):
        """
            z0: optional initial value for the eligibility trace
        """
        ValueFunctionPredictor.__init__(self, **kwargs)
        self.lam = lam
        if z0 is not None:
            self.init_vals["z"] = z0


class OffPolicyValueFunctionPredictor(ValueFunctionPredictor):

    """
        base class for value function predictors for a MDP given target and
        behaviour policy
    """

    def update_V_offpolicy(
        self, s0, s1, r, a, beh_pi, target_pi, f0=None, f1=None, theta=None,
            **kwargs):
        """
        off policy training version for transition (s0, a, s1) with reward r
        which was sampled by following the behaviour policy beh_pi.
        The parameters are learned for the target policy target_pi

         beh_pi, target_pi: S x A -> R
                numpy array of shape (n_s, n_a)
                *_pi(s,a) is the probability of taking action a in state s
        """
        rho = target_pi.p(s0, a) / beh_pi.p(s0, a)
        kwargs["rho"] = rho
        if not np.isfinite(rho):
            import ipdb
            ipdb.set_trace()
        return self.update_V(s0, s1, r, f0=f0, f1=f1, theta=theta, **kwargs)


class GTDBase(LinearValueFunctionPredictor, OffPolicyValueFunctionPredictor):

    """ Base class for GTD, GTD2 and TDC algorithm """

    def __init__(self, alpha, beta=None, mu=None, **kwargs):
        """
            alpha:  step size. This can either be a constant number or
                    an iterable object providing step sizes
            beta:   step size for weights w. This can either be a constant
                    number or an iterable object providing step sizes
            gamma: discount factor
        """
        LinearValueFunctionPredictor.__init__(self, **kwargs)
        OffPolicyValueFunctionPredictor.__init__(self, **kwargs)

        self.init_vals['alpha'] = alpha
        if beta is not None:
            self.init_vals['beta'] = beta
        else:
            self.init_vals["beta"] = alpha * mu

        self.reset()

    def clone(self):
        o = self.__class__(self.init_vals['alpha'], self.init_vals[
                           'beta'], gamma=self.gamma, phi=self.phi)
        return o

    def __getstate__(self):
        res = self.__dict__
        for n in ["alpha", "beta"]:
            if isinstance(res[n], itertools.repeat):
                res[n] = res[n].next()
        return res

    def __setstate__(self, state):
        self.__dict__ = state
        self.alpha = self._assert_iterator(self.init_vals['alpha'])
        self.beta = self._assert_iterator(self.init_vals['beta'])

    def reset(self):
        LinearValueFunctionPredictor.reset(self)
        self.alpha = self._assert_iterator(self.init_vals['alpha'])
        self.beta = self._assert_iterator(self.init_vals['beta'])
        self.w = np.zeros_like(self.init_vals['theta'])
        if hasattr(self, "A"):
            del(self.A)
        if hasattr(self, "b"):
            del(self.b)

        if hasattr(self, "F"):
            del(self.F)
        if hasattr(self, "Cmat"):
            del(self.Cmat)

    def init_deterministic(self, task):
        self.F, self.Cmat, self.b = self._compute_detTD_updates(task)
        self.A = np.array(self.F - self.Cmat)


class GTD(GTDBase):

    """
    GTD algorithm with linear function approximation
    for details see

    Maei, H. R. (2011). Gradient Temporal-Difference Learning Algorithms.
    (p. 36)
    """
    def update_V(self, s0, s1, r, f0=None, f1=None, rho=1, theta=None, **kwargs):
        """
            rho: weight for this sample in case of off-policy learning
        """
        w = self.w
        if theta is None:
            theta = self.theta
        if f0 is None or f1 is None:
            f0 = self.phi(s0)
            f1 = self.phi(s1)

        self._tic()

        delta = r + self.gamma * np.dot(theta, f1) - np.dot(theta, f0)
        a = np.dot(f0, w)

        w += self.beta.next() * rho * (delta * f0 - w)
        theta += self.alpha.next() * rho * (f0 - self.gamma * f1) * a

        self.w = w
        self.theta = theta

        self._toc()
        return theta

    def deterministic_update(self, theta=None):
        w = self.w
        if theta is None:
            theta = self.theta

        w_d = w + self.beta.next() * (np.dot(self.A, theta) - w + self.b)
        theta_d = theta + self.alpha.next() * (- np.dot(self.A.T, w) + self.b)
        self.theta = theta_d
        self.w = w_d
        return self.theta


class GTD2(GTDBase):

    """
    GTD2 algorithm with linear function approximation
    for details see

    Maei, H. R. (2011). Gradient Temporal-Difference Learning Algorithms.
    (p. 38)
    """

    def update_V(self, s0, s1, r, f0=None, f1=None, rho=1, theta=None, **kwargs):
        """
            rho: weight for this sample in case of off-policy learning
        """
        w = self.w
        if theta is None:
            theta = self.theta
        if f0 is None or f1 is None:
            f0 = self.phi(s0)
            f1 = self.phi(s1)

        self._tic()

        delta = r + self.gamma * np.dot(theta, f1) - np.dot(theta, f0)
        a = np.dot(f0, w)

        w += self.beta.next() * (rho * delta - a) * f0
        theta += self.alpha.next() * rho * a * (f0 - self.gamma * f1)

        self.w = w
        self.theta = theta
        self._toc()
        return theta

    def deterministic_update(self, theta=None):
        w = self.w
        if theta is None:
            theta = self.theta

        w_d = w + self.beta.next(
        ) * (np.dot(self.A, theta) - np.dot(self.Cmat, w) + self.b)
        theta_d = theta + self.alpha.next() * (- np.dot(self.A.T, w) + self.b)
        self.theta = theta_d
        self.w = w_d
        return self.theta


class TDC(GTDBase):

    """
    TDC algorithm with linear function approximation
    for details see

    Maei, H. R. (2011). Gradient Temporal-Difference Learning Algorithms.
    (p. 38)
    """

    def update_V(self, s0, s1, r, f0=None, f1=None, rho=1, theta=None, **kwargs):
        """
            rho: weight for this sample in case of off-policy learning
        """
        w = self.w
        if theta is None:
            theta = self.theta
        if f0 is None or f1 is None:
            f0 = self.phi(s0)
            f1 = self.phi(s1)

        self._tic()
        delta = r + self.gamma * np.dot(theta, f1) - np.dot(theta, f0)
        a = np.dot(f0, w)

        w += self.beta.next() * (rho * delta - a) * f0
        theta += self.alpha.next() * rho * (delta * f0 - self.gamma * f1 * a)
        self.w = w
        self.theta = theta
        self._toc()
        return theta

    def deterministic_update(self, theta=None):
        w = self.w
        if theta is None:
            theta = self.theta

        w_d = w + self.beta.next(
        ) * (np.dot(self.A, theta) - np.dot(self.Cmat, w) + self.b)
        theta_d = theta + self.alpha.next(
        ) * (np.dot(self.A, theta) - np.dot(self.F.T, w) + self.b)
        self.theta = theta_d
        self.w = w_d
        return self.theta


class KTD(LinearValueFunctionPredictor):

    """ Kalman Temporal Difference Learning

        for details see Geist, M. (2010).
            Kalman temporal differences. Journal of artificial intelligence research, 39, 483-532.
            Retrieved from http://www.aaai.org/Papers/JAIR/Vol39/JAIR-3911.pdf
            Algorithm 5 (XKTD-V)
    """
    def __init__(self, kappa=1., theta_noise=0.001, eta=None, P_init=10, reward_noise=0.001, **kwargs):
        LinearValueFunctionPredictor.__init__(self, **kwargs)
        self.kappa = kappa
        self.P_init = P_init
        self.reward_noise = reward_noise
        self.eta = eta
        if eta is not None and theta_noise is not None:
            print("Warning, eta and theta_noise are complementary")
        self.theta_noise = theta_noise
        self.reset()

    def reset(self):
        LinearValueFunctionPredictor.reset(self)
        self.p = len(self.theta)
        if self.theta_noise is not None:
            self.P_vi = np.eye(self.p) * self.theta_noise
        self.P = np.eye(self.p + 2) * self.P_init
        self.x = np.zeros(self.p + 2)
        self.x[:self.p] = self.theta
        self.F = np.eye(self.p + 2)
        self.F[-2:, -2:] = np.array([[0., 0.], [1., 0.]])

    def sample_sigma_points(self, mean, variance):
        n = len(mean)
        X = np.empty((2 * n + 1, n))
        X[:, :] = mean[None, :]
        C = np.linalg.cholesky((self.kappa + n) * variance)
        for j in range(n):
            X[j + 1, :] += C[:, j]
            X[j + n + 1, :] -= C[:, j]
        W = np.ones(2 * n + 1) * (1. / 2 / (self.kappa + n))
        W[0] = (self.kappa / (self.kappa + n))
        return X, W

    def update_V(self, s0, s1, r, f0=None, f1=None, theta=None, rho=1, **kwargs):
        if f0 is None or f1 is None:
            f0 = self.phi(s0)
            f1 = self.phi(s1)
        self._tic()

        if theta is not None:
            print("Warning, setting theta by hand is not valid")

        # Prediction Step
        xn = np.dot(self.F, self.x)
        Pn = np.dot(self.F, np.dot(self.P, self.F.T))
        if self.eta is not None:
            self.P_vi = self.eta * self.P[:-2, :-2]
        Pn[:-2, :-2] += self.P_vi
        Pn[-2:, -2:] += np.array([[1., -self.gamma], [-self.gamma,
                                 self.gamma ** 2]]) * self.reward_noise

        # Compute Sigma Points
        X, W = self.sample_sigma_points(xn, Pn)
        R = (np.dot(f0, X[:, :-2].T) - self.gamma * np.dot(f1, X[:,
             :-2].T) + X[:, -1].T).flatten()

        # Compute statistics of interest
        rhat = (W * R).sum()
        Pxr = ((W * (R - rhat))[:, None] * (X - xn)).sum(axis=0)
        Pr = max((W * (R - rhat) * (R - rhat)).sum(), 10e-5)
                 # ensure a minimum amount of noise to avoid numerical
                 # instabilities

        # Correction Step
        K = Pxr * (1. / Pr)
        # try:
        #    np.linalg.cholesky(Pn - np.outer(K,K)*Pr)
        # except Exception:
        #    import ipdb
        #    ipdb.set_trace()

        self.P = Pn - np.outer(K, K) * Pr

        self.x = xn + K * (r - rhat)
        self.theta = self.x[:-2]
        self._toc()


class KTD_Q(LinearValueFunctionPredictor):

    """ Kalman Temporal Difference Learning

        for details see Geist, M. (2010).
            Kalman temporal differences. Journal of artificial intelligence research, 39, 483-532.
            Retrieved from http://www.aaai.org/Papers/JAIR/Vol39/JAIR-3911.pdf
            Algorithm 3 (KTD-Q))
    """
    def __init__(self, kappa=1., theta_noise=0.001, eta=None, P_init=10, reward_noise=0.001, anum = None, **kwargs):
        LinearValueFunctionPredictor.__init__(self, **kwargs)
        self.kappa = kappa
        self.P_init = P_init
        self.reward_noise = reward_noise
        self.eta = eta
        if eta is not None and theta_noise is not None:
            print("Warning, eta and theta_noise are complementary")
        self.theta_noise = theta_noise
        self.anum = anum
        self.avgmeans = []
        self.avgcovs = []
        self.reset()

    def reset(self):
        LinearValueFunctionPredictor.reset(self)
        self.p = len(self.theta)
        if self.theta_noise is not None:
            self.P_vi = np.eye(self.p) * self.theta_noise
        self.P = np.eye(self.p) * self.P_init
        self.x = np.zeros(self.p)
        self.x[:self.p] = self.theta
        self.F = np.eye(self.p)

    def sample_sigma_points(self, mean, variance):
        n = len(mean)
        X = np.empty((2 * n + 1, n))
        X[:, :] = mean[None, :]
        C = np.linalg.cholesky((self.kappa + n) * variance)
        for j in range(n):
            X[j + 1, :] += C[:, j]
            X[j + n + 1, :] -= C[:, j]
        W = np.ones(2 * n + 1) * (1. / 2 / (self.kappa + n))
        W[0] = (self.kappa / (self.kappa + n))
        return X, W

    def update_V(self, s0, a0, s1, r, theta=None, rho=1, **kwargs):
        f0 = self.phi(s0, a0)
        f1s = [self.phi(s1, a) for a in range(self.anum)]
        
        if theta is not None:
            print("Warning, setting theta by hand is not valid")

        # Prediction Step
        xn = np.dot(self.F, self.x)
        Pn = np.dot(self.F, np.dot(self.P, self.F.T))
        if self.eta is not None:
            self.P_vi = self.eta * self.P
        Pn += self.P_vi
        
        # Compute Sigma Points
        X, W = self.sample_sigma_points(xn, Pn)
        R = (np.dot(f0, X.T) - self.gamma * np.max([np.dot(f1, X.T) for f1 in f1s], axis=0)).flatten()

        # Compute statistics of interest
        rhat = (W * R).sum()
        Pxr = ((W * (R - rhat))[:, None] * (X - xn)).sum(axis=0)
        Pr = max((W * (R - rhat) * (R - rhat)).sum(), 10e-5)
                 # ensure a minimum amount of noise to avoid numerical
                 # instabilities

        # Correction Step
        K = Pxr * (1. / Pr)
        # try:
        #    np.linalg.cholesky(Pn - np.outer(K,K)*Pr)
        # except Exception:
        #    import ipdb
        #    ipdb.set_trace()

        self.P = Pn - np.outer(K, K) * Pr
        self.P = 0.5*self.P + 0.5*np.transpose(self.P) + 1e-4*np.eye(len(self.theta))

        self.x = xn + K * (r - rhat)
        self.theta = self.x
        self.avgmeans.append(np.mean(self.x))
        self.avgcovs.append(np.mean(self.P))

class rbf(object):

    def __init__(self, snum, anum):

        self.snum = snum
        self.anum = anum

    def __call__(self, state, action, const=1.0):
        c1 = np.reshape(np.array([-np.pi/4.0, 0.0, np.pi/4.0]),(3,1)) # For inverted pendulum
        c2 = np.reshape(np.array([-1.0,0.0,1.0]), (1,3)) # For inverted pendulum
        basis = np.exp(-0.5*(c1-state[0])**2)*np.exp(-0.5*(c2-state[1])**2)
        basis = np.append(basis.flatten(), const)
        phi = np.zeros(self.anum * self.snum, dtype=np.float32)
        phi[action*self.snum:(action+1)*self.snum] = basis
        return phi


