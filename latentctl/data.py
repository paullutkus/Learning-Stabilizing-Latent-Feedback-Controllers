"""Trajectory datasets under the paper's Assumption data_model.

x_0 ~ N(0, sx^2 I_n), u_t ~ N(0, alpha_T(t) su^2 I_p) with alpha_T(t) = 1/T, independent across
trajectories.  Datasets for different N are nested (the first N trajectories of one draw); for a
given seed the initial states are the same for every T (the inputs are not).

linear_dataset  x_{t+1} = A x_t + B u_t
arm_dataset     the arm's nonlinear step in scaled coordinates x~ = x / eps, u~ = u / eps:
                x~_{t+1} = f(eps x~_t, eps u~_t) / eps.  The scaled plant has the same equilibrium and
                linearization (A, B) for every eps; eps sets how far the data reach into the
                nonlinearity (eps = 0.01: initial joint angles ~0.6 deg).
"""

from __future__ import annotations

import numpy as np


def excitation(n, p, T, N, seed, sx=1.0, su=0.5):
    rng = np.random.default_rng(seed)
    X0 = sx * rng.standard_normal((N, n))
    U = su * np.sqrt(1.0 / T) * rng.standard_normal((N, T, p))
    return X0, U


def rollouts(step, X0, U):
    N, T = U.shape[:2]
    X = np.empty((N, T + 1, X0.shape[1]))
    X[:, 0] = X0
    for t in range(T):
        X[:, t + 1] = step(X[:, t], U[:, t])
    return X


def linear_dataset(A, B, T, N, seed, sx=1.0, su=0.5):
    X0, U = excitation(A.shape[0], B.shape[1], T, N, seed, sx, su)
    return rollouts(lambda x, u: x @ A.T + u @ B.T, X0, U), U


def arm_dataset(arm, T, N, seed, eps=0.01, sx=1.0, su=0.5):
    """Nonlinear-arm trajectories in scaled coordinates (see module docstring)."""
    X0, U = excitation(arm.n, arm.p, T, N, seed, sx, su)
    X = rollouts(lambda x, u: arm.step(eps * x, eps * u, "nonlinear") / eps, X0, U)
    if not np.all(np.isfinite(X)):
        raise FloatingPointError(f"non-finite states in {np.sum(~np.all(np.isfinite(X), axis=(1, 2)))} trajectories")
    return X, U


def nonlinearity(arm, X, U, eps):
    """How far scaled nonlinear-arm data (X, U from arm_dataset) are from the linearized rollouts
    with the same (x_0, u): relative Frobenius deviation, and the fraction of trajectories in which
    some section passes 90 degrees within the horizon."""
    Xl = rollouts(lambda x, u: x @ arm.A.T + u @ arm.B.T, X[:, 0], U)
    dev = float(np.sqrt(((X - Xl) ** 2).sum() / (Xl ** 2).sum()))
    return dict(rel_dev=dev, fallen=float(np.mean(arm.fallen(eps * X).any(axis=1))))
