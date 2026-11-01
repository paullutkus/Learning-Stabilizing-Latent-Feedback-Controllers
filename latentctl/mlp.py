"""Encoder, decoder and control-affine latent dynamics as tanh MLPs (Secs. 1.1 and 6.3).

Latent model:  z_{t+1} = f_z(z_t) + g_z(z_t) u_t  (the paper's a_z is f_z here).

E, D, f_z, g_z are tanh MLPs (``depth`` hidden layers of width ``width``, linear output).  E, D and f_z
are anchored, E(x) = N_E(x) - N_E(0), so x = 0 maps to z = 0 and z = 0 is a latent equilibrium; u = K E(x)
then has the plant's equilibrium.  Every MLP reads and writes data units through fixed scales (s_x:
per-coordinate RMS of the training states, s_z = RMS|x| / sqrt(m)); without them the tanh units saturate
on the unstable latent coordinate.  f_z starts with J_fz(0) a random orthogonal matrix (as the linear
model's L).  At the equilibrium, (J_E(0), J_D(0), J_fz(0), g_z(0)) plays the role of the linear model's
(E, D, L, H).

rescale(th, old, new)  the parameters that represent the same functions E, D, f_z, g_z under new
                       scales (s_x, s_z): the warm start of the horizon path, whose scales grow with T.
make_loss(...)         the training loss on a minibatch (L_bwd, or the decoder-free L_fwd); full_objective(...)
                       the same loss on all trajectories and a fixed set of regularizer states (restart selection).
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp


def mlp_init(key, sizes, dtype):
    return [(jax.random.normal(k, (a, b), dtype) / float(np.sqrt(a)), jnp.zeros((b,), dtype))
            for k, (a, b) in zip(jax.random.split(key, len(sizes) - 1), zip(sizes[:-1], sizes[1:]))]


def mlp_apply(params, x):
    for W, b in params[:-1]:
        x = jnp.tanh(x @ W + b)
    W, b = params[-1]
    return x @ W + b


def anchored(params, x):
    return mlp_apply(params, x) - mlp_apply(params, jnp.zeros(x.shape[-1], x.dtype))


class Model:
    def __init__(self, n, m, p, width=64, depth=2, dtype=jnp.float32):
        self.n, self.m, self.p, self.width, self.depth, self.dtype = n, m, p, width, depth, dtype
        self.s_x, self.s_z = 1.0, 1.0

    @property
    def scales(self):
        return (self.s_x, self.s_z)

    def set_scales(self, X):
        X = np.asarray(X, float).reshape(-1, self.n)
        self.s_x = float(np.sqrt(np.mean(X ** 2)))
        self.s_z = float(np.sqrt(np.mean(np.sum(X ** 2, axis=1)) / self.m))

    def init(self, key):
        n, m, p, dt = self.n, self.m, self.p, self.dtype
        h = [self.width] * self.depth
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        f = mlp_init(k3, [m, *h, m], dt)
        # J_fz(0)^T = W_1 ... W_K with zero biases (row convention): choose the minimum-norm last
        # layer with (W_1 ... W_{K-1}) W_K = Q^T for a random orthogonal Q
        Q = jnp.linalg.qr(jax.random.normal(k5, (m, m), dt))[0]
        P = f[0][0]
        for W, _ in f[1:-1]:
            P = P @ W
        f[-1] = (jnp.linalg.pinv(P) @ Q.T, f[-1][1])
        return dict(E=mlp_init(k1, [n, *h, m], dt), D=mlp_init(k2, [m, *h, n], dt), f=f,
                    g=mlp_init(k4, [m, *h, m * p], dt))

    def enc(self, th, x):
        return self.s_z * anchored(th["E"], x / self.s_x)

    def dec(self, th, z):
        return self.s_x * anchored(th["D"], z / self.s_z)

    def f(self, th, z):
        return self.s_z * anchored(th["f"], z / self.s_z)

    def g(self, th, z):
        return mlp_apply(th["g"], z / self.s_z).reshape(z.shape[:-1] + (self.m, self.p))

    def step(self, th, z, u):
        return self.f(th, z) + jnp.einsum("...mp,...p->...m", self.g(th, z), u)

    def jac_E(self, th, x):
        """J_E at states x (k, n) -> (k, m, n)."""
        return jax.vmap(jax.jacrev(lambda y: self.enc(th, y)))(x)

    def linearize(self, th):
        """(J_E(0), J_D(0), J_fz(0), g_z(0)) in float64."""
        th = jax.tree_util.tree_map(lambda a: jnp.asarray(a, jnp.float64), th)
        x0, z0 = jnp.zeros(self.n, jnp.float64), jnp.zeros(self.m, jnp.float64)
        return tuple(np.asarray(a) for a in (jax.jacrev(lambda x: self.enc(th, x))(x0),
                                             jax.jacrev(lambda z: self.dec(th, z))(z0),
                                             jax.jacrev(lambda z: self.f(th, z))(z0), self.g(th, z0)))


def make_loss(model, lam_fwd, lam_reg, n_reg, fwd_horizon="T", loss="bwd", reg="orth"):
    """Empirical L_bwd^(T) on a minibatch of trajectories X (B, T+1, n), U (B, T, p):

        J_bwd^(T) + lam_fwd J_fwd + lam_reg R(E),  R(E) = mean_x ||I - J_E(x) J_E(x)^T||_F^2

    over ``n_reg`` random states of the batch (for a linear encoder, ||I - E E^T||_F^2); ``loss = "fwd"``
    drops J_bwd (the decoder-free L_fwd: D is not trained).  ``reg = "sigreg"`` replaces R by SIGReg on
    the trajectory states, the sliced Epps-Pulley statistic of E(x) at ``n_reg`` random states of the batch
    along 16 random unit directions (latentctl.losses.epps_pulley, as the linear models' 'sigreg').  The
    forward term, by ``fwd_horizon``:
        "T"   J_fwd^(T):  mean_{t=1..T} ||E(x_t) - z_t||^2 along the open-loop latent rollout z_t
        1     J_fwd^(1):  ||E(x_1) - z_1||^2, the one-step prediction from x_0 only
        "tf"  teacher forcing: mean_{t=0..T-1} ||E(x_{t+1}) - f_z(E(x_t)) - g_z(E(x_t)) u_t||^2,
              the one-step prediction from every observed state"""
    assert fwd_horizon in ("T", "tf", 1, "1") and loss in ("bwd", "fwd") and reg in ("orth", "sigreg")

    def loss_terms(th, X, U, key):
        Xt = jnp.swapaxes(X, 0, 1)
        if loss == "bwd" or fwd_horizon != "tf":
            z0 = model.enc(th, X[:, 0])

            def roll(z, u):
                z = model.step(th, z, u)
                return z, z

            _, Z = jax.lax.scan(roll, z0, jnp.swapaxes(U, 0, 1))
            Z = jnp.concatenate([z0[None], Z], axis=0)                      # (T+1, B, m)
        J_bwd = (jnp.mean(jnp.sum((Xt - model.dec(th, Z)) ** 2, axis=-1)) if loss == "bwd"
                 else jnp.zeros((), X.dtype))
        if fwd_horizon == "T":
            J_fwd = jnp.mean(jnp.sum((model.enc(th, Xt[1:]) - Z[1:]) ** 2, axis=-1))
        elif fwd_horizon == "tf":
            Ze = model.enc(th, Xt)                                          # (T+1, B, m)
            J_fwd = jnp.mean(jnp.sum((Ze[1:] - model.step(th, Ze[:-1], jnp.swapaxes(U, 0, 1))) ** 2, axis=-1))
        else:
            J_fwd = jnp.mean(jnp.sum((model.enc(th, Xt[1]) - Z[1]) ** 2, axis=-1))
        Xs = X.reshape(-1, X.shape[-1])
        if reg == "orth":
            JE = model.jac_E(th, Xs[jax.random.randint(key, (n_reg,), 0, Xs.shape[0])])
            G = jnp.einsum("kmn,kln->kml", JE, JE)
            R = jnp.mean(jnp.sum((jnp.eye(model.m, dtype=G.dtype) - G) ** 2, axis=(1, 2)))
        else:
            from latentctl.losses import epps_pulley
            k1, k2 = jax.random.split(key)
            a = jax.random.normal(k2, (16, model.m), X.dtype)
            R = epps_pulley(model.enc(th, Xs[jax.random.randint(k1, (n_reg,), 0, Xs.shape[0])]),
                            a / jnp.linalg.norm(a, axis=1, keepdims=True))
        tot = J_bwd + lam_fwd * J_fwd + lam_reg * R
        return tot, dict(J_bwd=J_bwd, J_fwd=J_fwd, R=R, total=tot)

    return loss_terms


def rescale(th, old, new):
    """MLP parameters representing the same E, D, f_z, g_z after the scales change from old = (s_x, s_z)
    to new: E(x) = s_z N_E(x / s_x) needs N_E'(y) = (s_z / s_z') N_E(y s_x' / s_x), i.e. the first layer
    times s_x' / s_x and the last layer times s_z / s_z'; likewise for D, f_z and g_z (g_z has no output
    scale)."""
    (sx0, sz0), (sx1, sz1) = old, new

    def scale(net, a_in, a_out):
        net = [(W, b) for W, b in net]
        net[0] = (net[0][0] * a_in, net[0][1])
        net[-1] = (net[-1][0] * a_out, net[-1][1] * a_out)
        return net

    return dict(E=scale(th["E"], sx1 / sx0, sz0 / sz1), D=scale(th["D"], sz1 / sz0, sx0 / sx1),
                f=scale(th["f"], sz1 / sz0, sz0 / sz1), g=scale(th["g"], sz1 / sz0, 1.0))


def full_objective(model, lam_fwd, lam_reg, fwd_horizon="T", n_reg=2048, seed=0, loss="bwd", reg="orth"):
    """obj(th, X, U): the loss of ``make_loss`` on all trajectories (X (N, T+1, n), U (N, T, p)) with
    a fixed random set of ``n_reg`` regularizer states: deterministic, so restarts and the warm and
    cold branches are compared on the same objective."""
    terms = make_loss(model, lam_fwd, lam_reg, n_reg, fwd_horizon, loss, reg)
    key = jax.random.PRNGKey(seed)
    return lambda th, X, U: terms(th, X, U, key)[0]
