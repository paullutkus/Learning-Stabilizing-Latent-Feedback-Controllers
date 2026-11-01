"""Latent control synthesis, lifted output feedback u = K E x, and closed-loop evaluation.

lqr     discrete LQR on (L, H) with Q = I_m, R = I_p.
hinf    discrete H-infinity state feedback for z_{t+1} = (L + H K) z_t + w_t with performance
        output y = [z; rho u], minimizing ||w -> y||_inf by the bounded-real LMI.  The z-block
        is the objective sup_{|z|>=1} ||(zI - (L+HK))^-1|| of Thm. resolvent_gain_bound; the rho u
        block replaces the hard gain bound by a soft one (rho = 1).

Evaluator(A, B, ...) scores a learned (E, D, L, H): subspace distance to G_+^R(A^T) (sub_dist =
Delta_F, the paper's Delta; sub_dist_op = operator norm; equal when m_+ = 1), forward residual in
the orthonormal-encoder gauge, conditioning of E E^T; and for each latent design the closed loop
A + B K E: stability (rho(A + B K E) < 1 - STAB_TOL), spectral radius, LQR cost relative to
full-state LQR (x_0 ~ N(0, I)), and the 2% settling time of |x_t| from random initial states.

standard_ics(arm)  the non-linear test set: 8 ICs per tip displacement along the buckling mode
                   (seeds 777, 778, ... per tip; with shared=True, as in every experiment, seed 777
                   at every tip, so the ICs of the tips are scaled copies of each other).
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.linalg as sla

from .linalg import normalize_gauge, spectral_radius, subspace_distance, unstable_left_basis

STAB_TOL = 1e-6


def dlqr(A, B, Q=None, R=None):
    n, p = B.shape
    Q = np.eye(n) if Q is None else Q
    R = np.eye(p) if R is None else R
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B))):
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            P = sla.solve_discrete_are(A, B, Q, R)
        K = -np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    except (np.linalg.LinAlgError, ValueError):
        return None
    if not np.all(np.isfinite(K)) or spectral_radius(A + B @ K) >= 1.0 - STAB_TOL:
        return None
    return K


class HinfSynth:
    """min gamma s.t. ||[x; rho u]||_2 <= gamma ||w||_2 for x+ = (L + HK) x + w (cached per shape)."""

    _cache = {}

    def __new__(cls, m, p, rho=1.0):
        key = (m, p, rho)
        if key not in cls._cache:
            obj = super().__new__(cls)
            obj._build(m, p, rho)
            cls._cache[key] = obj
        return cls._cache[key]

    def _build(self, m, p, rho):
        import cvxpy as cp
        self.cp = cp
        self.L, self.H = cp.Parameter((m, m)), cp.Parameter((m, p))
        X = cp.Variable((m, m), symmetric=True)
        Y = cp.Variable((p, m))
        g = cp.Variable()
        AX = self.L @ X + self.H @ Y
        CX = cp.vstack([X, rho * Y])
        I, Z = np.eye(m), np.zeros
        M = cp.bmat([[X, AX, I, Z((m, m + p))],
                     [AX.T, X, Z((m, m)), CX.T],
                     [I, Z((m, m)), g * np.eye(m), Z((m, m + p))],
                     [Z((m + p, m)), CX, Z((m + p, m)), g * np.eye(m + p)]])
        cons = [0.5 * (M + M.T) >> 1e-9 * np.eye(4 * m + p), X >> 1e-9 * np.eye(m)]
        self.prob = cp.Problem(cp.Minimize(g), cons)
        self.X, self.Y, self.g = X, Y, g

    def __call__(self, L, H):
        if not (np.all(np.isfinite(L)) and np.all(np.isfinite(H))):
            return None, np.inf
        self.L.value, self.H.value = L, H
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.prob.solve(solver=self.cp.CLARABEL)
        except Exception:
            return None, np.inf
        if self.prob.status not in ("optimal", "optimal_inaccurate") or self.X.value is None:
            return None, np.inf
        K = self.Y.value @ np.linalg.inv(self.X.value)
        if not np.all(np.isfinite(K)) or spectral_radius(L + H @ K) >= 1.0 - STAB_TOL:
            return None, np.inf
        return K, float(self.g.value)


def design(kind, L, H):
    if kind == "lqr":
        return dlqr(L, H)
    if kind == "hinf":
        return HinfSynth(L.shape[0], H.shape[1])(L, H)[0]
    raise ValueError(kind)


def settle_steps(rel, tol):
    """First step after which rel (steps, k) stays <= tol (inf if never within the window)."""
    outside = rel > tol
    last = rel.shape[0] - 1 - np.argmax(outside[::-1], axis=0)
    st = np.where(outside.any(axis=0), last + 1, 0).astype(float)
    st[outside[-1]] = np.inf
    return st


class Evaluator:
    def __init__(self, A, B, dt=0.05, n_ic=32, t_sim=30.0, settle_tol=0.02, seed=12345):
        self.A, self.B, self.dt = A, B, dt
        self.n, self.p = B.shape
        self.Qp = unstable_left_basis(A)
        self.X0 = np.random.default_rng(seed).standard_normal((n_ic, self.n))
        self.n_sim, self.settle_tol = int(np.ceil(t_sim / dt)), settle_tol
        P = sla.solve_discrete_are(A, B, np.eye(self.n), np.eye(self.p))
        self.J_star = float(np.trace(P))                       # full-state LQR cost, x_0 ~ N(0, I)
        self.K_star = -np.linalg.solve(np.eye(self.p) + B.T @ P @ B, B.T @ P @ A)

    def diagnostics(self, E, D, L, H):
        if not all(np.all(np.isfinite(X)) for X in (E, D, L, H)):
            return dict(finite=False, sub_dist=np.nan, fwd_res=np.nan, E_smin2=np.nan, E_smax2=np.nan,
                        rho_L=np.nan)
        sv = np.linalg.svd(E, compute_uv=False)
        out = dict(finite=True, E_smin2=float(sv[-1] ** 2), E_smax2=float(sv[0] ** 2),
                   rho_L=spectral_radius(L))
        if sv[-1] <= 1e-12 * max(sv[0], 1e-300):
            return dict(out, sub_dist=float(np.sqrt(self.Qp.shape[1])), sub_dist_op=1.0, fwd_res=np.nan)
        out["sub_dist"] = subspace_distance(self.Qp, E, "fro")
        out["sub_dist_op"] = subspace_distance(self.Qp, E)
        Eh, Lh, _ = normalize_gauge(E, L, H)
        out["fwd_res"] = float(np.linalg.norm(Eh @ self.A - Lh @ Eh))
        return out

    def closed_loop(self, F):
        """Metrics of u = F x on the true linear system."""
        nan = dict(stable=False, rho_cl=np.nan, cost_ratio=np.inf, settle=np.inf)
        if F is None or not np.all(np.isfinite(F)):
            return nan
        Acl = self.A + self.B @ F
        rho = spectral_radius(Acl)
        out = dict(rho_cl=rho, stable=bool(rho < 1.0 - STAB_TOL))
        if not out["stable"]:
            return dict(nan, rho_cl=rho)
        P = sla.solve_discrete_lyapunov(Acl.T, np.eye(self.n) + F.T @ F)
        out["cost_ratio"] = float(np.trace(P)) / self.J_star
        rel = self.response(F)
        st = settle_steps(rel, self.settle_tol) * self.dt
        out["settle"] = float(np.median(st))                  # reported; not part of "stable"
        return out

    def response(self, F):
        """|x_t| / |x_0| (steps+1, n_ic) of x+ = (A + B F) x from the test states."""
        Acl = self.A + self.B @ F
        X = self.X0.copy()
        norms = [np.linalg.norm(X, axis=1)]
        for _ in range(self.n_sim):
            X = X @ Acl.T
            norms.append(np.linalg.norm(X, axis=1))
        norms = np.stack(norms)
        return norms / norms[0]

    def evaluate(self, E, D, L, H, kinds=("lqr", "hinf")):
        """Diagnostics and, per latent design, the closed loop of u = K E x (raw gauge)."""
        res = self.diagnostics(E, D, L, H)
        gains = {}
        for kind in kinds:
            K = design(kind, L, H) if res["finite"] else None
            F = None if K is None else K @ E
            gains[kind] = F
            res.update({f"{kind}_{k}": v for k, v in self.closed_loop(F).items()})
        return res, gains


def standard_ics(arm, tips=(0.02, 0.10, 0.30), k=8, shared=False):
    """{tip [m]: (k, n) ICs}: the non-linear test set of every figure (seeds 777 + j, or 777 for every
    tip if ``shared``: the same sign and bend profile, scaled to each tip displacement)."""
    return {tip: arm.test_ics(k, tip=tip, seed=777 + (0 if shared else j)) for j, tip in enumerate(tips)}


def simulate(arm, F, X0, t_end=12.0, plant="nonlinear", u_max=None, encoder=None):
    """Closed loop x+ = plant(x, sat(F x)) (or sat(K E(x)) with ``encoder``) from X0 (k, n).

    Returns states (steps+1, k, n), inputs (steps, k) in physical units, and success:
    no section beyond 90 degrees and |x(t_end)| <= 5% |x(0)|."""
    steps = int(round(t_end / arm.dt))
    u_max = arm.u_max if u_max is None else u_max
    umax_n = u_max * arm.u_scale
    X = np.array(X0, float)
    xs, us = [X.copy()], []
    failed = np.zeros(len(X), bool)
    with np.errstate(all="ignore"):
        for _ in range(steps):
            U = (encoder(X) if encoder is not None else X @ F.T)
            U = np.clip(U, -umax_n, umax_n)
            X = arm.step(X, U, plant)
            bad = ~np.all(np.isfinite(X), axis=1)
            X[bad] = 0.0
            failed |= bad | arm.fallen(X)
            xs.append(X.copy())
            us.append(U[:, 0] / arm.u_scale)
    xs = np.stack(xs)
    ok = (~failed) & (np.linalg.norm(xs[-1], axis=1) <= 0.05 * np.linalg.norm(xs[0], axis=1))
    return xs, np.stack(us), ok
