"""Population and finite-sample latent losses of the paper, with the regularizers.

    L_bwd^(T)(E, D, L, H) = J_bwd^(T) + lam_fwd J_fwd^(h) + lam_reg R(E)      (eq. total_backwards_loss)
    L_fwd^(T)(E, L, H)    = J_fwd^(T) + lam_reg R(E)                         (eq. total_forwards_loss)

h = 1 (the one-step forward term J_fwd^(1), ``fwd_horizon = 1``) is what the experiments of Sec. 6 use
(see its methodology paragraph); h = T (``'T'``) is eq. total_backwards_loss as written.

Population forms are Fact LTI_system_losses under the data model x_0 ~ N(0, sx^2 I),
u_t ~ N(0, alpha_T(t) su^2 I) with alpha_T(t) = 1/T and beta_T(k) = sum_{j<T-k} alpha_T(j).
Finite-sample forms are those of Remark empirical_losses.  Every term is quadratic in
the data, so they are evaluated exactly from the data's second moments (``moments``,
``emp_terms``): a gradient step costs the same for N = 10 and N = 10^4.  The moment
form cancels "square - 2 cross + square", which loses precision when L^t grows along a
latent direction the decoder ignores; ``emp_terms_direct`` recomputes the same terms
from per-trajectory residuals, and every reported finite-sample objective is checked
against it.

Regularizers R(E) (base measure in parentheses):
    orth        ||I_m - E E^T||_F^2                                       (weights)
    vicreg_x0   R_vr(E; mu, beta), C = Cov_mu(E x)                        (x_0 ~ N(0, sx^2 I))
    sigreg_x0   R_sr(E; mu), sliced Epps-Pulley                           (x_0 ~ N(0, sx^2 I))
    vicreg      as vicreg_x0                                              (all trajectory states)
    sigreg      as sigreg_x0                                              (all trajectory states)
With the x_0 measure and beta = 1, vicreg_x0 = orth / m (sx = 1).  For a linear encoder on
Gaussian data every projection a^T E x is Gaussian (a Gaussian mixture over t for the
trajectory measure) and SIGReg has a closed form:
    g(s) = sqrt(2 pi) [ (1+2s)^-1/2 - 2 (2+s)^-1/2 ] + sqrt(2 pi/3),   s = a^T Cov a.
"""

from __future__ import annotations

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

REGS = ("orth", "vicreg_x0", "sigreg_x0", "vicreg", "sigreg")
SQ2PI = float(np.sqrt(2.0 * np.pi))
SIG_C0 = float(np.sqrt(2.0 * np.pi / 3.0))


def alpha_beta(T):
    alpha = np.full(T, 1.0 / T)
    beta = np.array([alpha[:T - k].sum() for k in range(T)])
    return alpha, beta


def sphere_dirs(m, k=128, seed=0):
    """Deterministic directions for the population expectation over S^{m-1}."""
    if m == 1:
        return np.ones((1, 1))
    if m == 2:
        th = np.pi * (np.arange(64) + 0.5) / 64
        return np.stack([np.cos(th), np.sin(th)], axis=1)
    V = np.random.default_rng(seed).standard_normal((k, m))
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def sumsq(X):
    return jnp.sum(X * X)


def mat_powers(L, T):
    """(L^0, ..., L^T) stacked."""
    P = [jnp.eye(L.shape[0], dtype=L.dtype)]
    for _ in range(T):
        P.append(P[-1] @ L)
    return jnp.stack(P)


def sigreg_g(s):
    return SQ2PI * (1.0 / jnp.sqrt(1.0 + 2.0 * s) - 2.0 / jnp.sqrt(2.0 + s)) + SIG_C0


def sigreg_mixture(s, pi):
    """s (K, nt): variances along each direction for each mixture component."""
    t1 = jnp.einsum("t,u,ktu->k", pi, pi, SQ2PI / jnp.sqrt(1.0 + s[:, :, None] + s[:, None, :]))
    t2 = jnp.einsum("t,kt->k", pi, SQ2PI / jnp.sqrt(2.0 + s))
    return jnp.mean(t1 - 2.0 * t2 + SIG_C0)


def vicreg_from_cov(C, beta):
    d = jnp.diag(C)
    off = C - jnp.diag(d)
    return (jnp.sum((1.0 - d) ** 2) + beta * jnp.sum(off ** 2)) / C.shape[0]


# -----------------------------------------------------------------------------------------
# Population
# -----------------------------------------------------------------------------------------
class PopLoss:
    """Population L_bwd^(T) or L_fwd^(T) of one system, horizon and regularizer.

    ``terms(th, c)`` returns every data term (J_bwd only for loss = 'bwd') and R;
    ``total(th, lam_reg, c)`` the objective; th = (E, D, L, H), c = the constants ``self.c``
    (A^t, A^k B, Cov(x_t), ...), passed as an argument so that XLA does not constant-fold them."""

    def __init__(self, A, B, m, T, loss, reg, sx=1.0, su=0.5, lam_fwd=1.0, fwd_horizon="T", vic_beta=1.0):
        assert loss in ("bwd", "fwd") and reg in REGS and fwd_horizon in ("T", 1, "1")
        A, B = np.asarray(A, float), np.asarray(B, float)
        n, p = B.shape
        self.n, self.p, self.m, self.T, self.loss, self.reg = n, p, m, T, loss, reg
        h = T if fwd_horizon == "T" else 1
        alpha, beta = alpha_beta(T)
        Ap = [np.eye(n)]
        for _ in range(T):
            Ap.append(Ap[-1] @ A)
        Ap = np.stack(Ap)
        AkB = np.stack([Ap[k] @ B for k in range(T)])
        Sig = [sx ** 2 * np.eye(n)]                                # Cov(x_t), t = 0..T
        for t in range(1, T + 1):
            Sig.append(A @ Sig[-1] @ A.T + su ** 2 * alpha[t - 1] * (B @ B.T))
        Sig = np.stack(Sig)
        self.Sig = Sig
        self.c = {k: jnp.asarray(v) for k, v in dict(Ap=Ap, AkB=AkB, Sig=Sig, A=A, B=B, dirs=sphere_dirs(m),
                                                     pi=np.full(T + 1, 1.0 / (T + 1))).items()}
        wb_s, wb_i = sx ** 2 / (T + 1), jnp.asarray(su ** 2 * beta / (T + 1))
        wf_s, wf_i = sx ** 2 / T, jnp.asarray(su ** 2 * beta / T)
        a0 = su ** 2 * alpha[0]
        Im = jnp.eye(m)

        def terms(th, c=None):
            c = self.c if c is None else c
            E, D, L, H = th
            Lp = mat_powers(L, T)                                   # (T+1, m, m)
            LH = Lp[:T] @ H                                          # (T, m, p)
            out = {}
            Fs = E @ c["Ap"][1:] - Lp[1:] @ E                        # (T, m, n)
            Fi = E @ c["AkB"] - LH
            out["J_fwd"] = wf_s * sumsq(Fs) + jnp.sum(wf_i * jnp.sum(Fi * Fi, axis=(1, 2)))
            out["J_fwd1"] = sx ** 2 * sumsq(E @ c["A"] - L @ E) + a0 * sumsq(E @ c["B"] - H)
            if loss == "bwd":
                Rs = c["Ap"] - D @ (Lp @ E)                          # (T+1, n, n)
                Ri = c["AkB"] - D @ LH
                out["J_bwd"] = wb_s * sumsq(Rs) + jnp.sum(wb_i * jnp.sum(Ri * Ri, axis=(1, 2)))
            out["R"] = pop_reg(reg, E, c, sx, vic_beta, Im)
            return out

        def total(th, lam_reg, c=None):
            q = terms(th, c)
            if loss == "bwd":
                return q["J_bwd"] + lam_fwd * (q["J_fwd"] if h == T else q["J_fwd1"]) + lam_reg * q["R"]
            return q["J_fwd"] + lam_reg * q["R"]

        self.terms, self.total = terms, total


def pop_reg(reg, E, c, sx, vic_beta, Im):
    if reg == "orth":
        return sumsq(Im - E @ E.T)
    if reg == "vicreg_x0":
        return vicreg_from_cov(sx ** 2 * E @ E.T, vic_beta)
    if reg == "vicreg":
        return vicreg_from_cov(E @ c["Sig"].mean(0) @ E.T, vic_beta)
    if reg == "sigreg_x0":
        return jnp.mean(sigreg_g(sx ** 2 * jnp.sum((c["dirs"] @ E) ** 2, axis=1)))
    if reg == "sigreg":
        St = jnp.einsum("ij,tjk,lk->til", E, c["Sig"], E)           # E Sig_t E^T
        s = jnp.einsum("ka,tab,kb->kt", c["dirs"], St, c["dirs"])
        return sigreg_mixture(s, c["pi"])
    raise ValueError(reg)


# -----------------------------------------------------------------------------------------
# Finite sample
# -----------------------------------------------------------------------------------------
def lag_index(T):
    """[t, j] -> source time t-1-j of the input u_{t-1-j} and its validity (j < t)."""
    k_src = np.arange(T + 1)[:, None] - 1 - np.arange(T)[None, :]
    return np.clip(k_src, 0, max(T - 1, 0)), (k_src >= 0).astype(float)


def lag_inputs(U):
    """U (..., N, T, p) -> (..., N, T+1, T, p) with entry [t, j] = u_{t-1-j} (0 for j >= t)."""
    k_src, valid = lag_index(U.shape[-2])
    return U[..., k_src, :] * valid[:, :, None]


def moments(X, U):
    """Second moments of trajectories X (N, T+1, n), U (N, T, p) used by ``emp_terms``.

    With z_hat_t = L^t z_0 + sum_j G_j u_{t-1-j}, G_j = L^j H, every cross moment of the
    losses is a contraction of the lag-indexed moments
        Cx0[t] = E x_t x_0^T,  Cxx[t] = E x_t x_t^T,  Cxu[t, j] = E x_t u_{t-1-j}^T,
        C0u[t, j] = E x_0 u_{t-1-j}^T,  Cuu[t, j, k] = E u_{t-1-j} u_{t-1-k}^T,
    plus sx[t] = E|x_t|^2 and the first moments VICReg centers with."""
    N = X.shape[0]
    Us = lag_inputs(U)
    return dict(Cx0=np.einsum("isa,ib->sab", X, X[:, 0]) / N,
                Cxx=np.einsum("isa,isb->sab", X, X) / N,
                Cxu=np.einsum("ita,itjb->tjab", X, Us) / N,
                C0u=np.einsum("ia,itjb->tjab", X[:, 0], Us) / N,
                Cuu=np.einsum("itja,itkb->tjkab", Us, Us) / N,
                sx=np.einsum("isa,isa->s", X, X) / N,
                xbar=X.mean(0), xbar_pool=X.mean((0, 1)))


def emp_terms(th, Mo, T):
    """Exact empirical J_bwd^(T), J_fwd^(T), J_fwd^(1) from the moments (batched contractions,
    no recursion):  Y_t = E x_t zhat_t^T,  Z_t = E zhat_t zhat_t^T."""
    E, D, L, H = th
    Lp = mat_powers(L, T)
    G = Lp[:T] @ H
    LpT = jnp.swapaxes(Lp, 1, 2)
    Y = (Mo["Cx0"] @ E.T) @ LpT + jnp.einsum("tjnp,jmp->tnm", Mo["Cxu"], G)
    S0 = E @ Mo["Cxx"][0] @ E.T
    cross = Lp @ jnp.einsum("mn,tjnp,jqp->tmq", E, Mo["C0u"], G)
    Zin = jnp.einsum("jma,tjaq->tmq", G, jnp.einsum("tjkab,kmb->tjam", Mo["Cuu"], G))
    Z = Lp @ S0 @ LpT + cross + jnp.swapaxes(cross, 1, 2) + Zin
    bwd_t = Mo["sx"] - 2.0 * jnp.einsum("ij,tij->t", D, Y) + jnp.einsum("ij,tjk,ik->t", D, Z, D)
    ExE = jnp.einsum("ij,tjk,lk->til", E, Mo["Cxx"], E)
    fwd_t = (jnp.trace(ExE, axis1=1, axis2=2) - 2.0 * jnp.einsum("ij,tji->t", E, Y)
             + jnp.trace(Z, axis1=1, axis2=2))
    return dict(J_bwd=jnp.mean(bwd_t), J_fwd=jnp.mean(fwd_t[1:]), J_fwd1=fwd_t[1])


def emp_terms_direct(th, X, Ul, w):
    """The terms of ``emp_terms`` from per-trajectory residuals (sums of squares: no
    cancellation).  X (N, T+1, n), Ul = lag_inputs(U), w (N,) sample weights."""
    E, D, L, H = th
    T = X.shape[1] - 1
    Lp = mat_powers(L, T)
    G = Lp[:T] @ H
    Z = jnp.einsum("tqm,im->itq", Lp, X[:, 0] @ E.T) + jnp.einsum("itjp,jmp->itm", Ul, G)
    bwd_t = jnp.einsum("i,itn->t", w, (X - Z @ D.T) ** 2)
    fwd_t = jnp.einsum("i,itm->t", w, (X @ E.T - Z) ** 2)
    return dict(J_bwd=jnp.mean(bwd_t), J_fwd=jnp.mean(fwd_t[1:]), J_fwd1=fwd_t[1])


def emp_reg(reg, E, Mo, key, Xs, N, T, vic_beta=1.0, sig_batch=512, sig_dirs=16,
            knots=np.linspace(0.0, 5.0, 21)):
    """Empirical regularizer.  Xs (N, T+1, n) raw states, read only by SIGReg, which uses
    the empirical characteristic function of a minibatch of ``sig_batch`` embedded states
    (fixed by ``key``, so the objective is deterministic and L-BFGS applies)."""
    m = E.shape[0]
    if reg == "orth":
        return sumsq(jnp.eye(m) - E @ E.T)
    if reg in ("vicreg", "vicreg_x0"):
        if reg == "vicreg_x0":
            mu, C = E @ Mo["xbar"][0], E @ Mo["Cxx"][0] @ E.T
        else:
            mu, C = E @ Mo["xbar_pool"], E @ Mo["Cxx"].mean(0) @ E.T
        return vicreg_from_cov(C - jnp.outer(mu, mu), vic_beta)
    if reg in ("sigreg", "sigreg_x0"):
        k1, k2, k3 = jax.random.split(key, 3)
        i = jax.random.randint(k1, (sig_batch,), 0, N)
        t = jax.random.randint(k2, (sig_batch,), 0, T + 1) if reg == "sigreg" else jnp.zeros(sig_batch, int)
        z = Xs[i, t] @ E.T
        a = jnp.ones((1, 1)) if m == 1 else jax.random.normal(k3, (sig_dirs, m))
        a = a / jnp.linalg.norm(a, axis=1, keepdims=True)
        return epps_pulley(z, a, knots)
    raise ValueError(reg)


def epps_pulley(z, a, knots=np.linspace(0.0, 5.0, 21)):
    """Sliced Epps-Pulley statistic of the samples z (k, m) along the unit directions a (d, m): the
    weighted distance of the empirical characteristic function to that of N(0, 1) on ``knots``
    (trapezoid rule), averaged over the directions (also the MLP SIGReg, latentctl.mlp)."""
    kn = jnp.asarray(knots, z.dtype)
    arg = (z @ a.T)[:, :, None] * kn[None, None, :]
    g = jnp.exp(-0.5 * kn ** 2)
    err = ((jnp.mean(jnp.cos(arg), 0) - g) ** 2 + jnp.mean(jnp.sin(arg), 0) ** 2) * g
    h = float(knots[1] - knots[0])
    return jnp.mean(2.0 * h * (jnp.sum(err, axis=1) - 0.5 * (err[:, 0] + err[:, -1])))


class EmpLoss:
    """Empirical L_bwd^(T) / L_fwd^(T) (Remark empirical_losses) over a bank of datasets.

    Xs (S, Nmax, T+1, n), Us (S, Nmax, T, p): S datasets.  A *problem* is (dataset index
    sidx, number of trajectories N): it uses the first N trajectories of dataset sidx, so
    sample-size comparisons share random numbers.  ``aux(sidx, N)`` builds the per-problem
    data (moments, sample weights, SIGReg key), which can be stacked and vmapped over;
    ``data`` holds the raw trajectories, shared by all problems.  Both are passed as
    arguments (never captured as constants: XLA would try to constant-fold them).

    objective(th, lam_reg, aux, data)          moment form (fast; what is optimized)
    objective_direct(th, lam_reg, aux, data)   per-trajectory residuals (exact; for reporting)
    terms_direct(th, aux, data)                the data terms and R, exact"""

    def __init__(self, Xs, Us, m, loss, reg, lam_fwd=1.0, fwd_horizon="T", vic_beta=1.0):
        assert loss in ("bwd", "fwd") and reg in REGS
        self.Xs_np, self.Us_np = Xs, Us
        S, Nmax, T1, n = Xs.shape
        T = T1 - 1
        self.T, self.n, self.m, self.loss, self.reg = T, n, m, loss, reg
        h = T if fwd_horizon == "T" else 1
        self.data = dict(X=jnp.asarray(Xs), Ul=jnp.asarray(lag_inputs(Us)))

        def combine(q, R, lam_reg):
            if loss == "bwd":
                return q["J_bwd"] + lam_fwd * (q["J_fwd"] if h == T else q["J_fwd1"]) + lam_reg * R
            return q["J_fwd"] + lam_reg * R

        def reg_term(th, aux, data):
            return emp_reg(reg, th[0], aux["Mo"], aux["key"], data["X"][aux["sidx"]], aux["N"], T, vic_beta)

        def objective(th, lam_reg, aux, data):
            return combine(emp_terms(th, aux["Mo"], T), reg_term(th, aux, data), lam_reg)

        def terms_direct(th, aux, data):
            q = emp_terms_direct(th, data["X"][aux["sidx"]], data["Ul"][aux["sidx"]], aux["w"])
            return dict(q, R=reg_term(th, aux, data))

        def objective_direct(th, lam_reg, aux, data):
            q = terms_direct(th, aux, data)
            return combine(q, q["R"], lam_reg)

        self.objective, self.terms_direct, self.objective_direct = objective, terms_direct, objective_direct

    def aux(self, sidx, N, key=None):
        X, U = self.Xs_np[sidx, :N], self.Us_np[sidx, :N]
        Nmax = self.Xs_np.shape[1]
        return dict(Mo={k: jnp.asarray(v) for k, v in moments(X, U).items()},
                    w=jnp.asarray((np.arange(Nmax) < N) / N), N=N, sidx=sidx,
                    key=jax.random.PRNGKey(7 + 1000 * sidx + N) if key is None else key)

    def aux_batch(self, problems):
        """Stack aux over problems [(sidx, N), ...] for vmapping."""
        auxs = [self.aux(s, N) for s, N in problems]
        return jax.tree_util.tree_map(lambda *a: jnp.stack([jnp.asarray(x) for x in a]), *auxs)


def growth_rate(X):
    """Per-step growth of the data energy over the last step, (E|x_T|^2 / E|x_{T-1}|^2)^1/2,
    for trajectories X (N, T+1, n): an estimate of rho(A) available before training (the
    rho-hat of Sec. 6.4)."""
    e = np.mean(np.sum(X[:, -2:] ** 2, axis=-1), axis=0)
    return float(np.sqrt(e[1] / e[0]))
