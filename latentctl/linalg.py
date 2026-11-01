"""Model-based constructions of the paper: Algorithm 1, canonical variables, subspaces.

select_eigs        the eigenvalue multiset Lambda of Algorithm 1 (sigma_+(A) first, then the
                   slowest stable modes, conjugate pairs kept together)
alg1_latent        (E, L, H) of Algorithm 1: E = Q_1^T (orthonormal rows), L = R_11^T, H = E B
canonical          canonical variables (E_Lambda, D_Lambda, L_Lambda, H_Lambda) of
                   Def. canonical_variables: oblique encoder, D = Q_+
unstable_left_basis  orthonormal basis of G_+^R(A^T)
subspace_distance  Delta(G; row E) = ||P_row(E)^perp Q_G||_2 or _F (the paper's Delta is _F)
conjugacy          forward-conjugacy residuals of a linear latent model (E, L, H) of (A, B)
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

TOL = 1e-6


def select_eigs(ev, m):
    """Indices of Algorithm 1's Lambda: unstable/marginal eigenvalues, then stable conjugate
    pairs and real eigenvalues in decreasing modulus.  None if no compatible choice exists."""
    idx = list(range(len(ev)))
    plus = [i for i in idx if abs(ev[i]) >= 1.0 - TOL]
    rest = sorted([i for i in idx if i not in plus], key=lambda i: -abs(ev[i]))
    sel = list(plus)
    if len(sel) > m:
        return None
    for i in [i for i in rest if ev[i].imag > TOL]:
        if len(sel) + 2 > m:
            break
        j = min((k for k in rest if k not in sel and k != i), key=lambda k: abs(ev[k] - np.conj(ev[i])))
        sel += [i, j]
    reals = [i for i in rest if abs(ev[i].imag) <= TOL and i not in sel]
    while len(sel) < m and reals:
        sel.append(reals.pop(0))
    return sel if len(sel) == m else None


def ordered_schur(M, targets):
    """Real Schur form of M with the eigenvalues ``targets`` in the leading block."""
    targets = np.asarray(targets)

    def pick(x, y=None):
        z = x if y is None else x + 1j * y
        return bool(np.min(np.abs(targets - z)) < TOL * max(1.0, abs(z)))

    return sla.schur(M, output="real", sort=pick)


def alg1_latent(A, B, m):
    """(E, L, H) of Algorithm 1 (E with orthonormal rows)."""
    ev = np.linalg.eigvals(A)
    sel = select_eigs(ev, m)
    if sel is None:
        raise ValueError("latent dimension incompatible with the spectrum")
    T, Z, sdim = ordered_schur(A.T, ev[sel])
    if sdim != m:
        raise RuntimeError(f"ordered Schur selected {sdim} != {m} eigenvalues")
    E = Z[:, :m].T
    return E, T[:m, :m].T, E @ B


def canonical(A, B, sel):
    """Canonical variables for the eigenvalues ev[sel] (Def. canonical_variables):
    E = Q_+^T + X Q_-^T (oblique), D = Q_+, L = R_++, H = E B."""
    ev = np.linalg.eigvals(A)
    T, Z, d = ordered_schur(A, ev[sel])
    Qp, Qm = Z[:, :d], Z[:, d:]
    X = sla.solve_sylvester(T[:d, :d], -T[d:, d:], T[:d, d:])
    E = Qp.T + X @ Qm.T
    return dict(E=E, D=Qp, L=T[:d, :d], H=E @ B, eigs=ev[sel])


def canonical_targets(A, B, m):
    """Targets used by the diagnostics: canonical variables for Algorithm 1's Lambda ('ell')
    and for sigma_+(A) ('0'), and an orthonormal basis of G_+^R(A^T) ('Q_plus')."""
    ev = np.linalg.eigvals(A)
    plus = [i for i in range(len(ev)) if abs(ev[i]) >= 1.0 - TOL]
    out = {"ell": canonical(A, B, select_eigs(ev, m)), "0": canonical(A, B, plus)}
    out["Q_plus"] = unstable_left_basis(A)
    out["eigs_A"] = ev
    return out


def unstable_left_basis(A):
    """Orthonormal basis (n x m_+) of the real left unstable subspace G_+^R(A^T)."""
    ev = np.linalg.eigvals(A)
    plus = ev[np.abs(ev) >= 1.0 - TOL]
    T, Z, sdim = ordered_schur(A.T, plus)
    assert sdim == len(plus), (sdim, len(plus))
    return Z[:, :sdim]


def row_basis(E, rtol=1e-10):
    U, s, _ = np.linalg.svd(np.asarray(E).T, full_matrices=False)
    return U[:, s > rtol * max(s[0], 1e-300)]


def subspace_distance(Q, E, ord=2):
    """Delta(span Q; row E) = ||(I - P_row(E)) Q|| for Q with orthonormal columns: ord = 2 gives
    Delta_op, ord = 'fro' the paper's Delta_F (equal when span Q is one-dimensional)."""
    V = row_basis(E)
    return float(np.linalg.norm(Q - V @ (V.T @ Q), ord))


def normalize_gauge(E, L, H):
    """Gauge with orthonormal encoder rows: E^ = S^-1 E, L^ = S^-1 L S, H^ = S^-1 H, S = (EE^T)^1/2."""
    w, U = np.linalg.eigh(E @ E.T)
    if w.min() <= 1e-300:
        raise ValueError("rank-deficient encoder")
    S, Si = (U * np.sqrt(w)) @ U.T, (U / np.sqrt(w)) @ U.T
    return Si @ E, Si @ L @ S, Si @ H


def conjugacy(E, L, H, A, B, lam):
    """Forward-conjugacy residuals of (E, L, H) as a latent model of (A, B):
        raw_A  ||E A - L E||_F (depends on the latent coordinates: z -> S z multiplies it by S)
        res_A  ||E^ A - L^ E^||_F and res_B ||E^ B - H^||_F with orthonormal encoder rows (normalize_gauge)
        r_u    ||e^T A - mu e^T|| for the eigenvalue mu of L nearest lam, its left eigenvector w scaled so that
               e = E^T w has unit norm, and r_h = |e^T B - w^T H| (both invariant under z -> S z; r_u = 0 iff e
               is a left eigenvector of A)."""
    E, L, H = (np.asarray(x, float) for x in (E, L, H))
    nan = dict(raw_A=np.nan, res_A=np.nan, res_B=np.nan, r_u=np.nan, r_h=np.nan)
    if not all(np.all(np.isfinite(x)) for x in (E, L, H)):
        return nan
    out = dict(nan, raw_A=float(np.linalg.norm(E @ A - L @ E)))
    try:
        Eh, Lh, Hh = normalize_gauge(E, L, H)
    except ValueError:
        return out
    out.update(res_A=float(np.linalg.norm(Eh @ A - Lh @ Eh)), res_B=float(np.linalg.norm(Eh @ B - Hh)))
    mu, W = sla.eig(L.T)
    i = int(np.argmin(np.abs(mu - lam)))
    w = W[:, i] / np.linalg.norm(E.T @ W[:, i])
    e = E.T @ w
    out.update(r_u=float(np.linalg.norm(e @ A - mu[i] * e)), r_h=float(np.linalg.norm(e @ B - w @ H)))
    return out


def spectral_radius(M):
    return float(np.max(np.abs(np.linalg.eigvals(M))))
