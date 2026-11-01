"""Plants of the paper: the flexible arm (Secs. 1.1 and 6) and the 2x2 non-normal example (Sec. 5.2).

FlexibleArm
    Planar chain of n_sec rigid sections joined by torsional springs, section angles
    phi = S q (paper eq. arm_dynamics),

        M(q) qdd + C(q, qd) qd + (c0 M(q) + c1 K_eff) qd + kappa K0 q - S^T (w .* sin(S q)) = 1 u,
        K_eff = kappa K0 - S^T diag(w) S,

    with the configuration-dependent inertia of the chain and its centrifugal terms,

        M(q) = S^T M_phi(S q) S,   M_phi(phi)_jk = N_jk cos(phi_j - phi_k) + delta_jk I_j,
        C(q, qd) qd = S^T c,       c_j = sum_k N_jk sin(phi_j - phi_k) phid_k^2,

    N_jk = sum_i m_i a_ij a_ik (a_ij = lever arm of section j on the centre of mass of
    section i), I_j = m_j l_j^2 / 12; c0 M(q) is viscous drag on every mass element.
    Linearized about the upright equilibrium (M = M(0), C = 0) and discretized by
    zero-order hold (dt = 0.05 s).  The input column is normalized, ||B|| = 1 (physical
    torque = u / u_scale).  The stiffness scale kappa sits a fraction ``alpha_frac`` of
    the way between two consecutive generalized eigenvalues of (S^T diag(w) S, K0), so
    that exactly ``n_unstable`` stiffness directions are negative (gravity buckles that
    many bending modes).  Defaults give the arm of the motivating example: one unstable
    eigenvalue 1.129, slowest stable 0.874 and 0.581 +- 0.633j.

    ``unstable_eigs`` overrides the modal stiffness of the buckling modes so that
    they grow at prescribed discrete rates (the three-mode arm of Sec. 6.4 uses
    1.15 / 1.12 / 1.09); every other mode, the mass, the geometry and the Rayleigh
    damping are unchanged.  ``coords = 'mass'`` uses x = (M^1/2 q, M^1/2 qd), in which
    left eigenvectors of different physical modes are orthogonal.

    The nonlinear plant (joint coordinates only) is advanced over one ZOH period by an
    exponential integrator in momentum form with the linear part exact (_step_exact).

nonnormal
    A_eta = [[lam_u, eta], [0, lam_s]], B = [b1, b2]^T (paper eq. nonnormal_example).
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla


def zoh(Ac, Bc, dt):
    """Exact zero-order-hold discretization via the augmented matrix exponential."""
    n, p = Bc.shape
    Z = np.zeros((n + p, n + p))
    Z[:n, :n], Z[:n, n:] = Ac, Bc
    Phi = sla.expm(Z * dt)
    return Phi[:n, :n], Phi[:n, n:]


class FlexibleArm:
    def __init__(self, n_sec=20, length=1.0, total_mass=0.8, gravity=9.81, c0=0.4, c1=0.02,
                 dt=0.05, n_unstable=1, alpha_frac=0.5, unstable_eigs=None, coords="joint",
                 substeps=50, u_max=40.0):
        """substeps per ZOH period of the nonlinear step (one-step error vs a Radau reference <~ 1e-4
        on falling arms, ~1e-6 typical)."""
        assert coords in ("joint", "mass")
        self.n_sec, self.dt, self.coords, self.u_max = n_sec, dt, coords, u_max
        ell = np.full(n_sec, length / n_sec)
        mass = np.full(n_sec, total_mass / n_sec)
        S = np.tril(np.ones((n_sec, n_sec)))                    # section angles phi = S q
        tail = np.cumsum(mass[::-1])[::-1] - mass
        w = gravity * ell * (0.5 * mass + tail)                 # gravity weights
        G = S.T @ np.diag(w) @ S                                # negative stiffness
        K0 = np.diag(np.linspace(1.0, 1.5, n_sec))
        mu = sla.eigh(G, K0, eigvals_only=True)                 # ascending buckling values
        k = n_sec - n_unstable - 1
        kappa = mu[k] + alpha_frac * (mu[k + 1] - mu[k])
        K_eff = kappa * K0 - G
        M = np.zeros((n_sec, n_sec))                            # upright inertia M(0)
        for i in range(n_sec):
            Jx = ell[:i] @ S[:i, :] + 0.5 * ell[i] * S[i, :]
            M += mass[i] * np.outer(Jx, Jx) + mass[i] * ell[i] ** 2 / 12.0 * np.outer(S[i], S[i])
        lever = np.tril(np.ones((n_sec, n_sec)), -1) * ell[None, :] + np.diag(0.5 * ell)   # a_ij
        Nphi = lever.T @ (mass[:, None] * lever)                # N_jk = sum_i m_i a_ij a_ik
        Mphi0 = Nphi + np.diag(mass * ell ** 2 / 12.0)          # M_phi(0), M(0) = S^T M_phi(0) S
        if unstable_eigs is not None:
            nu, V = sla.eigh(K_eff, M)                          # M-orthonormal undamped modes
            assert np.sum(nu < 0) == n_unstable
            s = np.log(np.sort(np.asarray(unstable_eigs, float))[::-1]) / dt
            nu = nu.copy()                                      # s^2 + (c0 + c1 nu) s + nu = 0
            nu[:n_unstable] = -(s ** 2 + c0 * s) / (1.0 + c1 * s)
            K_eff = M @ V @ np.diag(nu) @ V.T @ M
            K_eff = 0.5 * (K_eff + K_eff.T)
        Minv = np.linalg.inv(M)
        Dmp = c0 * M + c1 * K_eff
        Z, I = np.zeros((n_sec, n_sec)), np.eye(n_sec)
        Ac = np.block([[Z, I], [-Minv @ K_eff, -Minv @ Dmp]])
        Bc = np.vstack([np.zeros((n_sec, 1)), (Minv @ np.ones(n_sec))[:, None]])
        A, B = zoh(Ac, Bc, dt)
        if coords == "mass":
            ev, U = np.linalg.eigh(M)
            Tm = np.kron(np.eye(2), (U * np.sqrt(ev)) @ U.T)
            A, B = Tm @ A @ np.linalg.inv(Tm), Tm @ B
        self.u_scale = float(np.linalg.norm(B))
        B = B / self.u_scale
        self.A, self.B, self.Ac, self.Bc = A, B, Ac, Bc
        self.M, self.Minv, self.K_eff, self.S, self.w, self.G, self.ell = M, Minv, K_eff, S, w, G, ell
        self.kappa, self.buckling = kappa, mu
        self.nu, self.V = sla.eigh(K_eff, M)                     # nu[0] < 0: buckling mode
        self.c_tip = ell @ S                                     # d tip_x / dq at q = 0
        self.Nphi, self.Mphi0, self.c0, self.c1 = Nphi, Mphi0, c0, c1
        self._Sinv, self._MinvST = np.linalg.inv(S), Minv @ S.T
        h = dt / substeps
        self._Eh = sla.expm(Ac * h)
        self._Ph = np.linalg.solve(Ac, self._Eh - np.eye(2 * n_sec))
        self._Wm = self._Ph / h
        self._substeps = substeps

    @property
    def n(self):
        return self.A.shape[0]

    @property
    def p(self):
        return self.B.shape[1]

    # ---- nonlinear plant (joint coordinates) ----------------------------------------
    def _dM_phi(self, phi):
        """M_phi(phi) - M_phi(0) = -2 N .* sin^2((phi_j - phi_k) / 2) for phi (k, n_sec)."""
        sh, ch = np.sin(0.5 * phi), np.cos(0.5 * phi)
        d = sh[:, :, None] * ch[:, None, :] - ch[:, :, None] * sh[:, None, :]      # sin((phi_j - phi_k) / 2)
        return -2.0 * self.Nphi * d * d

    def _step_exact(self, X, U, n_iter=3):
        """One ZOH period of the nonlinear plant.

        Momentum form: with m = (M(q) - M(0)) qd, the plant reads
            M(0) (qdd + c0 qd) = F_lin + rho - dm/dt,   rho = S^T (w .* (sin phi - phi) + dT/dphi)
                                                             - c0 (M(q) - M(0)) qd,
        F_lin the force of the linearization, T the kinetic energy.  Per substep h,
            x1 = E x0 + P_h B u + P_h (rho~0 + rho~1) / 2 - P_h / h (m~1 - m~0),
        E = e^(Ac h), P_h = int_0^h e^(Ac s) ds, y~ = (0, M(0)^-1 y): the variation-of-constants
        formula with rho and dm/dt constant over the substep.  A transient of the stiff
        (overdamped) bending modes thus enters the slow modes as the momentum change it causes
        (P_h / h ~ I there), not as its acceleration (an explicit residual weighs that by
        h |lambda_fast| ~ 300).  x1 is implicit: n_iter passes, each solving exactly for qd1
        given q1 (the m term is linear in qd1) and refreshing q1 and rho~1."""
        n, S, N = self.n_sec, self.S, self.Nphi
        Eh, Ph, W = self._Eh, self._Ph[:, n:], self._Wm[:, n:]
        MS = self._MinvST                                                             # M(0)^-1 S^T
        WMS = W[n:] @ MS

        def parts(X):
            phi, phid = X[:, :n] @ S.T, X[:, n:] @ S.T
            sn, cs = np.sin(phi), np.cos(phi)
            dM = self._dM_phi(phi)
            dMv = (dM @ phid[..., None])[..., 0]
            dT = -phid * (sn * ((cs * phid) @ N) - cs * ((sn * phid) @ N))         # dT/dphi
            return dM, (self.w * (sn - phi) + dT - self.c0 * dMv) @ MS.T, dMv @ MS.T  # dM_phi, rho~_v, m~_v

        Bu = (U / self.u_scale) @ self.Bc.T
        k, I = X.shape[0], np.eye(n)
        for _ in range(self._substeps):
            _, rho0, m0 = parts(X)
            lin = X @ Eh.T + Bu @ self._Ph.T
            b0 = lin + 0.5 * rho0 @ Ph.T + m0 @ W.T
            X1 = lin + rho0 @ Ph.T                                                    # predictor
            for _ in range(n_iter):
                dM, rho1, _ = parts(X1)
                b = b0 + 0.5 * rho1 @ Ph.T
                dMS = (dM.reshape(k * n, n) @ S).reshape(k, n, n)                     # dM_phi S (one GEMM)
                C = (dMS.transpose(0, 2, 1).reshape(k * n, n) @ WMS.T).reshape(k, n, n).transpose(0, 2, 1)
                qd1 = np.linalg.solve(I + C, b[:, n:, None])[..., 0]
                m1 = (dM @ (qd1 @ S.T)[..., None])[..., 0] @ MS.T
                X1 = np.concatenate([b[:, :n] - m1 @ W[:n].T, qd1], axis=1)
            X = X1
        return X

    def step(self, X, U, plant="nonlinear"):
        """One ZOH period for a batch: X (k, n), U (k, p) in normalized input units."""
        if plant == "linear":
            return X @ self.A.T + U @ self.B.T
        assert self.coords == "joint"
        return self._step_exact(X, U)

    # ---- physical quantities ----------------------------------------------------------
    def centerline(self, q):
        phi = self.S @ q
        return (np.concatenate([[0.0], np.cumsum(self.ell * np.sin(phi))]),
                np.concatenate([[0.0], np.cumsum(self.ell * np.cos(phi))]))

    def fallen(self, X):
        """Some section more than 90 degrees from upright."""
        return np.abs(X[..., :self.n_sec] @ self.S.T).max(axis=-1) > np.pi / 2

    def test_ics(self, k, tip=0.10, seed=0):
        """k ICs released from rest: the tip pushed ``tip`` along the buckling mode (random
        sign), plus a random bend that leaves the (linearized) tip in place and whose
        section-angle profile S r has 30% of the norm of the buckling displacement's."""
        rng = np.random.default_rng(seed)
        v1 = self.V[:, 0] / abs(self.c_tip @ self.V[:, 0])
        q = rng.choice([-1.0, 1.0], size=(k, 1)) * tip * v1[None, :]
        r = rng.standard_normal((k, self.n_sec))
        r = r - np.outer(r @ self.c_tip, self.c_tip) / (self.c_tip @ self.c_tip)
        r = r * (0.3 * tip * np.linalg.norm(self.S @ v1) / np.linalg.norm(r @ self.S.T, axis=1))[:, None]
        return np.concatenate([q + r, np.zeros((k, self.n_sec))], axis=1)


def flexible_arm(**kw):
    """The arm of the motivating example: n = 40, one unstable eigenvalue (1.129)."""
    return FlexibleArm(**kw)


def flexible_arm_3modes(**kw):
    """The arm of Sec. 6.4: three buckling modes growing at 1.15 / 1.12 / 1.09, mass coordinates."""
    kw = dict(dict(n_unstable=3, alpha_frac=0.9, unstable_eigs=(1.15, 1.12, 1.09), coords="mass"), **kw)
    return FlexibleArm(**kw)


def nonnormal(eta, lam_u=1.5, lam_s=0.5, b1=1.0, b2=1.0):
    """(A_eta, B) of paper eq. nonnormal_example."""
    return np.array([[lam_u, eta], [0.0, lam_s]]), np.array([[b1], [b2]])
