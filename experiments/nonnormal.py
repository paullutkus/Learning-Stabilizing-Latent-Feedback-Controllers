"""Insufficiency of the single-step backwards loss on the 2x2 non-normal example (Sec. 5.2).

System (paper eq. nonnormal_example): A_eta = [[lam_u, eta], [0, lam_s]], B = [b1, b2]^T with
lam_u = 1.5, lam_s = 0.5, B = [1, 1]^T, latent dimension m = 1.

Loss: the total backwards loss (eq. total_backwards_loss), population form,

    L_bwd^(T) = J_bwd^(T)(E, D, L, H) + lam_fwd J_fwd^(h)(E, L, H) + lam_reg R(E),
    R(E) = ||I - E E^T||_F^2 = (1 - |E|^2)^2,

with sx = su = 1, alpha_T(k) = 1/T, lam_fwd = 1, lam_reg = 100, and the one-step forward term
h = 1 (``fwd_horizon``; "T" gives J_fwd^(T); both coincide at T = 1).  lam_reg must be large
here: the loss of a non-degenerate model grows linearly in eta (the reference point in the
proof of Thm. bwd_single_step_nonnormal), while letting E -> 0, L, D -> infinity costs only
about lam_reg, so for lam_reg = 1 and eta >~ 8 the infimum is that degenerate sequence and
no minimizer exists.  (The theorem holds either way: its proof only uses that the loss is
at most C eta.)  With lam_reg = 100 every minimizer here has |E| >= 0.99.

Global minimization (6 parameters): a grid over the encoder direction phi and the latent
pole L (E = [cos phi, sin phi], R(E) = 0); at each grid point L-BFGS over the decoder and
input map (D, H); the best grid points are then polished by L-BFGS over all of
(E, D, L, H).  The gauge (E, D, H) -> -(E, D, H) is fixed by H >= 0.

Gain sets (eq. gain_sets), exact intervals:
    K_L = {K : |L + H K| < 1},   K_A = {K : rho(A_eta + B K E) < 1}  (Jury criterion).
The latent H-infinity design of a scalar latent model places the latent pole at the
origin, K = -L/H; its true closed loop rho(A_eta + B K E) is reported.

Figures:  insufficiency_eta   (T = 1, eta sweep)       -> fig:insufficiency_K_vs_b
          insufficiency_T     (eta fixed, T sweep)     -> fig:insufficiency_K_vs_T

    python -m experiments.nonnormal [--profile paper|fast] [--stage run|plot|all]
"""

from __future__ import annotations

import os

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from latentctl import io, style as S
from latentctl.systems import nonnormal

CONFIGS = {
    "paper": dict(lam_u=1.5, lam_s=0.5, b=(1.0, 1.0), sx=1.0, su=1.0, lam_fwd=1.0, lam_reg=100.0,
                  fwd_horizon=1, eta_fix=10.0,                 # 1: J_fwd^(1) in L_bwd; "T": J_fwd^(T)
                  etas=np.linspace(0.0, 10.0, 201), T_max=16, n_phi=180, n_L=321, topk=16,
                  inner_iters=40, polish_iters=600),
}
CONFIGS["fast"] = dict(CONFIGS["paper"], etas=np.linspace(0.0, 10.0, 41), T_max=12, n_phi=90, n_L=161, topk=8,
                       inner_iters=30, polish_iters=300)                  # coarser grids and eta sweep


# -----------------------------------------------------------------------------------------
# Loss and its global minimization
# -----------------------------------------------------------------------------------------
def make_loss(T, cfg):
    """L_bwd^(T)(th; eta) for th = (e (2), d (2), L, H) and A-powers passed as arguments."""
    sx, su, lf, lr = cfg["sx"], cfg["su"], cfg["lam_fwd"], cfg["lam_reg"]
    h = T if cfg["fwd_horizon"] == "T" else 1
    alpha = np.full(T, 1.0 / T)
    beta = np.array([alpha[:T - k].sum() for k in range(T)])
    wb_s = np.full(T + 1, sx ** 2 / (T + 1))
    wb_i = su ** 2 * beta / (T + 1)
    wf_s = np.zeros(T + 1)
    wf_s[1:h + 1] = sx ** 2 / h
    wf_i = np.zeros(T)
    wf_i[:h] = su ** 2 * (beta[:h] if h == T else alpha[:1]) / h
    w = [jnp.asarray(x) for x in (wb_s, wb_i, wf_s, wf_i)]

    def loss(th, Ap, AkB):
        e, d, L, H = th[0:2], th[2:4], th[4], th[5]
        Lp = jnp.concatenate([jnp.ones(1), jnp.cumprod(jnp.full(T, L))])
        Rs = Ap - Lp[:, None, None] * jnp.outer(d, e)[None]
        Ri = AkB - (Lp[:T] * H)[:, None] * d[None]
        J_bwd = jnp.sum(w[0] * jnp.sum(Rs ** 2, axis=(1, 2))) + jnp.sum(w[1] * jnp.sum(Ri ** 2, axis=1))
        F = jnp.einsum("tji,j->ti", Ap, e) - Lp[:, None] * e[None]
        Fi = AkB @ e - Lp[:T] * H
        J_fwd = jnp.sum(w[2] * jnp.sum(F ** 2, axis=1)) + jnp.sum(w[3] * Fi ** 2)
        return J_bwd + lf * J_fwd + lr * (1.0 - e @ e) ** 2

    return loss


def _lbfgs(f, x, iters):
    lb = optax.lbfgs()
    st = lb.init(x)
    vg = optax.value_and_grad_from_state(f)

    def step(c, _):
        x, st = c
        v, g = vg(x, state=st)
        up, st2 = lb.update(g, st, x, value=v, grad=g, value_fn=f)
        x2 = optax.apply_updates(x, up)
        ok = jnp.all(jnp.isfinite(x2))
        return (jnp.where(ok, x2, x), jax.tree_util.tree_map(lambda a, b: jnp.where(ok, a, b), st2, st)), None

    (x, _), _ = jax.lax.scan(step, (x, st), None, length=iters)
    return x, f(x)


def make_fitter(T, cfg):
    """fit(eta) -> dict(e, d, L, H, f): global minimizer of L_bwd^(T) for A_eta (one compile per T)."""
    loss = make_loss(T, cfg)
    B = np.array(cfg["b"])
    phis = np.linspace(0.0, np.pi, cfg["n_phi"], endpoint=False)

    def inner(th, Ap, AkB):
        e, L = th[0:2], th[4]
        f = lambda v: loss(jnp.concatenate([e, v[:2], jnp.array([L]), v[2:]]), Ap, AkB)
        v, fv = _lbfgs(f, jnp.concatenate([th[2:4], th[5:6]]), cfg["inner_iters"])
        return jnp.concatenate([e, v[:2], jnp.array([L]), v[2:]]), fv

    inner_b = jax.jit(jax.vmap(inner, in_axes=(0, None, None)))
    polish_b = jax.jit(jax.vmap(lambda th, Ap, AkB: _lbfgs(lambda t: loss(t, Ap, AkB), th, cfg["polish_iters"]),
                                in_axes=(0, None, None)))

    def fit(eta):
        A, _ = nonnormal(eta, cfg["lam_u"], cfg["lam_s"], *cfg["b"])
        Ap = np.stack([np.linalg.matrix_power(A, t) for t in range(T + 1)])
        AkB = Ap[:T] @ B
        Lmax = max(4.0, cfg["lam_u"] + 1.6 * np.sqrt(eta)) if T <= 2 else 4.0
        P, LL = np.meshgrid(phis, np.linspace(-Lmax, Lmax, cfg["n_L"]), indexing="ij")
        e0 = np.stack([np.cos(P.ravel()), np.sin(P.ravel())], 1)
        th0 = np.concatenate([e0, np.tile([1.0, 0.0], (len(e0), 1)), LL.ravel()[:, None], (e0 @ B)[:, None]], 1)
        args = (jnp.asarray(Ap), jnp.asarray(AkB))
        thI, fI = inner_b(jnp.asarray(th0), *args)
        fI = np.where(np.isfinite(np.asarray(fI)), np.asarray(fI), np.inf)
        idx = np.argsort(fI)[:cfg["topk"]]
        thP, fP = polish_b(thI[idx], *args)
        fP = np.where(np.isfinite(np.asarray(fP)), np.asarray(fP), np.inf)
        j = int(np.argmin(fP))
        th = np.asarray(thP[j])
        e, d, L, H = th[0:2], th[2:4], float(th[4]), float(th[5])
        if H < 0:
            e, d, H = -e, -d, -H
        return dict(e=e, d=d, L=L, H=H, f=float(fP[j]), f_grid=float(fI[idx[0]]))

    return fit


# -----------------------------------------------------------------------------------------
# Gain sets
# -----------------------------------------------------------------------------------------
def interval_KL(L, H):
    if abs(H) < 1e-14:
        return (-np.inf, np.inf) if abs(L) < 1 else None
    a, c = (-1.0 - L) / H, (1.0 - L) / H
    return (min(a, c), max(a, c))


def interval_KA(A, B, e):
    """{K : rho(A + B K e^T) < 1} for 2x2 A (Jury: |det| < 1, |tr| < 1 + det, affine in K)."""
    adj = np.array([[A[1, 1], -A[0, 1]], [-A[1, 0], A[0, 0]]])
    tr0, tr1 = np.trace(A), float(e @ B)
    det0, det1 = np.linalg.det(A), float(e @ adj @ B)
    lo, hi = -np.inf, np.inf
    for p, q in ((1 - det0, -det1), (1 + det0, det1), (1 + det0 - tr0, det1 - tr1), (1 + det0 + tr0, det1 + tr1)):
        if abs(q) < 1e-14:
            if p <= 0:
                return None
        elif q > 0:
            lo = max(lo, -p / q)
        else:
            hi = min(hi, -p / q)
    return None if lo >= hi else (lo, hi)


def overlap(a, b):
    if a is None or b is None:
        return None
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return None if hi <= lo else (lo, hi)


def summarize(fits, etas, cfg):
    B = np.array(cfg["b"])
    rows = []
    for eta, r in zip(etas, fits):
        A, _ = nonnormal(eta, cfg["lam_u"], cfg["lam_s"], *cfg["b"])
        KL, KA = interval_KL(r["L"], r["H"]), interval_KA(A, B, r["e"])
        K = -r["L"] / r["H"]                                       # latent H-infinity design
        rho = float(np.max(np.abs(np.linalg.eigvals(A + K * np.outer(B, r["e"])))))
        rows.append(dict(r, eta=eta, KL=KL, KA=KA, OV=overlap(KL, KA), K_hinf=K, rho_hinf=rho))
    return rows


# -----------------------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------------------
def run(cfg, out):
    with io.Timer("T = 1, eta sweep"):
        fit = make_fitter(1, cfg)
        fits = [fit(float(eta)) for eta in cfg["etas"]]
        rows_eta = summarize(fits, cfg["etas"], cfg)
    with io.Timer(f"eta = {cfg['eta_fix']}, T sweep"):
        Ts = list(range(1, cfg["T_max"] + 1))
        fits = [make_fitter(T, cfg)(cfg["eta_fix"]) for T in Ts]
        rows_T = summarize(fits, [cfg["eta_fix"]] * len(Ts), cfg)
        for T, r in zip(Ts, rows_T):
            r["T"] = T
    io.dump(dict(cfg=cfg, eta=rows_eta, T=rows_T), os.path.join(out, "nonnormal.pkl"))
    for r in rows_eta[:: max(1, len(rows_eta) // 10)]:
        print(f"  eta={r['eta']:5.2f}  L={r['L']:+.3f} H={r['H']:+.3f} |E|={np.linalg.norm(r['e']):.3f}  "
              f"K_L={fmt(r['KL'])} K_A={fmt(r['KA'])}  rho_hinf={r['rho_hinf']:.3f}")
    for r in rows_T:
        print(f"  T={r['T']:2d}  L={r['L']:+.3f} H={r['H']:+.3f} |E|={np.linalg.norm(r['e']):.3f}  "
              f"K_L={fmt(r['KL'])} K_A={fmt(r['KA'])}  rho_hinf={r['rho_hinf']:.3f}")


def fmt(iv):
    return "empty" if iv is None else f"({iv[0]:+.3f}, {iv[1]:+.3f})"


# -----------------------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------------------
# gain sets: latent model only / true system only / both (the gains that transfer)
C_L, C_A, C_OV = "#b7d3f6", "#e3e1db", "#8fd6bb"
E_L, E_A = S.BLUE, S.INK2


def _neg_iv(r, key):
    iv = r[key]
    return (np.nan, np.nan) if iv is None else (-iv[1], -iv[0])          # interval of -K


def _dense_rows(rows, cfg, n=600):
    """The fitted (E, L, H) vary smoothly with eta; interpolate them onto a dense eta grid so
    that the set boundaries (and where the intersection closes) are drawn exactly."""
    eta = np.array([r["eta"] for r in rows])
    grid = np.linspace(eta[0], eta[-1], n)
    L = np.interp(grid, eta, [r["L"] for r in rows])
    H = np.interp(grid, eta, [r["H"] for r in rows])
    E = np.stack([np.interp(grid, eta, [r["e"][i] for r in rows]) for i in range(2)], 1)
    fits = [dict(e=E[i], d=None, L=L[i], H=H[i], f=np.nan) for i in range(n)]
    return summarize(fits, grid, cfg)


def _sets_continuous(ax, y, rows):
    lo_L, hi_L = np.array([_neg_iv(r, "KL") for r in rows]).T
    lo_A, hi_A = np.array([_neg_iv(r, "KA") for r in rows]).T
    lo_O, hi_O = np.array([_neg_iv(r, "OV") for r in rows]).T
    ax.fill_betweenx(y, lo_L, hi_L, color=C_L, lw=0, zorder=1)
    ax.fill_betweenx(y, lo_A, hi_A, color=C_A, lw=0, zorder=1)
    ok = np.isfinite(lo_O)
    ax.fill_betweenx(y, np.where(ok, lo_O, 1.0), np.where(ok, hi_O, 1.0), where=ok, color=C_OV, lw=0,
                     zorder=2, interpolate=False)
    for v in (lo_L, hi_L):
        ax.plot(v, y, color=E_L, lw=0.8, zorder=3)
    for v in (lo_A, hi_A):
        ax.plot(v, y, color=E_A, lw=0.8, zorder=3)


def _sets_rows(ax, y, rows, h=0.78):
    """One row per horizon; a 2-pt surface gap separates the rows."""
    for yi, r in zip(y, rows):
        for key, col, z in (("KL", C_L, 1), ("KA", C_A, 1), ("OV", C_OV, 2)):
            a, b = _neg_iv(r, key)
            if np.isfinite(a):
                ax.fill_between([a, b], yi - h / 2, yi + h / 2, color=col, lw=0, zorder=z)
        for key, col in (("KL", E_L), ("KA", E_A)):
            a, b = _neg_iv(r, key)
            if np.isfinite(a):
                ax.plot([a, a], [yi - h / 2, yi + h / 2], color=col, lw=0.8, zorder=3, solid_capstyle="butt")
                ax.plot([b, b], [yi - h / 2, yi + h / 2], color=col, lw=0.8, zorder=3, solid_capstyle="butt")


def _gain_axis(ax, lo=0.08, hi=11.0):
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    S.log_ticks(ax, "x", [0.1, 0.2, 0.5, 1, 2, 5, 10])
    ax.set_xlabel(r"gain $-K$")
    ax.grid(axis="y", visible=False)


def _legend(ax, hinf):
    hs = [Patch(fc=C_L, ec=E_L, lw=0.8, label=r"$\mathcal{K}_L$ only"),
          Patch(fc=C_A, ec=E_A, lw=0.8, label=r"$\mathcal{K}_{A_\eta}$ only"),
          Patch(fc=C_OV, ec="none", label="both"), hinf]
    ax.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=len(hs), handlelength=1.5,
              columnspacing=1.3, borderaxespad=0.0)


HINF = r"latent $\mathcal{H}_\infty$ gain $K$"            # the left panel's dashed line


def _rho_panel(ax, rho, y, xlabel, marker=None, label_y=None, xmax=3.3):
    """Closed loop of the latent H-infinity gain, drawn in the style of that gain's line on the left."""
    ax.axvspan(0, 1, color="#f4f3f0", lw=0, zorder=0)
    ax.axvline(1.0, color=S.AXIS, lw=0.7, zorder=1)
    ax.plot(rho, y, color=S.INK, lw=1.2, ls=(0, (4, 2)), marker=marker, ms=2.8, zorder=3)
    S.title(ax, None, HINF)
    ax.set_xlim(0, xmax)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel(xlabel)
    ax.tick_params(labelleft=False)
    ax.grid(axis="y", visible=False)
    if label_y is not None:
        ax.text(0.5, label_y, "stable", ha="center", va="center", fontsize=7.5, color=S.INK2, zorder=2,
                bbox=dict(facecolor="#f4f3f0", edgecolor="none", pad=1.0))     # above the dotted lam_s line


def _axes(fig):
    ax = fig.add_axes([0.075, 0.185, 0.615, 0.66])
    axr = fig.add_axes([0.755, 0.185, 0.225, 0.66], sharey=ax)
    return ax, axr


def fig_eta(res, path):
    rows, cfg = res["eta"], res["cfg"]
    dense = _dense_rows(rows, cfg)
    y = np.array([r["eta"] for r in dense])
    fig = plt.figure(figsize=(S.TEXT_W, 2.2))
    ax, axr = _axes(fig)
    _sets_continuous(ax, y, dense)
    ax.plot([-r["K_hinf"] for r in dense], y, color=S.INK, lw=1.1, ls=(0, (4, 2)), zorder=4)
    sep = [r["eta"] for r in dense if r["OV"] is None and r["eta"] > 0]
    if sep:
        for a in (ax, axr):
            a.axhline(sep[0], color=S.INK2, lw=0.6, ls=(0, (1, 1.5)), zorder=4)
    _gain_axis(ax)
    ax.set_ylim(y[0], y[-1])
    ax.set_ylabel(r"non-normality $\eta$")
    _legend(ax, plt.Line2D([], [], color=S.INK, lw=1.1, ls=(0, (4, 2)), label=HINF))
    _rho_panel(axr, [r["rho_hinf"] for r in dense], y, r"$\rho(A_\eta + B K \bar E_1)$", label_y=0.9 * y[-1])
    S.save(fig, path)


def fig_T(res, path):
    rows = res["T"]
    T = np.array([r["T"] for r in rows])
    fig = plt.figure(figsize=(S.TEXT_W, 2.2))
    ax, axr = _axes(fig)
    _sets_rows(ax, T, rows)
    ax.plot([-r["K_hinf"] for r in rows], T, color=S.INK, lw=1.0, ls=(0, (4, 2)), marker="o", ms=2.8, zorder=4)
    _gain_axis(ax)
    ax.set_ylim(T[0] - 0.6, T[-1] + 0.6)
    ax.set_yticks([t for t in T if t in (1, 2, 3, 4) or t % 4 == 0])
    ax.set_ylabel(r"horizon $T$")
    _legend(ax, plt.Line2D([], [], color=S.INK, lw=1.0, ls=(0, (4, 2)), marker="o", ms=2.8, label=HINF))
    rho = np.array([r["rho_hinf"] for r in rows])
    _rho_panel(axr, rho, T, r"$\rho(A_\eta + B K \bar E_T)$", marker="o", label_y=1.4)
    axr.axvline(res["cfg"]["lam_s"], color=S.AXIS, lw=0.7, ls=(0, (1, 1.5)), zorder=1)
    S.save(fig, path)


def plot(profile):
    res = io.load(os.path.join(io.results_dir(profile, "nonnormal"), "nonnormal.pkl"))
    fig_eta(res, io.figure_path(profile, "insufficiency_eta"))
    fig_T(res, io.figure_path(profile, "insufficiency_T"))


def main():
    args, cfg = io.cli(__doc__, CONFIGS)
    if args.stage in ("run", "all"):
        run(cfg, io.results_dir(args.profile, "nonnormal"))
    if args.stage in ("plot", "all"):
        plot(args.profile)


if __name__ == "__main__":
    main()
