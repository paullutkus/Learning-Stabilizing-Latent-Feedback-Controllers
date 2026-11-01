"""Several unstable modes (Sec. 6.4, Fig. exp_multi_unstable, Table tab:exp_multi_unstable): initialization of L.

Plant: the arm with three buckling modes growing at 1.15 / 1.12 / 1.09 (stable partners 0.860 /
0.880 / 0.902 in the same mode planes), mass-normalized coordinates, n = 40, p = 1; m = m_+ = 3.
Data: N = 2000 trajectories, x_0 ~ N(0, I), u_t ~ N(0, 0.5^2 / T); data set s uses seed 1000 + s.

Loss: empirical L_bwd with linear (E, D, L, H) and R(E) = ||I - E E^T||_F^2,
    J_bwd + lam_fwd J_fwd^(1) + lam_reg R(E),  lam_fwd = lam_reg = 1,
evaluated exactly from the data's second moments; every reported objective is recomputed from
per-trajectory residuals (runs where the two disagree are numerically invalid and count as failures).
Optimizer: Adam 2000 steps (lr 3e-3, warmup + cosine to 1e-5, clip 10), then L-BFGS 2000 steps.

Initializations (E, D ~ N(0, 1/n), H ~ 0.1 N(0, 1)), ``R`` per data set and strategy:
    orth             L random orthogonal (QR of a Gaussian)                 the standard init
    eye*1.2          L = 1.2 I
    eye*1.2+hold500  L = 1.2 I, held for the first 500 Adam steps (L's update masked)
    auto+hold500     L = rho-hat I, held for 500 steps; rho-hat = (E|x_T|^2 / E|x_{T-1}|^2)^1/2
Mode i is captured when |P_row(E) w_i| > 0.9 (w_i its unit left eigenvector); all three are captured
exactly when Delta_F(G_+^R(A^T); row E) < 0.5 (identical classification in every run).  f*(T, s) is the
lowest objective of any valid run on data set s ("the lowest objective found").  Each T is trained
separately (no horizon path): the experiment studies initialization.

Outputs: the table multi_unstable_mu -> tab:exp_multi_unstable (runs capturing all three modes; rows only, the
paper has the header), results/<profile>/multi_unstable/SUMMARY.md (also: runs within 1% of f*, the gap of
Delta_F between capturing and other runs), and the figure arm_multi_unstable -> fig:exp_multi_unstable: training
traces at T = T_trace on data set 1 of the standard init and of the held rho-hat I init -- (a) objective / f*,
(b) latent eigenvalues, (c) captured unstable subspace ||P_row(E) W_+||_F^2 (out of 3).

    python -m experiments.multi_unstable [--profile paper|fast] [--stage run|plot|all] [--jobs N]
"""

from __future__ import annotations

import os
import re

import numpy as np

from latentctl import io

STRATEGIES = ("orth", "eye*1.2", "eye*1.2+hold500", "auto+hold500")
CONFIGS = {
    "paper": dict(m=3, N=2000, su=0.5, lam_fwd=1.0, lam_reg=1.0, fwd_horizon=1, n_adam=2000, n_lbfgs=2000,
                  lr=3e-3, lr_end=1e-5, clip=10.0, record_every=20, T_trace=40, unstable=(1.15, 1.12, 1.09),
                  Ts=(10, 20, 30, 40), seeds=(0, 1, 2, 3, 4), R=16, strategies=STRATEGIES),
}
CONFIGS["fast"] = dict(CONFIGS["paper"], seeds=(0,), R=8)                 # one data set, 8 runs per strategy


def parse(name):
    m = re.fullmatch(r"(orth|eye|auto)(?:\*([0-9.]+))?(?:\+hold(\d+))?", name)
    base, rho, hold = m.groups()
    return base, (float(rho) if rho else 1.0), (int(hold) if hold else 0)


def tasks(cfg):
    return [f"{T}:{s}:{st}" for T in sorted(cfg["Ts"], reverse=True) for s in cfg["seeds"] for st in cfg["strategies"]]


# -----------------------------------------------------------------------------------------
# Problem
# -----------------------------------------------------------------------------------------
class Problem:
    def __init__(self, T, seed, cfg):
        from latentctl.systems import flexible_arm_3modes
        from latentctl.data import linear_dataset
        from latentctl.losses import EmpLoss, growth_rate
        self.arm = flexible_arm_3modes()
        A, B = self.arm.A, self.arm.B
        self.T, self.m, self.cfg = T, cfg["m"], cfg
        self.X, self.U = linear_dataset(A, B, T, cfg["N"], seed=1000 + seed, su=cfg["su"])
        self.el = EmpLoss(self.X[None], self.U[None], cfg["m"], "bwd", "orth", lam_fwd=cfg["lam_fwd"],
                          fwd_horizon=cfg["fwd_horizon"])
        self.aux = self.el.aux(0, cfg["N"])
        self.rho_hat = growth_rate(self.X)
        ev, Wl = np.linalg.eig(A.T)
        idx = [int(np.argmin(np.abs(ev - lam))) for lam in cfg["unstable"]]
        W = np.real(Wl[:, idx])
        self.W = W / np.linalg.norm(W, axis=0)                    # unit left eigenvectors, (n, 3)
        self.f = lambda th, aux, data: self.el.objective(th, cfg["lam_reg"], aux, data)
        self.f_direct = lambda th, aux, data: self.el.objective_direct(th, cfg["lam_reg"], aux, data)

    def cos(self, E):
        V = np.linalg.qr(np.asarray(E).T)[0]
        return np.linalg.norm(V.T @ self.W, axis=0)

    def init(self, name, R, key):
        import jax
        from latentctl.optim import init_batch
        base, rho, hold = parse(name)
        keys = jax.random.split(key, R)
        if base == "orth":
            return init_batch(keys, self.arm.n, self.m, self.arm.p, "orth"), hold
        return init_batch(keys, self.arm.n, self.m, self.arm.p, "eye", self.rho_hat if base == "auto" else rho), hold


def worker(task, cfg, out):
    import jax
    import jax.numpy as jnp
    from latentctl.optim import make_solver
    T, seed, name = task.split(":")
    T, seed = int(T), int(seed)
    prob = Problem(T, seed, cfg)
    th0, hold = prob.init(name, cfg["R"], jax.random.PRNGKey(10_000 * seed + 7))
    solve = jax.jit(jax.vmap(make_solver(prob.f, n_adam=cfg["n_adam"], n_lbfgs=cfg["n_lbfgs"], lr=cfg["lr"],
                                         lr_end=cfg["lr_end"], clip=cfg["clip"], hold_L=hold,
                                         record_every=cfg["record_every"]), in_axes=(0, None, None)))
    with io.Timer(task):
        th, info, frames = solve(th0, prob.aux, prob.el.data)
    th = tuple(np.asarray(x) for x in th)
    obj_direct = np.asarray(jax.jit(jax.vmap(prob.f_direct, in_axes=(0, None, None)))(
        tuple(jnp.asarray(x) for x in th), prob.aux, prob.el.data))
    cos = np.stack([prob.cos(E) for E in th[0]])
    lam = np.stack([np.sort_complex(np.linalg.eigvals(L))[::-1] for L in th[2]])
    fr = {k: np.swapaxes(np.asarray(v), 0, 1) for k, v in zip(("E", "L", "obj"), frames)}     # (frames, runs, ...)
    fr["cos"] = np.array([[prob.cos(E) for E in Ef] for Ef in fr["E"]])            # (frames, R, 3)
    fr["lam"] = np.array([[np.linalg.eigvals(L) for L in Lf] for Lf in fr["L"]])    # (frames, R, 3)
    del fr["E"]
    res = dict(task=task, T=T, seed=seed, strategy=name, hold=hold, rho_hat=prob.rho_hat, obj=np.asarray(info["obj"]),
               obj_direct=obj_direct, cos=cos, lam=lam, params=th, frames=fr, record_every=cfg["record_every"],
               n_adam=cfg["n_adam"])
    io.dump(res, os.path.join(out, task.replace(":", "__").replace("*", "x").replace("+", "_") + ".pkl"))
    ncap = (cos > 0.9).sum(1)
    print(f"  {task}: rho_hat={prob.rho_hat:.4f}  #captured {np.bincount(ncap, minlength=4).tolist()}  "
          f"objective {np.round(obj_direct, 5)}", flush=True)


# -----------------------------------------------------------------------------------------
# Analysis, table and figure
# -----------------------------------------------------------------------------------------
LABEL = {"orth": r"random orthogonal $L$", "eye*1.2": r"$L = 1.2\,I_m$",
         "eye*1.2+hold500": r"$L = 1.2\,I_m$, held for $500$ steps",
         "auto+hold500": r"$L = \hat{\rho}\,I_m$, held for $500$ steps"}


def records(profile):
    import glob
    from latentctl.systems import flexible_arm_3modes
    from latentctl.linalg import subspace_distance, unstable_left_basis
    Q = unstable_left_basis(flexible_arm_3modes().A)
    runs = [io.load(f) for f in glob.glob(os.path.join(io.results_dir(profile, "multi_unstable"), "*.pkl"))]
    fstar = {}
    for r in runs:
        valid = np.abs(r["obj"] - r["obj_direct"]) <= 1e-6 * np.maximum(1.0, np.abs(r["obj_direct"]))
        r["valid"] = valid
        r["ncap"] = np.where(valid, (r["cos"] > 0.9).sum(1), 0)
        r["delta_F"] = np.array([subspace_distance(Q, E, "fro") for E in r["params"][0]])
        for v, ok in zip(r["obj_direct"], valid):
            if ok:
                fstar[(r["T"], r["seed"])] = min(fstar.get((r["T"], r["seed"]), np.inf), v)
    return runs, fstar


def table(profile):
    runs, fstar = records(profile)
    Ts = sorted({r["T"] for r in runs})
    cell = {}
    for r in runs:
        k = (r["strategy"], r["T"])
        a, n, b = cell.get(k, (0, 0, 0))
        best = np.sum(r["valid"] & (r["obj_direct"] <= 1.01 * fstar[(r["T"], r["seed"])]))
        cell[k] = (a + int(np.sum(r["ncap"] == 3)), n + len(r["ncap"]), b + int(best))
    rows, md = [], ["| initialization | " + " | ".join(f"T = {T}" for T in Ts) + " |", "|---|" + "---|" * len(Ts)]
    for s in STRATEGIES:
        if not any(k[0] == s for k in cell):
            continue
        cs = [cell.get((s, T)) for T in Ts]
        rows.append(LABEL[s] + " & " + " & ".join("--" if c is None else f"{c[0]}/{c[1]}" for c in cs) + r" \\")
        md.append(f"| {s} | " + " | ".join("—" if c is None else f"{c[0]}/{c[1]} (best {c[2]})" for c in cs) + " |")
    with open(io.table_path(profile, "multi_unstable_mu"), "w") as fh:
        fh.write(f"% generated by experiments.multi_unstable (profile {profile}): runs capturing all three modes\n"
                 + "\n".join(rows) + "\n")
    # gap of Delta_F between capturing and other runs; objective ratios of the 1-, 2-mode end points
    dF = np.concatenate([r["delta_F"][r["valid"]] for r in runs])
    nc = np.concatenate([r["ncap"][r["valid"]] for r in runs])
    md += ["", f"Delta_F: runs capturing all three modes <= {dF[nc == 3].max():.3f}, all other runs >= {dF[nc < 3].min():.3f}; "
           f"Delta_F < 0.5 <=> three modes captured: {bool(np.all((dF < 0.5) == (nc == 3)))}"]
    for T in Ts:
        ratio = {k: [] for k in range(4)}
        for r in runs:
            if r["T"] == T:
                for c, o, v in zip(r["ncap"], r["obj_direct"], r["valid"]):
                    if v:
                        ratio[int(c)].append(o / fstar[(T, r["seed"])])
        md.append(f"T = {T}: objective / f* by number of captured modes (median, n): " + ", ".join(
            f"{k}: {np.median(v):.3g} ({len(v)})" for k, v in ratio.items() if v))
    md += ["", "f* per (T, data seed): " + ", ".join(f"T={T} s={s}: {v:.6g}" for (T, s), v in sorted(fstar.items()))]
    with open(os.path.join(io.results_dir(profile, "multi_unstable"), "SUMMARY.md"), "w") as fh:
        fh.write("all three unstable modes captured (k/n), in parentheses: final objective <= 1.01 f*\n\n")
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))
    return runs, fstar


def plot(profile):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from latentctl import style as S
    runs, fstar = table(profile)
    cfg = CONFIGS[profile]
    T = cfg["T_trace"]
    pick = {s: [r for r in runs if r["T"] == T and r["strategy"] == s and r["seed"] == min(cfg["seeds"])]
            for s in ("orth", "auto+hold500")}
    if not all(pick.values()):
        print("  (no traces for the figure)")
        return
    sty = {"orth": (S.MUTED, 0.75, "random orthogonal $L_0$"), "auto+hold500": (S.BWD, 0.9, r"$L_0 = \hat\rho I$, held")}
    fig = plt.figure(figsize=(S.TEXT_W, 2.1))
    gs = fig.add_gridspec(1, 3, left=0.075, right=0.99, top=0.80, bottom=0.2, wspace=0.36)
    ax = [fig.add_subplot(gs[0, j]) for j in range(3)]
    for s in ("orth", "auto+hold500"):
        r = pick[s][0]
        col, lw, _ = sty[s]
        fr = r["frames"]                                                       # (frames, runs, ...)
        it = (np.arange(fr["obj"].shape[0]) + 1) * r["record_every"]           # Adam, then L-BFGS iterations
        f0 = fstar[(r["T"], r["seed"])]
        ax[0].plot(it, fr["obj"] / f0, color=col, lw=lw, alpha=0.85)
        lam = np.sort(fr["lam"].real, axis=-1)
        for i in range(lam.shape[1]):
            for k in range(lam.shape[2]):
                ax[1].plot(it, lam[:, i, k], color=col, lw=lw * 0.8, alpha=0.75)
        cap = np.sum(fr["cos"] ** 2, axis=-1)
        ax[2].plot(it, cap, color=col, lw=lw, alpha=0.85)
    hold = parse("auto+hold500")[2]
    for a in ax:
        a.axvspan(0, hold, color=S.BLUE_TINT, lw=0, zorder=0)
        a.axvline(cfg["n_adam"], color=S.INK2, lw=0.7, ls=(0, (3, 2)), zorder=0.5)       # Adam -> L-BFGS
        a.grid(axis="x", visible=False)
        a.set_xlim(0, cfg["n_adam"] + cfg["n_lbfgs"])
        a.set_xticks([0, 1000, 2000, 3000, 4000])
        a.set_xticklabels(["0", "1k", "2k", "3k", "4k"])
        a.set_xlabel("iteration")
    ax[0].set_yscale("log")
    S.pow10_ticks(ax[0])
    ax[0].set_ylabel(r"objective $/\ f^\star$")
    for lam in cfg["unstable"]:
        ax[1].axhline(lam, color=S.INK, lw=0.5, ls=(0, (1, 1.5)), zorder=1)
    ax[1].set_ylim(-1.1, 1.3)
    ax[1].set_ylabel(r"Re $\lambda(L)$")
    ax[2].set_ylim(-0.05, 3.15)
    ax[2].set_yticks([0, 1, 2, 3])
    ax[2].set_ylabel(r"$\|P_{\mathrm{row}(E)} W_+\|_F^2$")
    S.title(ax[0], "a", "objective")
    S.title(ax[1], "b", "latent eigenvalues")
    S.title(ax[2], "c", "unstable modes captured")
    hs = [S.handle(sty[s][0], lw=1.4, label=sty[s][2]) for s in ("orth", "auto+hold500")]
    hs.append(Patch(fc=S.BLUE_TINT, ec="none", label=f"$L$ held ({hold} Adam steps)"))
    fig.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, columnspacing=2.0)
    S.save(fig, io.figure_path(profile, "arm_multi_unstable"))


def main():
    def extra(ap):
        ap.add_argument("--worker", default=None, help="internal: run one T:seed:strategy")
        ap.add_argument("--jobs", type=int, default=8)
    args, cfg = io.cli(__doc__, CONFIGS, extra)
    out = io.results_dir(args.profile, "multi_unstable")
    if args.worker:
        return worker(args.worker, cfg, out)
    if args.stage in ("run", "all"):
        io.run_workers("experiments.multi_unstable", args.profile, tasks(cfg), args.jobs, os.path.join(out, "logs"),
                       args.cpus)
    if args.stage in ("plot", "all"):
        plot(args.profile)


if __name__ == "__main__":
    main()
