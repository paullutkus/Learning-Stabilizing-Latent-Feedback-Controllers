"""Linear latent models on the linearized flexible arm (Sec. 6.2): regularizers, horizons, finite samples.

Plant: the arm of the motivating example (n = 40, p = 1, sigma_+(A) = {1.129}), m = 4.
Data model: x_0 ~ N(0, I), u_t ~ N(0, su^2 / T), su = 0.5.

Population (pop:<loss>:<reg>:<lam>):  minimize the population L_bwd^(T) (lam_fwd = 1, with the
    one-step forward term J_fwd^(1), ``fwd_horizon = 1``) or L_fwd^(T) over linear (E, D, L, H) along
    the horizon path T in {1, 2, 5, 10, 20, 40} (optim.horizon_path: at each T, the better of the
    previous T's solution polished by L-BFGS and a fresh random init trained by Adam + L-BFGS), from
    ``seeds`` random initializations.
Finite sample (fin:<loss>:<reg>:<lam>:d<s>):  the empirical losses of N trajectories (N in
    {5, 20, 100, 500, 1000}, nested) of data set s (seed 1000 + s, drawn with N_max = 2000
    trajectories), ``fin_inits`` initializations per N; the same path protocol over ``Ts_fin``.

The reported model of a problem is its lowest-objective run (sweeps.mark_selected).  Every model is
scored by latentctl.control.Evaluator: Delta = Delta_F(G_+^R(A^T); row E), latent LQR and H-infinity
gains lifted to u = K E x, stability rho(A + B K E) < 1, LQR cost relative to full-state LQR, settling
time, and success on the nonlinear arm from control.standard_ics (the same 8 ICs scaled to each tip
displacement 2 / 10 / 20 / 30 / 40 / 60 / 80 cm along the buckling mode; |u| <= 40 N m): every population
run at 10 cm, the reported population model at the largest T and every finite-sample run at T_fin at
every tip.

lam_reg: L_bwd at lam_reg = 1; L_fwd at the lam_reg in LAMS most favourable to it (``lam_shown``: the
largest mean stabilizing rate over the path; ties -> smaller Delta of the lowest-objective model at the
largest T -> smaller lam).  The finite-sample L_fwd runs use FWD_LAM, the population selection (checked
by ``plot``).

Figure: arm_linear_models -> fig:exp_arm_recovery (panels b, e count the latent H-infinity gains; panel c
is the latent LQR closed loop);  tables linear_stability -> tab:exp_gains, linear_conjugacy ->
tab:exp_conjugacy; results/<profile>/linear/SUMMARY.md.

    python -m experiments.linear_models [--profile paper|fast] [--stage run|plot|all] [--jobs N]
"""

from __future__ import annotations

import glob
import os

import numpy as np

from latentctl import io

LAMS = (1.0, 10.0, 100.0, 1000.0)
FWD_LAM = {"orth": 1.0, "vicreg_x0": 1000.0, "sigreg_x0": 1000.0, "vicreg": 10.0, "sigreg": 100.0}     # = lam_shown
REGS_ALL = ("orth", "vicreg_x0", "sigreg_x0", "vicreg", "sigreg")
TIPS = (0.02, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80)
TIPS_CM = tuple(round(100 * t) for t in TIPS)                 # tip columns of the conjugacy table
CONFIGS = {
    "paper": dict(m=4, sx=1.0, su=0.5, lam_fwd=1.0, fwd_horizon=1, Ts=(1, 2, 5, 10, 20, 40), T_fin=40,
                  N=(5, 20, 100, 500, 1000), N_max=2000, n_nl_ic=8, tips=TIPS, seeds=8, data_seeds=5, fin_inits=2,
                  pop=[("bwd", r, 1.0) for r in REGS_ALL] + [("fwd", r, lam) for r in REGS_ALL for lam in LAMS],
                  fin=[("bwd", r, 1.0) for r in ("orth", "vicreg", "sigreg")]
                      + [("fwd", r, FWD_LAM[r]) for r in ("orth", "vicreg", "sigreg")],
                  Ts_fin=(10, 20, 40),
                  opt=dict(protocol="restart", n_adam=3000, n_lbfgs=2000, n_warm=2000, lr=1e-2, lr_end=1e-5)),
}
# reduced sweeps: 3 inits, L_fwd only at FWD_LAM (no lam_reg sweep), finite samples only for L_bwd with R_orth
# (2 data sets, 1 init), fewer L-BFGS steps
CONFIGS["fast"] = dict(CONFIGS["paper"], seeds=3, data_seeds=2, fin_inits=1,
                       pop=[("bwd", r, 1.0) for r in REGS_ALL] + [("fwd", r, FWD_LAM[r]) for r in REGS_ALL],
                       fin=[("bwd", "orth", 1.0)], opt=dict(CONFIGS["paper"]["opt"], n_lbfgs=1500, n_warm=1500))


def tasks(cfg):
    """Longest first: finite-sample paths (one task per data set), then population (T = 40 dominates)."""
    out = [f"fin:{l}:{r}:{lam:g}:d{s}" for l, r, lam in cfg["fin"] for s in range(cfg["data_seeds"])]
    return out + [f"pop:{l}:{r}:{lam:g}" for l, r, lam in cfg["pop"]]


# -----------------------------------------------------------------------------------------
# Workers
# -----------------------------------------------------------------------------------------
def worker(task, cfg, out):
    from latentctl.systems import flexible_arm
    from latentctl.control import Evaluator, simulate, standard_ics
    from latentctl.sweeps import pop_path, emp_path, score_models
    arm = flexible_arm()
    ev = Evaluator(arm.A, arm.B, dt=arm.dt)
    ics = standard_ics(arm, tips=cfg["tips"], k=cfg["n_nl_ic"], shared=True)
    kw = dict(lam_fwd=cfg["lam_fwd"], fwd_horizon=cfg["fwd_horizon"])
    kind, loss, reg, lam, *variant = task.split(":")
    lam = float(lam)
    rows, params = [], {}
    if kind == "pop":
        seeds = list(range(cfg["seeds"]))
        with io.Timer(task):
            for T, th, info in pop_path(arm.A, arm.B, cfg["m"], cfg["Ts"], loss, reg, lam, seeds, cfg["opt"],
                                        sx=cfg["sx"], su=cfg["su"], **kw):
                meta = [dict(kind=kind, loss=loss, reg=reg, lam=lam, T=T, seed=s) for s in seeds]
                r, gains = score_models(ev, arm, th, info, meta, nl_ics={0.10: ics[0.10]})   # every run: 10 cm
                if T == max(cfg["Ts"]):                         # the reported (lowest-objective) model: every tip
                    i = int(np.nanargmin([x["obj"] for x in r]))
                    for k in ("lqr", "hinf"):
                        for tip, X0 in ics.items():
                            ok = gains[i][k] is not None and r[i][f"{k}_stable"]
                            r[i][f"{k}_nl_succ_{round(100 * tip)}"] = float(simulate(arm, gains[i][k], X0)[2].mean()) if ok else 0.0
                rows += r
                params[T] = th
                report(task, T, r)
    else:
        from latentctl.data import linear_dataset
        dsets = [int(v[1:]) for v in variant if v[:1] == "d" and v[1:].isdigit()] or list(range(cfg["data_seeds"]))
        problems = [(i, N, r) for i in range(len(dsets)) for N in cfg["N"] for r in range(cfg["fin_inits"])]

        def make_data(T, _cache={}):
            if T not in _cache:
                d = [linear_dataset(arm.A, arm.B, T, cfg["N_max"], seed=1000 + s, sx=cfg["sx"], su=cfg["su"]) for s in dsets]
                _cache.clear()
                _cache[T] = (np.stack([x for x, _ in d]), np.stack([u for _, u in d]))
            return _cache[T]

        with io.Timer(task):
            for T, th, info in emp_path(make_data, cfg["m"], cfg["Ts_fin"], loss, reg, lam, problems, cfg["opt"],
                                        init_seed=1000, data_ids=dsets, **kw):
                meta = [dict(kind="fin", loss=loss, reg=reg, lam=lam, T=T, seed=dsets[i], N=N, init=r)
                        for i, N, r in problems]
                r, _ = score_models(ev, arm, th, info, meta, nl_ics=ics if T == cfg["T_fin"] else None)
                rows += r
                params[T] = th
                report(task, T, r, by="N")
    io.dump(dict(task=task, cfg=cfg, rows=rows, params=params), os.path.join(out, task.replace(":", "__") + ".pkl"))


def report(task, T, rows, by=None):
    import pandas as pd
    d = pd.DataFrame(rows)
    if by:
        g = d.groupby(by)
        print(f"  {task} T={T:3d}  Delta med {g['sub_dist'].median().round(4).to_dict()}  "
              f"LQR stable {g['lqr_stable'].mean().round(2).to_dict()}", flush=True)
    else:
        print(f"  {task} T={T:3d}  Delta med {d['sub_dist'].median():.2e}  obj {np.round(d['obj'].values, 3)}  "
              f"LQR {d['lqr_stable'].mean():.2f}  Hinf {d['hinf_stable'].mean():.2f}", flush=True)


# -----------------------------------------------------------------------------------------
# Analysis and figure
# -----------------------------------------------------------------------------------------
def load_rows(profile):
    """All rows, with 'selected' = the reported (lowest-objective) run of each problem, and the run's place in
    its task's parameters: params[task][T][k][pidx]."""
    import pandas as pd
    from latentctl.sweeps import mark_selected
    files = glob.glob(os.path.join(io.results_dir(profile, "linear"), "*.pkl"))
    res = [io.load(f) for f in files]
    df = pd.concat([pd.DataFrame(r["rows"]).assign(task=r["task"]) for r in res], ignore_index=True)
    df["pidx"] = df.groupby(["task", "T"]).cumcount()            # rows follow the problem order at every T
    df["obj_sel"] = df["obj_direct"].where(df["kind"] == "fin", df["obj"])
    keys = ["kind", "loss", "reg", "lam", "T", "seed", "N"]
    sel = mark_selected(df[df.kind == "fin"], keys, "obj_sel")
    pop = mark_selected(df[df.kind != "fin"], keys[:5], "obj_sel")
    df["selected"] = False
    df.loc[sel.index, "selected"] = sel["selected"]
    df.loc[pop.index, "selected"] = pop["selected"]
    params = {r["task"]: r.get("params") for r in res}
    return df, params


def stab_score(h):
    return 0.5 * (h["lqr_stable"].mean() + h["hinf_stable"].mean())


def lam_shown(df):
    """lam_reg per (kind, loss, reg): L_bwd at 1 (fixed); L_fwd at the value most favourable to it --
    the largest mean stabilizing rate (LQR and H-inf, all runs) over the path; ties -> smaller Delta of
    the lowest-objective model at the largest T -> smaller lam."""
    out = {}
    groups = sorted(df.groupby(["kind", "loss", "reg"]), key=lambda kg: kg[0][0] == "fin")
    for key, g in groups:
        lams = sorted(set(g.lam))
        if key[0] == "fin" and ("pop",) + key[1:] in out and out[("pop",) + key[1:]] in lams:
            out[key] = out[("pop",) + key[1:]]                 # finite samples follow the population selection
            continue
        if key[1] == "bwd":
            out[key] = 1.0 if 1.0 in lams else lams[0]
            continue
        Tm = g["T"].max()
        sc = {}
        for lam, h in g.groupby("lam"):
            d = h[(h["T"] == Tm) & h.selected]["sub_dist"].median()
            sc[lam] = (round(stab_score(h), 12), -d, -lam)
        out[key] = max(sc, key=sc.get)
    return out


REGN = {"orth": r"$R_{\mathrm{orth}}$", "vicreg_x0": r"$R_{\mathrm{vr}}$, $\mathsf{N}(0,I)$",
        "sigreg_x0": r"$R_{\mathrm{sr}}$, $\mathsf{N}(0,I)$", "vicreg": r"$R_{\mathrm{vr}}$, traj.",
        "sigreg": r"$R_{\mathrm{sr}}$, traj."}
LOSSN = {"bwd": r"$\mathcal{L}_{\mathrm{bwd}}^{(T)}$", "fwd": r"$\mathcal{L}_{\mathrm{fwd}}^{(T)}$"}
LOSSN_FIN = {"bwd": r"$\widehat{\mathcal{L}}_{\mathrm{bwd}}^{(T)}$", "fwd": r"$\widehat{\mathcal{L}}_{\mathrm{fwd}}^{(T)}$"}  # empirical


def plot(profile):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from latentctl import style as S
    from latentctl.systems import flexible_arm
    from latentctl.control import Evaluator, dlqr

    df, params = load_rows(profile)
    lam = lam_shown(df)
    sel = lambda kind, loss, reg: df[(df.kind == kind) & (df.loss == loss) & (df.reg == reg) &
                                     (df.lam == lam.get((kind, loss, reg), np.nan))]
    # colour = loss; regularizer = marker shape (family) + line and fill (base measure: solid line,
    # filled marker = N(0, I); dashed, hollow = trajectories), and its own horizontal offset
    DASH = (0, (3.0, 1.6))
    REG = {"orth": ("o", True, r"$R_{\mathrm{orth}}$"), "vicreg_x0": ("^", True, r"$R_{\mathrm{vr}}$, $\mathsf{N}(0,I)$"),
           "sigreg_x0": ("s", True, r"$R_{\mathrm{sr}}$, $\mathsf{N}(0,I)$"), "vicreg": ("^", False, r"$R_{\mathrm{vr}}$, trajectories"),
           "sigreg": ("s", False, r"$R_{\mathrm{sr}}$, trajectories")}
    LOSS = {"bwd": (S.BWD, r"$\mathcal{L}_{\mathrm{bwd}}^{(T)}$, $\lambda_{\mathrm{reg}} = 1$"),
            "fwd": (S.FWD, r"$\mathcal{L}_{\mathrm{fwd}}^{(T)}$, most favourable $\lambda_{\mathrm{reg}}$")}
    regs = [r for r in REG if r in set(df.reg)]
    slot = {r: i - (len(regs) - 1) / 2 for i, r in enumerate(regs)}          # fixed per regularizer
    dodge = lambda x, reg: np.asarray(x, float) * 2.0 ** (0.08 * slot[reg])  # the dodge rule of every figure

    def reg_style(reg, color, ms=4.0):
        mk, filled, _ = REG[reg]                                                 # hollow markers read larger
        return dict(color=color, ls="-" if filled else DASH, marker=mk, ms=ms + 0.4 if filled else ms - 0.4,
                    mfc=color if filled else S.SURFACE, mec=S.SURFACE if filled else color, mew=0.6 if filled else 0.9)

    def line(ax, x, y, loss, reg):
        ax.plot(dodge(x, reg), y, lw=1.0, zorder=3, **reg_style(reg, LOSS[loss][0]))

    def bars(ax, x, lo, hi, loss, reg):
        S.errbar(ax, dodge(x, reg), lo, hi, LOSS[loss][0])

    def count_ticks(ax, n):
        ax.set_ylim(-0.05, 1.05)
        ks = sorted({0, n // 2, n})
        ax.set_yticks([k / n for k in ks])
        ax.set_yticklabels([f"{k}/{n}" for k in ks])

    fig = plt.figure(figsize=(S.TEXT_W, 3.85))
    gs = fig.add_gridspec(2, 3, left=0.085, right=0.985, top=0.93, bottom=0.115, wspace=0.42, hspace=0.62)
    ax = {k: fig.add_subplot(gs[i, j]) for k, (i, j) in zip("abcdef", [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)])}

    # (a, b) population, vs T: (a) the lowest-objective run (line), interquartile range over the inits (bars);
    # (b) k/n inits stabilized by the latent H-infinity gain
    Ts = sorted(set(df[df.kind == "pop"]["T"]))
    n_pop = int(df[df.kind == "pop"].groupby(["loss", "reg", "lam", "T"]).size().max())
    for loss in ("bwd", "fwd"):
        for reg in regs:
            g = sel("pop", loss, reg)
            if g.empty:
                continue
            q = g.groupby("T")["sub_dist"].quantile([0.25, 0.75]).unstack().reindex(Ts)
            bars(ax["a"], Ts, q[0.25].values, q[0.75].values, loss, reg)
            line(ax["a"], Ts, g[g.selected].set_index("T")["sub_dist"].reindex(Ts).values, loss, reg)
            line(ax["b"], Ts, g.groupby("T")["hinf_stable"].mean().reindex(Ts).values, loss, reg)
    # (d, e) finite sample, vs N, T = T_fin: median of the reported models (line), spread over data sets (bars);
    # k/n data sets
    fin = df[(df.kind == "fin") & df.selected]
    T_fin = fin["T"].max() if not fin.empty else None
    Ns = sorted(set(fin["N"])) if not fin.empty else []
    n_fin = int(fin[fin["T"] == T_fin].groupby(["loss", "reg", "N"]).size().max()) if Ns else 1
    for loss in ("bwd", "fwd"):
        for reg in regs:
            g = sel("fin", loss, reg)
            g = g[(g["T"] == T_fin) & g.selected]
            if g.empty:
                continue
            lohi = np.array([S.spread(g[g["N"] == n]["sub_dist"]) for n in Ns])     # over data sets (S.SPREAD)
            bars(ax["d"], Ns, lohi[:, 0], lohi[:, 1], loss, reg)
            line(ax["d"], Ns, g.groupby("N")["sub_dist"].median().reindex(Ns).values, loss, reg)
            line(ax["e"], Ns, g.groupby("N")["hinf_stable"].mean().reindex(Ns).values, loss, reg)
    # reference: Algorithm 1 with known (A, B) stabilizes at every T and N (rho(A + BKE) = 0.87 H-inf, 0.90 LQR)
    REF_LS = (0, (4, 1.5, 1, 1.5))
    for key in ("b", "e"):
        ax[key].axhline(1.0, color=S.INK, lw=0.8, ls=REF_LS, zorder=1)

    for key, xs, xl in (("a", Ts, r"horizon $T$"), ("b", Ts, r"horizon $T$"), ("d", Ns, r"trajectories $N$"),
                        ("e", Ns, r"trajectories $N$")):
        a = ax[key]
        a.set_xscale("log")
        S.log_ticks(a, "x", xs)
        if key in ("d", "e"):                                   # 500 and 1000 are 0.3 decades apart: 1000 -> 1k
            a.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"))
        a.set_xlabel(xl)
    for key in ("a", "d"):
        ax[key].set_yscale("log")
        ax[key].set_ylim(1e-4, 2.0)
        S.pow10_ticks(ax[key])
        ax[key].set_ylabel(r"$\Delta(\mathcal{G}^{\mathbb{R}}_+(A^\top);\ \mathrm{row}\,E)$")
    count_ticks(ax["b"], n_pop)
    count_ticks(ax["e"], n_fin)
    ax["b"].set_ylabel("initializations stabilized")
    ax["e"].set_ylabel("data sets stabilized")
    S.title(ax["a"], "a", "unstable subspace")
    S.title(ax["b"], "b", r"closed-loop stability, $H_\infty$")
    S.title(ax["d"], "d", f"unstable subspace, $T = {T_fin}$")
    S.title(ax["e"], "e", f"$H_\\infty$ stability, $T = {T_fin}$")

    # (c) closed-loop response of the reported latent LQR at the largest T vs full-state LQR
    arm = flexible_arm()
    ev = Evaluator(arm.A, arm.B, dt=arm.dt)
    Tmax = max(Ts)
    g = sel("pop", "bwd", "orth")
    task = f"pop:bwd:orth:{lam.get(('pop', 'bwd', 'orth'), 1.0):g}"
    ac = ax["c"]
    t = np.arange(ev.n_sim + 1) * arm.dt
    if task in params and params[task] is not None and Tmax in params[task]:
        h = g[g["T"] == Tmax]
        i = int(np.argmax(h["selected"].values))               # the run with the lowest objective
        E, D, L, H = (np.asarray(x[int(h["seed"].values[i])]) for x in params[task][Tmax])
        K = dlqr(L, H)
        curves = [(ev.response(K @ E), S.BWD, "-", 2.4, r"latent LQR, $u = KEx$", ev.closed_loop(K @ E))] if K is not None else []
        curves.append((ev.response(ev.K_star), S.INK, (0, (3, 1.5)), 1.0, "full-state LQR", ev.closed_loop(ev.K_star)))
        for rel, col, ls, lw, lab, cl in curves:
            q = np.quantile(rel, [0.25, 0.5, 0.75], axis=1)
            ac.fill_between(t, q[0], q[2], color=col, alpha=0.10, lw=0)
            ac.plot(t, q[1], color=col, ls=ls, lw=lw, label=lab, alpha=0.9 if lw > 2 else 1.0)
            ac.plot([cl["settle"]] * 2, [0.02 / 1.6, 0.02 * 1.6], color=col, lw=1.2, solid_capstyle="butt")
        ac.axhline(0.02, color=S.AXIS, lw=0.6, zorder=0)
        ac.set_yscale("log")
        ac.set_ylim(3e-3, 4.0)
        S.pow10_ticks(ac)
        ac.set_xlim(0, 3.5)
        ac.legend(loc="upper right", handlelength=1.6)
        cr = curves[0][5]["cost_ratio"] if K is not None else np.nan
        ac.text(0.97, 0.62, f"$J/J^\\star = {cr:.2f}$", transform=ac.transAxes, ha="right", fontsize=7.5,
                color=S.INK2)
    ac.set_xlabel("time [s]")
    ac.set_ylabel(r"$\|x_t\| / \|x_0\|$")
    S.title(ac, "c", f"closed loop, $T = {Tmax}$")

    # (f) legend
    af = ax["f"]
    af.axis("off")
    hl = [S.handle(LOSS[l][0], lw=1.6, label=LOSS[l][1]) for l in ("bwd", "fwd")]
    hl.append(S.handle(S.INK, ls=REF_LS, lw=0.9, label="known $(A, B)$ + Alg. 1"))
    hr = [plt.Line2D([], [], lw=1.0, label=REG[r][2], **reg_style(r, S.INK2, ms=4.4)) for r in regs]
    l1 = af.legend(handles=hl, loc="upper left", bbox_to_anchor=(-0.12, 1.14), title="loss:", alignment="left",
                   handlelength=2.4)
    af.add_artist(l1)
    af.legend(handles=hr, loc="upper left", bbox_to_anchor=(-0.12, 0.5), title="regularizer:", alignment="left",
              handlelength=2.4)
    S.save(fig, io.figure_path(profile, "arm_linear_models"))
    summary(profile, df, lam, params)


def summary(profile, df, lam, params):
    """SUMMARY.md, and the tables linear_stability and linear_conjugacy (LaTeX tabular / row bodies)."""
    pick = lambda kind, loss, reg, l: df[(df.kind == kind) & (df.loss == loss) & (df.reg == reg) & (df.lam == l)]
    lines = ["| kind | loss | reg | lam | T | N | runs | Delta (reported) | Delta (median) | LQR stable | Hinf stable | "
             "warm | lam_min(EE^T) | LQR cost / J* | nonlinear (Hinf, 10 cm) |", "|" + "---|" * 15]
    for (kind, loss, reg), l in sorted(lam.items()):
        g = pick(kind, loss, reg, l)
        keys = ["T"] + (["N"] if kind == "fin" else [])
        for key, h in g.groupby(keys):
            key = key if isinstance(key, tuple) else (key,)
            hs = h[h.selected]
            lines.append(f"| {kind} | {loss} | {reg} | {l:g} | {key[0]} | {key[1] if len(key) > 1 else ''} | {len(h)} | "
                         f"{hs['sub_dist'].median():.2e} | {h['sub_dist'].median():.2e} | {int(h['lqr_stable'].sum())}/{len(h)} | "
                         f"{int(h['hinf_stable'].sum())}/{len(h)} | {int(h['from_warm'].sum())} | {hs['E_smin2'].median():.3g} | "
                         f"{hs['lqr_cost_ratio'].median():.3f} | {h['hinf_nl_succ'].mean():.2f} |")
    with open(os.path.join(io.results_dir(profile, "linear"), "SUMMARY.md"), "w") as fh:
        fh.write(f"profile {profile}; 'reported' = lowest-objective run (per data set and N for the finite sample)\n\n"
                 + "\n".join(lines) + "\n")

    def sci(x):
        if not np.isfinite(x):
            return "--"
        if abs(x) >= 0.01:
            return f"{x:.2g}"
        m_, e = f"{x:.1e}".split("e")
        return f"{m_}\\cdot 10^{{{int(e)}}}"

    hdr = lambda name: f"% generated by experiments.linear_models (profile {profile}): {name}\n"
    stability_table(profile, df, lam, hdr, sci)
    conjugacy_table(profile, df, lam, params, hdr, sci)
    for reg in ("orth", "vicreg", "sigreg"):                     # the finite-sample L_fwd lam must be the population one
        lf, lp = lam.get(("fin", "fwd", reg)), lam.get(("pop", "fwd", reg))
        if lf is not None and lp is not None and lf != lp:
            print(f"  WARNING: finite-sample L_fwd/{reg} ran at lam = {lf:g}, the population selects {lp:g}; "
                  "update FWD_LAM and rerun it")


def stability_table(profile, df, lam, hdr, sci):
    """tab:exp_gains, a complete tabular (the numbers of Fig. exp_arm_recovery plus LQR) at the lam_reg of the
    figure: entries '$LQR stabilizing, H-inf stabilizing, Delta$'.  Population block: inits per T (panel b) and
    Delta of the lowest-objective one (panel a); finite-sample block at the largest T: data sets per N (panel e)
    and the median Delta over the data sets (panel d)."""
    pop = df[df.kind == "pop"]
    fin = df[(df.kind == "fin") & df.selected]
    Ts = sorted(set(pop["T"]))
    T_fin = fin["T"].max() if not fin.empty else None
    fin = fin[fin["T"] == T_fin]
    Ns = sorted(set(fin["N"]))
    w = max(len(Ts), len(Ns))
    pad = lambda cells: cells + [""] * (w - len(cells))

    def block(kind, g, key, cols):
        rows = []
        for loss in ("bwd", "fwd"):
            j = 0
            for reg in REGS_ALL:
                if (kind, loss, reg) not in lam:
                    continue
                h = g[(g.loss == loss) & (g.reg == reg) & (g.lam == lam[(kind, loss, reg)])]
                if h.empty:
                    continue
                cells = []
                for c in cols:
                    hc = h[h[key] == c]
                    d = hc[hc.selected]["sub_dist"].iloc[0] if kind == "pop" else hc["sub_dist"].median()
                    cells.append(f"${int(hc.lqr_stable.sum())}, {int(hc.hinf_stable.sum())}, {sci(d)}$")
                name = (LOSSN_FIN if kind == "fin" else LOSSN)[loss]
                rows.append(f"{name if j == 0 else ''} & {REGN[reg]} & ${lam[(kind, loss, reg)]:g}$ & "
                            + " & ".join(pad(cells)) + r" \\")
                j += 1
            if loss == "bwd" and j:
                rows.append(r"\midrule")
        while rows and rows[-1] == r"\midrule":
            rows.pop()
        return rows

    head = lambda label, n: (rf"{label}: \# stable LQR gains (of ${n}$), \# stable $H_\infty$ gains (of ${n}$), "
                             r"$\Delta$")
    n_pop = int(pop.groupby(["loss", "reg", "lam", "T"]).size().max())
    rows = [r"\begin{tabular}{lll" + "c" * w + "}", r"\toprule",
            f" & & & \\multicolumn{{{len(Ts)}}}{{c}}{{{head('population', n_pop)}}}" + " &" * (w - len(Ts)) + r" \\",
            f"\\cmidrule(l){{4-{3 + len(Ts)}}}",
            r"loss & regularizer & $\lambda_{\mathrm{reg}}$ & " + " & ".join(pad([f"$T = {Ts[0]}$"] + [f"${t}$" for t in Ts[1:]]))
            + r" \\", r"\midrule"] + block("pop", pop, "T", Ts)
    if Ns:
        n_fin = int(fin.groupby(["loss", "reg", "N"]).size().max())
        rows += [r"\midrule[\heavyrulewidth]",
                 f" & & & \\multicolumn{{{len(Ns)}}}{{c}}{{{head('finite sample', n_fin)}}}" + " &" * (w - len(Ns))
                 + r" \\", f"\\cmidrule(l){{4-{3 + len(Ns)}}}",
                 " & & & " + " & ".join(pad([f"$N = {int(Ns[0])}$"] + [f"${int(n)}$" for n in Ns[1:]])) + r" \\", r"\midrule"]
        rows += block("fin", fin, "N", Ns)
    rows += [r"\bottomrule", r"\end{tabular}"]
    with open(io.table_path(profile, "linear_stability"), "w") as fh:
        fh.write(hdr("LQR / H-inf stabilizing models and Delta: population inits per T (Delta of the lowest-objective "
                     f"init), finite-sample data sets per N at T = {T_fin} (median Delta)") + "\n".join(rows) + "\n")


CONJ_NS = (100, 500, 1000)                                  # finite-sample rows of the conjugacy table


def conjugacy_table(profile, df, lam, params, hdr, sci):
    """tab:exp_conjugacy body: the reported models at the largest T, at the lam_reg of Fig. exp_arm_recovery --
    population: the lowest-objective init; finite sample (N in CONJ_NS): the reported models of the data sets.
    Columns: lam_min(E E^T), r_A = ||E^A - L^E^||_F (orthonormal encoder rows), r_u and r_h (linalg.conjugacy;
    models with Delta < 0.5 only) -- medians over the data sets -- and the H-inf success on the non-linear arm
    per tip of TIPS_CM (mean over the data sets)."""
    import pandas as pd
    from latentctl.systems import flexible_arm
    from latentctl.linalg import conjugacy
    arm = flexible_arm()
    lam_u = float(np.real(max(np.linalg.eigvals(arm.A), key=abs)))

    def med(h, key, captured=False):
        v = h[key][h.sub_dist < 0.5] if captured else h[key]
        return float(np.median(v)) if len(v) else np.nan

    def with_res(h):
        res = []
        for _, r in h.iterrows():
            E, _, L, H = (np.asarray(x[int(r.pidx)]) for x in params[r.task][int(r["T"])])
            res.append(conjugacy(E, L, H, arm.A, arm.B, lam_u))
        return pd.concat([h, pd.DataFrame(res, index=h.index)], axis=1)

    Tm = df[df.kind == "pop"]["T"].max()
    fin = df[(df.kind == "fin") & df.selected]
    T_fin = fin["T"].max() if not fin.empty else None
    rows = []
    for loss in ("bwd", "fwd"):
        i = 0
        for reg in REGS_ALL:
            g = df[(df.kind == "pop") & (df.loss == loss) & (df.reg == reg) & (df.lam == lam.get(("pop", loss, reg)))
                   & (df["T"] == Tm)]
            if g.empty:
                continue
            parts = [("pop.", with_res(g[g.selected]))]
            if ("fin", loss, reg) in lam:
                f = fin[(fin.loss == loss) & (fin.reg == reg) & (fin.lam == lam[("fin", loss, reg)]) & (fin["T"] == T_fin)]
                for N in CONJ_NS:
                    h = f[f.N == N]
                    if not h.empty:
                        parts.append((f"${N}$", with_res(h)))
            for j, (lab, h) in enumerate(parts):
                first = (f"{LOSSN[loss] if i == 0 else ''} & {REGN[reg]}") if j == 0 else " & "
                cells = [med(h, "E_smin2"), med(h, "res_A"), med(h, "r_u", True), med(h, "r_h", True)]
                tips = [h.get(f"hinf_nl_succ_{c}", pd.Series(np.nan, index=h.index)).mean() for c in TIPS_CM]
                rows.append(f"{first} & {lab} & " + " & ".join(f"${sci(v)}$" if np.isfinite(v) else "--" for v in cells)
                            + " & " + " & ".join(f"{v:.2f}" if np.isfinite(v) else "--" for v in tips) + r" \\")
            i += 1
        if loss == "bwd":
            rows.append(r"\midrule")
    with open(io.table_path(profile, "linear_conjugacy"), "w") as fh:
        fh.write(hdr(f"T = {Tm} (population: lowest-objective init) / T = {T_fin} (finite: median, tips mean over data "
                     "sets): lam_min(EE^T), orthonormal-gauge ||EA-LE||_F, r_u, r_h (Delta < 0.5), H-inf success on the "
                     f"nonlinear arm at {'/'.join(map(str, TIPS_CM))} cm") + "\n".join(rows) + "\n")


def main():
    def extra(ap):
        ap.add_argument("--worker", default=None, help="internal: run one task")
        ap.add_argument("--jobs", type=int, default=8)
    args, cfg = io.cli(__doc__, CONFIGS, extra)
    out = io.results_dir(args.profile, "linear")
    if args.worker:
        return worker(args.worker, cfg, out)
    if args.stage in ("run", "all"):
        io.run_workers("experiments.linear_models", args.profile, tasks(cfg), args.jobs,
                       os.path.join(out, "logs"), args.cpus)
    if args.stage in ("plot", "all"):
        plot(args.profile)


if __name__ == "__main__":
    main()
