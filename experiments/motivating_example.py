"""Motivating example (Sec. 1.1, Fig. arm_model): backwards vs decoder-free forwards loss on the non-linear arm.

Plant: the 20-section flexible arm, n = 40, one unstable eigenvalue (1.129).  Data: data set 1 of Sec. 6.3
(experiments.nonlinear_models): N = 2000 trajectories of length T = 40 of the non-linear arm at the data scale
eps = 0.01.  Models: tanh-MLP encoder, decoder and control-affine latent dynamics z+ = a_z(z) + g_z(z) u, m = 4,
trained with the MLP protocol of experiments.nonlinear_models at T = 40 only (8 random inits, 1000 burn-in steps,
the best trained to 30k minibatch Adam steps) on

    L_bwd = J_bwd^(T) + J_fwd^(T) + SIGReg       (eq. loss_combo_bwd;   task mlp:nonlinear:s0:sr:hT)
    L_fwd =             J_fwd^(T) + SIGReg       (eq. loss_combo_sigreg; task mlp:nonlinear:s0:Lf:sr:hT)

with SIGReg the sliced Epps-Pulley statistic of the embedded trajectory states and lam_fwd = lam_reg = 1.
For each model, K is the latent LQR gain of the linearization (J_az(0), g_z(0)), and u = sat(K J_E(0) x)
(|u| <= 40 N m) drives the non-linear arm from initial state 7 of control.standard_ics(shared=True) with its
(linearized) tip coordinate at 110 cm, released from rest.

Figure: arm_example -> fig:arm_model
    (A, B) arm centerlines under each controller at 7 snapshot times, opacity = time: (A) up to the time from which
    the tip stays within 2 cm of the vertical (rounded up to 0.25 s); (B) up to its contact with the ground
    (x at the tip), with its own time key; the initial state in gray.  (C) latent spectra sigma(J_az(0)),
    (D) closed-loop spectra sigma(A + B K J_E(0)) at the equilibrium, both over the open-loop sigma(A),
    (E) the latent feedback loop.

    python -m experiments.motivating_example [--profile paper|fast] [--stage run|plot|all] [--jobs 2]
"""

from __future__ import annotations

import os

import numpy as np

from latentctl import io
from experiments import nonlinear_models

CONFIGS = {"paper": dict(nonlinear_models.CONFIGS["paper"], Ts=(40,), seeds=(0,), eps_sweep=(), tips=(0.02, 0.10, 0.30))}
CONFIGS["fast"] = dict(CONFIGS["paper"])          # no reduced version: the two models take ~10-20 min each
TASKS = {"bwd": "mlp:nonlinear:s0:sr:hT", "fwd": "mlp:nonlinear:s0:Lf:sr:hT"}
TIP = 1.1                   # [m] linearized tip coordinate of the initial state
IC = 6                      # initial state 7 of 8: the largest section angle
TIP_TOL = 0.02              # [m] lateral tip offset counted as vertical (about the drawn line width)
FINE = 10                   # (B) is located on a grid of dt / FINE: the arm falls fast


def mlp_law(arm, path):
    """(J_E(0), J_az(0), K, x -> u) of the MLP selected at the largest T of the results file ``path``:
    u = K J_E(0) x before saturation, in normalized input units (u = 0 if (J_az(0), g_z(0)) has no stabilizing
    LQR gain)."""
    import jax
    jax.config.update("jax_enable_x64", True)
    from latentctl.linalg import canonical_targets
    r = io.load(path)
    rc, rec = r["cfg"], r["recs"][max(r["recs"])]
    JE, JD, Jf, g0 = (np.asarray(a) for a in nonlinear_models.linearization("mlp", rec, rc, arm)[0])
    K = nonlinear_models.equilibrium_metrics(JE, JD, Jf, g0, arm.A, arm.B, canonical_targets(arm.A, arm.B, rc["m"]))["K"]
    if K is None:
        return JE, Jf, np.zeros((arm.p, rc["m"])), (lambda X: np.zeros((len(X), arm.p)))
    return JE, Jf, K, (lambda X: X @ (K @ JE).T)


def rollout(arm, fb, x0, t_end=12.0, r=1):
    """States of u = sat(fb(x)) on the non-linear arm from x0, the input held over each dt and the states recorded
    every dt / r; NaN from the first non-finite state on.  Returns (states, whether the integration blew up)."""
    from latentctl.systems import flexible_arm
    fine = arm if r == 1 else flexible_arm(dt=arm.dt / r, substeps=arm._substeps // r)
    umax = arm.u_max * arm.u_scale
    steps = int(round(t_end / arm.dt))
    X, xs = np.array(x0, float)[None], [np.array(x0, float)]
    with np.errstate(all="ignore"):
        for _ in range(steps):
            U = np.clip(fb(X), -umax, umax)
            Uf = U if r == 1 else U / arm.u_scale * fine.u_scale             # the same torque on the fine grid
            for _ in range(r):
                X = fine.step(X, Uf, "nonlinear")
                xs.append(X[0].copy())
            if not np.all(np.isfinite(X)):
                xs += [np.full(arm.n, np.nan)] * (steps * r + 1 - len(xs))
                return np.stack(xs), True
    return np.stack(xs), False


def ground_step(arm, xs):
    """First step at which a point of the arm is 1 cm below min(0, its lowest point at t = 0), or None."""
    low = np.array([arm.centerline(x[:arm.n_sec])[1].min() for x in xs])
    hit = np.nonzero(low[1:] < min(0.0, low[0]) - 0.01)[0]
    return int(hit[0]) + 1 if len(hit) else None


def upright_time(arm, xs):
    """Time [s] from which the lateral tip offset stays below TIP_TOL, rounded up to 0.25 s."""
    off = np.array([abs(arm.centerline(x[:arm.n_sec])[0][-1]) for x in xs])
    k = len(off) - int(np.argmax(off[::-1] >= TIP_TOL))
    return float(np.ceil(k * arm.dt / 0.25 - 1e-9) * 0.25)


def closed_loops(profile):
    """The closed loops of Fig. arm_model: (arm, {loss: xs, L, K, E [, dt, snapshots, stop]}, snapshot times)."""
    from latentctl.control import standard_ics
    from latentctl.linalg import subspace_distance, unstable_left_basis
    from latentctl.systems import flexible_arm
    arm = flexible_arm()
    x0 = standard_ics(arm, (TIP,), 8, shared=True)[TIP][IC]
    cx, cy = arm.centerline(x0[:arm.n_sec])
    print(f"  initial state: largest section angle {np.degrees(np.abs(arm.S @ x0[:arm.n_sec])).max():.0f} deg from "
          f"upright, tip {100 * abs(cx[-1]):.0f} cm to the side, {100 * cy[-1]:.0f} cm high", flush=True)
    d = io.results_dir(profile, "motivating_example")
    res = {}
    for loss, task in TASKS.items():
        JE, Jf, K, fb = mlp_law(arm, os.path.join(d, task.replace(":", "__") + ".pkl"))
        xs, blew = rollout(arm, fb, x0)
        res[loss] = dict(xs=xs, L=Jf, K=K, E=JE, fb=fb, blew=blew)
        print(f"  {loss}: Delta = {subspace_distance(unstable_left_basis(arm.A), JE, 'fro'):.2e}, |eig J_az(0)| "
              f"{np.round(np.sort(np.abs(np.linalg.eigvals(Jf)))[::-1], 3)}, rho(A + B K J_E(0)) = "
              f"{np.max(np.abs(np.linalg.eigvals(arm.A + arm.B @ K @ JE))):.4f}" + (" (blow-up)" if blew else ""), flush=True)
    xs = res["bwd"]["xs"]
    settled = (not res["bwd"]["blew"]) and bool(np.linalg.norm(xs[-1]) <= 0.05 * np.linalg.norm(xs[0]))
    t_end = upright_time(arm, xs) if settled else 1.75
    print(f"  bwd: " + (f"upright from t = {t_end:g} s" if settled else "does not settle within 12 s"), flush=True)
    snapshots = np.linspace(0.0, t_end, 7)
    # (B): until it reaches the ground (or its last finite state) within the window, on the fine grid
    Xf = rollout(arm, res["fwd"]["fb"], x0, t_end, FINE)[0]
    ends = [k for k in (ground_step(arm, Xf),
                        int(np.argmax(~np.all(np.isfinite(Xf), axis=1))) - 1 if not np.all(np.isfinite(Xf)) else None)
            if k is not None and k <= round(t_end / arm.dt) * FINE]
    if ends:
        kf = min(ends)
        res["fwd"].update(xs=Xf, dt=arm.dt / FINE, snapshots=np.linspace(0.0, kf * arm.dt / FINE, 7), stop=kf)
        print(f"  fwd: reaches the ground at t = {kf * arm.dt / FINE:.3f} s", flush=True)
    else:
        print("  fwd: does not reach the ground within the window", flush=True)
    return arm, res, snapshots


# -----------------------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------------------
def draw_start(ax, cx, cy, label=False):
    """The initial arm (centerline cx, cy) drawn like a snapshot in the gray of the time key at t = 0 (its color and
    opacity), under the snapshots; ``label``: 't = 0'."""
    from latentctl import style as S
    ax.plot(cx, cy, color=(0.32, 0.32, 0.32), alpha=0.18, lw=1.7, zorder=2, solid_capstyle="round")
    ax.plot(cx[-1], cy[-1], "o", color=(0.32, 0.32, 0.32), alpha=0.18, ms=3.0, mec="none", zorder=2)
    if label:
        ax.text(cx[-1] - 0.05, cy[-1] - 0.03, "$t = 0$", ha="right", va="top", fontsize=7, color=S.INK2)


def plot(profile):
    import matplotlib.pyplot as plt
    from matplotlib.path import Path
    from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch, Rectangle
    from latentctl import style as S

    arm, res, snapshots = closed_loops(profile)
    col = {"bwd": S.BWD, "fwd": S.FWD}
    labels = {loss: r"$\mathcal{L}_{\mathrm{%s}}$" % loss for loss in col}
    W, H = S.TEXT_W, 3.7
    fig = plt.figure(figsize=(W, H))
    box = lambda x, y, w, h: [x / W, y / H, w / W, h / H]                # inches -> figure fraction

    # ---- (A, B) the arm: centerlines at the snapshot times, opacity = time ------------------------
    xlim, ylim = (-1.0, 1.0), (-0.42, 1.06)
    aw = 1.9
    ah = aw * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    y_arm = H - 0.27 - ah
    x_margin = 0.1                                                       # [m] on both sides, at the same scale
    mw = x_margin * aw / (xlim[1] - xlim[0])                             # margin [in]
    for j, (loss, letter) in enumerate((("bwd", "A"), ("fwd", "B"))):
        ax = fig.add_axes(box(0.15 + 2.1 * j - mw, y_arm, aw + 2 * mw, ah))
        ax.set_xlim(xlim[0] - x_margin, xlim[1] + x_margin)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.plot(xlim, [0, 0], color=S.AXIS, lw=0.6, zorder=1)
        ax.plot([0, 0], [0, 1.0], color=S.AXIS, lw=0.7, ls=(0, (2, 2)), zorder=1)
        ax.add_patch(Rectangle((-0.12, -0.05), 0.24, 0.05, color=S.INK, lw=0, zorder=20))
        xs = res[loss]["xs"]
        # a panel may have its own snapshot times (res[loss]['snapshots'], states every res[loss]['dt'] s; own time key)
        snaps = np.round(np.asarray(res[loss].get("snapshots", snapshots)) / res[loss].get("dt", arm.dt)).astype(int)
        stop = res[loss].get("stop")                  # step at which the drawing ends (x at the tip), or None
        frames = [(k, idx, "o") for k, idx in enumerate(snaps) if stop is None or idx < stop]
        if stop is not None:
            frames.append(((len(snaps) - 1) * stop / snaps[-1], stop, "x"))
        for k, idx, mk in frames:
            a = min(1.0, 0.18 + 0.82 * k / (len(snaps) - 1))
            cx, cy = arm.centerline(xs[int(idx)][:arm.n_sec])
            if k == 0:
                draw_start(ax, cx, cy, label=loss == "bwd")
                continue
            ax.plot(cx, cy, color=col[loss], alpha=a, lw=1.7, zorder=3 + k, solid_capstyle="round")
            ax.plot(cx[-1], cy[-1], mk, color=col[loss], **(dict(ms=3.0, mec="none", alpha=a, zorder=3 + k) if mk == "o"
                                                            else dict(ms=6.0, mew=1.6, zorder=30)))  # x: opaque, on top
        S.title(ax, letter, labels[loss], x=mw / (aw + 2 * mw))
    # time key in (A), and in (B) if it has its own snapshot times: opacity encodes time
    keys = [(0, snapshots[-1])] + ([(1, res["fwd"]["snapshots"][-1])] if "snapshots" in res["fwd"] else [])
    for j, t_max in keys:
        kax = fig.add_axes(box(0.15 + 2.1 * j + aw - 0.62, y_arm + 0.06, 0.42, 0.055))
        rgba = np.zeros((1, 128, 4))
        rgba[..., :3] = 0.32
        rgba[..., 3] = 0.18 + 0.82 * np.linspace(0, 1, 128)[None, :]
        kax.imshow(rgba, aspect="auto", extent=(0, 1, 0, 1))
        kax.axis("off")
        kw = dict(fontsize=7, color=S.INK2, va="center", transform=kax.transAxes)
        kax.text(-0.08, 0.5, "$0$", ha="right", **kw)
        kax.text(1.08, 0.5, f"${round(t_max, 2):g}$ s", ha="left", **kw)

    # ---- (C, D) spectra over the open-loop spectrum ------------------------------------------------
    ev = np.linalg.eigvals(arm.A)
    distinct = []
    for z in sorted(ev, key=abs):
        if all(abs(z - w) > 0.04 for w in distinct):
            distinct.append(z)
    distinct = np.array(distinct)
    sw = 1.42
    y_sp = 0.36
    for j, (letter, title) in enumerate((("C", "latent spectrum"), ("D", "closed-loop spectrum"))):
        ax = fig.add_axes(box(0.42 + 2.1 * j, y_sp, sw, sw))
        S.unit_circle(ax, lim=1.28)
        ax.scatter(distinct.real, distinct.imag, s=50, facecolor="none", edgecolor=S.MUTED, lw=0.8, zorder=3)
        for loss, marker, size in (("bwd", "o", 22), ("fwd", "D", 15)):
            r = res[loss]
            z = np.linalg.eigvals(r["L"]) if letter == "C" else np.linalg.eigvals(arm.A + arm.B @ r["K"] @ r["E"])
            ax.scatter(z.real, z.imag, s=size, marker=marker, color=col[loss], edgecolor=S.SURFACE, lw=0.5,
                       zorder=5 if loss == "bwd" else 6)
        ax.tick_params(labelsize=7)
        ax.text(0.0, 0.98, r"$\sigma(J_{a_z}(0))$" if letter == "C" else r"$\sigma(A{+}BKJ_E(0))$",   # clear of x = 0
                transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=S.INK2)
        S.title(ax, letter, title)
    hs = [S.handle(S.BWD, marker="o", ls="none", ms=4.6, label=labels["bwd"]),
          S.handle(S.FWD, marker="D", ls="none", ms=3.8, label=labels["fwd"]),
          S.handle(S.MUTED, marker="o", ls="none", ms=5.4, mfc="none", mec=S.MUTED, mew=0.8,
                   label=r"open loop $\sigma(A)$")]
    fig.legend(handles=hs, loc="lower center", bbox_to_anchor=((0.15 + 2.1 + aw) / 2 / W, 0.0), ncol=3,
               columnspacing=1.8, handletextpad=0.3)

    # ---- (E) latent feedback loop --------------------------------------------------------------------
    ex, ew = 4.6, 1.85
    ax = fig.add_axes(box(ex, 0.08, ew, H - 0.35))
    h = H - 0.35
    ax.set_xlim(0, ew)
    ax.set_ylim(0, h)
    ax.axis("off")
    S.title(ax, "E", "latent feedback loop")
    cx, bh = 0.84, 0.48
    ys = np.linspace(h - 0.36, 0.30, 4)
    spec = (("plant", r"$x_{t+1} = f(x_t, u_t)$", "plant", S.GRAY_TINT, 1.42),
            ("enc", r"$z_t = J_E(0)\,x_t$", "encoder", S.BLUE_TINT, 1.42),
            ("ctrl", r"$u_t = K z_t$", "controller", S.SURFACE, 1.10),
            ("lat", r"$z_{t+1} = f_z(z_t, u_t)$", "latent model", S.BLUE_TINT, 1.42))
    for (key, eq, name, fc, bw), yc in zip(spec, ys):
        if key == "enc":
            ax.add_patch(Polygon([(cx - bw / 2, yc + bh / 2), (cx + bw / 2, yc + bh / 2), (cx + 0.38, yc - bh / 2),
                                  (cx - 0.38, yc - bh / 2)], closed=True, fc=fc, ec=S.INK2, lw=0.8))
        else:
            ax.add_patch(FancyBboxPatch((cx - bw / 2, yc - bh / 2), bw, bh, boxstyle="round,pad=0,rounding_size=0.05",
                                        fc=fc, ec=S.INK2, lw=0.8))
        ax.text(cx, yc + 0.075, eq, ha="center", va="center", fontsize=7.6, color=S.INK)
        ax.text(cx, yc - 0.13, name, ha="center", va="center", fontsize=6.8, color=S.INK2)

    def arrow(pts, ls="-"):
        ax.add_patch(FancyArrowPatch(path=Path(pts), arrowstyle="-|>,head_length=3.5,head_width=2",
                                     color=S.INK2, lw=0.9, ls=ls, shrinkA=0, shrinkB=0, joinstyle="miter"))

    yp, ye, yk, yl = ys
    lab = dict(fontsize=7.6, color=S.INK, va="center")
    arrow([(cx, yp - bh / 2), (cx, ye + bh / 2)])
    ax.text(cx + 0.06, (yp + ye) / 2, r"$x_t$", ha="left", **lab)
    arrow([(cx, ye - bh / 2), (cx, yk + bh / 2)])
    ax.text(cx + 0.06, (ye + yk) / 2, r"$z_t$", ha="left", **lab)
    xr = cx + 0.86
    arrow([(cx + 0.55, yk), (xr, yk), (xr, yp), (cx + 0.71, yp)])
    ax.text(xr - 0.05, (yk + yp) / 2, r"$u_t$", ha="right", **lab)
    arrow([(cx, yl + bh / 2), (cx, yk - bh / 2)], ls=(0, (3, 2)))
    ax.text(cx + 0.06, (yl + yk) / 2, "LQR", ha="left", fontsize=7, color=S.INK2, va="center")
    S.save(fig, io.figure_path(profile, "arm_example"))


def main():
    def extra(ap):
        ap.add_argument("--worker", default=None, help="internal: train one model (a task of TASKS)")
        ap.add_argument("--jobs", type=int, default=2)
    args, cfg = io.cli(__doc__, CONFIGS, extra)
    out = io.results_dir(args.profile, "motivating_example")
    if args.worker:
        return nonlinear_models.worker(args.worker, cfg, out)
    if args.stage in ("run", "all"):
        io.run_workers("experiments.motivating_example", args.profile, list(TASKS.values()), args.jobs,
                       os.path.join(out, "logs"), args.cpus)
    if args.stage in ("plot", "all"):
        plot(args.profile)


if __name__ == "__main__":
    main()
