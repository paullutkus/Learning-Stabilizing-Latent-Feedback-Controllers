"""Non-linear latent models and plant (Sec. 6.3): tanh-MLP vs linear models on the flexible arm.

Plant: the arm of the motivating example (n = 40, p = 1, one unstable eigenvalue 1.129): the nonlinear arm
(latentctl.systems), in coordinates scaled by eps (data.arm_dataset; eps = 0.01: joint angles of ~0.6 deg at
t = 0; at T = 40 the unstable mode saturates and about half of the trajectories fall over).  Data set s:
N = 2000 trajectories, seed 1000 + s, x_0 ~ N(0, I), u_t ~ N(0, 0.5^2 / T) in scaled coordinates, regenerated
at every T with the same x_0.

Models (latentctl.mlp): E, D, f_z, g_z tanh MLPs (2 x 64) with z+ = f_z(z) + g_z(z) u, or the linear
(E, D, L, H); m = 4.  Loss: empirical L_bwd^(T) = J_bwd + lam_fwd J_fwd + lam_reg R(E) with
R(E) = mean_x ||I - J_E(x) J_E(x)^T||_F^2 over the training states, lam_fwd = lam_reg = 1.  Forward
term: the MLPs use teacher forcing, the one-step prediction from every observed state
(``fwd_horizon_mlp = "tf"``); the linear models the one-step term from x_0, J_fwd^(1), as in
Fig. exp_arm_recovery.

Protocol (one task per model, data set [and eps]): the horizons T in ``Ts`` are solved in increasing order,
as in Fig. exp_arm_recovery.
  MLP     cold: ``restarts`` random inits (fresh at every T) trained for ``burn`` Adam steps; the one with
          the lowest full-data objective (latentctl.mlp.full_objective) is trained to ``steps``.
          warm: the model selected at the previous T (rescaled to this T's data scales) trained for
          ``warm_steps`` at peak lr ``warm_lr``.  The lower full-data objective is reported.
          Minibatch Adam (128 trajectories, warmup + cosine, clip 10), float32; diagnostics in float64.
  linear  the finite-sample solver of Fig. exp_arm_recovery (sweeps.emp_path: full batch, float64,
          Adam + L-BFGS, warm/cold path) from ``lin_inits`` inits; the lowest objective is reported.
The data scale sweep (``eps_sweep``) repeats the protocol at larger eps (tasks '...:e<eps>', results in
results/<profile>/nonlinear_eps/; the tables show ``eps_table_Ts``).  Every record keeps the selected
model's parameters (``params``; MLP: with ``scales``).

Diagnostics at the equilibrium, from the linearization (J_E(0), J_D(0), J_fz(0), g_z(0)):
    sub_dist   Delta(G_+^R(A^T); row J_E(0))   (capture: < 0.5)
    lam_err    |mu - 1.129| / 1.129, mu the eigenvalue of J_fz(0) nearest the unstable eigenvalue
    H0_err     |w^T g_z(0) - H_0| / |H_0|, w the left eigenvector of mu scaled so that the state
               functional e = J_E(0)^T w satisfies e . D_0 = 1 (gauge-free comparison with the
               canonical unstable variables (E_0, D_0, L_0, H_0), Appendix I)
    rho_cl     rho(A + B K J_E(0)), K = LQR on (J_fz(0), g_z(0)): the linearized closed loop of u = K E(x)
Closed loops on the nonlinear arm (control.standard_ics): u = sat(K E(x)) and u = sat(K J_E(0) x),
|u| <= 40 N m, the same 8 ICs scaled to each tip displacement 2 / 10 / 20 / 30 / 40 / 60 / 80 cm (the
linearized tip coordinate), 12 s.

The task 'mlp:nonlinear:s<seed>' may carry the suffixes ':Lf' (decoder-free L_fwd: J_bwd dropped), ':sr'
(SIGReg on the trajectory states instead of R_orth) and ':hT' / ':h1' / ':htf' (forward term J_fwd^(T) /
J_fwd^(1) / teacher forcing); experiments.motivating_example trains its two models this way.

Figure: arm_nonlinear_models -> fig:exp_arm_nonlinear ((b), (c): captured models only, the T axis from the first
capture on; (e) one dot per data set: ICs stabilized at the largest T);  tables nonlinear_conjugacy ->
tab:exp_conjugacy_nonlinear, nonlinear_eps_tips -> tab:exp_eps_tips; results/<profile>/nonlinear/SUMMARY.md.

    python -m experiments.nonlinear_models [--profile paper|fast] [--stage run|plot|all] [--jobs N]
"""

from __future__ import annotations

import os

import numpy as np

from latentctl import io

LIN_OPT = dict(protocol="restart", n_adam=3000, n_lbfgs=2000, n_warm=2000, lr=1e-2, lr_end=1e-5)
CONFIGS = {
    "paper": dict(m=4, eps=0.01, N=2000, sx=1.0, su=0.5, lam_fwd=1.0, lam_reg=1.0, fwd_horizon=1,
                  fwd_horizon_mlp="tf", width=64, depth=2, batch=128, lr=3e-3, lr_end=1e-5, clip=10.0, n_reg=256,
                  n_sel=2048, log_every=500, tips=(0.02, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80),
                  table_tips=(0.02, 0.10, 0.30), n_ic=8, Ts=(1, 2, 5, 7, 10, 14, 20, 28, 40), seeds=(0, 1, 2, 3, 4),
                  restarts=8, burn=1000, steps=30000, warm_steps=10000, warm_lr=1e-3, lin_inits=8, lin_opt=LIN_OPT,
                  eps_sweep=(0.03, 0.1), eps_Ts=(1, 2, 5, 7, 10, 14, 20, 28, 40), eps_table_Ts=(10, 20, 40)),
}
# reduced sweeps: 4 horizons, 2 data sets, 8k + 3k MLP steps, 4 linear inits, eps = 0.1 at T = 10, 20 only
CONFIGS["fast"] = dict(CONFIGS["paper"], Ts=(5, 10, 20, 40), seeds=(0, 1), steps=8000, warm_steps=3000, lin_inits=4,
                       lin_opt=dict(LIN_OPT, n_lbfgs=1500, n_warm=1500), eps_sweep=(0.1,), eps_Ts=(10, 20))


def tasks(cfg):
    """Longest first: MLP paths (eps sweep included), then the linear models."""
    out = []
    for model in ("mlp", "linear"):
        out += [f"{model}:nonlinear:s{s}" for s in cfg["seeds"]]
        out += [f"{model}:nonlinear:s{s}:e{e:g}" for e in cfg["eps_sweep"] for s in cfg["seeds"]]
    return out


def parse(task, cfg):
    """(model, seed, eps, forward term, loss 'bwd' | 'fwd', regularizer 'orth' | 'sigreg') of a task."""
    model, plant, s, *rest = task.split(":")
    assert plant == "nonlinear", task
    eps = next((float(v[1:]) for v in rest if v[:1] == "e"), None)
    fwd = cfg["fwd_horizon_mlp"] if model == "mlp" else cfg["fwd_horizon"]
    for v, h in (("h1", 1), ("hT", "T"), ("htf", "tf")):
        if v in rest:
            fwd = h
    return model, int(s[1:]), eps, fwd, "fwd" if "Lf" in rest else "bwd", "sigreg" if "sr" in rest else "orth"


# -----------------------------------------------------------------------------------------
# Equilibrium diagnostics
# -----------------------------------------------------------------------------------------
def equilibrium_metrics(JE, JD, Jf, g0, A, B, tg):
    import scipy.linalg as sla
    from latentctl.control import dlqr
    from latentctl.linalg import subspace_distance
    if not all(np.all(np.isfinite(a)) for a in (JE, JD, Jf, g0)):
        return dict(finite=False, sub_dist=1.0, lam_err=np.nan, H0_err=np.nan, rho_cl=np.nan, K=None)
    out = dict(finite=True, sub_dist=subspace_distance(tg["Q_plus"], JE, "fro"))
    ev = np.linalg.eigvalsh(JE @ JE.T)
    out["EEt_min"], out["EEt_max"] = float(ev[0]), float(ev[-1])
    out["fwd_res"] = float(np.linalg.norm(JE @ A - Jf @ JE))
    lam_u = float(np.real(tg["0"]["L"][0, 0]))
    E0, D0, H0 = tg["0"]["E"][0], tg["0"]["D"][:, 0], float(tg["0"]["H"][0, 0])
    mu = np.linalg.eigvals(Jf)
    i = int(np.argmin(np.abs(mu - lam_u)))
    out["lam_hat"], out["lam_err"] = complex(mu[i]), float(abs(mu[i] - lam_u) / abs(lam_u))
    wl, Wl = sla.eig(Jf.T)
    w = Wl[:, int(np.argmin(np.abs(wl - mu[i])))]
    e = JE.T @ w
    s = e @ D0
    if abs(s) > 1e-12:
        e, w = e / s, w / s
        out["E0_err"] = float(np.linalg.norm(e - E0) / np.linalg.norm(E0))
        out["H0_err"] = float(abs(w @ g0[:, 0] - H0) / abs(H0))
    else:
        out["E0_err"] = out["H0_err"] = np.nan
    K = dlqr(Jf, g0)
    out["K"] = K
    out["rho_cl"] = np.nan if K is None else float(np.max(np.abs(np.linalg.eigvals(A + B @ K @ JE))))
    out["eigs_f"] = mu
    return out


def closed_loops(arm, cfg, K, JE, enc=None):
    """Success fractions on the nonlinear arm {(law, 'nonlinear', tip): value}; law KE = sat(K E(x)),
    KJE = sat(K J_E(0) x) (the same for a linear model, ``enc = None``)."""
    from latentctl.control import simulate, standard_ics
    sims = {}
    if K is None:
        return sims
    for tip, X0 in standard_ics(arm, cfg["tips"], cfg["n_ic"], shared=True).items():
        v = float(simulate(arm, K @ JE, X0)[2].mean())
        sims[("KJE", "nonlinear", tip)] = v
        sims[("KE", "nonlinear", tip)] = v if enc is None else float(simulate(arm, None, X0, encoder=enc)[2].mean())
    return sims


# -----------------------------------------------------------------------------------------
# Workers: one (model, data set[, eps]) along the horizon path
# -----------------------------------------------------------------------------------------
def worker(task, cfg, out):
    model_kind, s, eps, fwd, loss, reg = parse(task, cfg)
    cfg = dict(cfg, eps=cfg["eps"] if eps is None else eps, loss=loss, reg=reg)
    Ts = cfg["Ts"] if eps is None else cfg["eps_Ts"]
    from latentctl.systems import flexible_arm
    from latentctl.linalg import canonical_targets
    from latentctl.data import arm_dataset, nonlinearity
    arm = flexible_arm()
    tg = canonical_targets(arm.A, arm.B, cfg["m"])
    cache = {}

    def data(T):
        if T not in cache:
            cache.clear()
            X, U = arm_dataset(arm, T, cfg["N"], seed=1000 + s, eps=cfg["eps"], sx=cfg["sx"], su=cfg["su"])
            cache[T] = (X, U, nonlinearity(arm, X, U, cfg["eps"]))
        return cache[T]

    run = worker_mlp if model_kind == "mlp" else worker_linear
    recs = run(arm, tg, data, Ts, s, fwd, cfg, task)
    dest = out if eps is None else io.results_dir(os.path.basename(os.path.dirname(out)), os.path.basename(out) + "_eps")
    io.dump(dict(task=task, model=model_kind, plant="nonlinear", seed=s, eps=cfg["eps"], fwd=fwd, loss=loss, reg=reg,
                 cfg=cfg, recs=recs),
            os.path.join(dest, task.replace(":", "__") + ".pkl"))


def report(task, T, rec):
    print(f"  {task} T={T:2d} [{rec['branch']}]  obj {rec['obj']:.3f}  Delta={rec['sub_dist']:.2e}  "
          f"rel err lam={rec['lam_err']:.1e}  H0={rec.get('H0_err', np.nan):.1e}  rho_cl={rec['rho_cl']:.4f}  "
          f"KE(x) nonlinear arm {[round(rec['sims'].get(('KE', 'nonlinear', tp), 0.0), 2) for tp in (0.02, 0.10, 0.30)]}",
          flush=True)


def worker_mlp(arm, tg, data, Ts, s, fwd, cfg, task):
    import time
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import optax
    from latentctl.mlp import Model, make_loss, full_objective, rescale
    F32 = jnp.float32
    R, chunk = cfg["restarts"], cfg["log_every"]
    tree = jax.tree_util.tree_map

    def optimizer(lr, steps):
        sched = optax.warmup_cosine_decay_schedule(0.0, lr, max(1, steps // 50), steps, cfg["lr_end"])
        return optax.apply_if_finite(optax.chain(optax.clip_by_global_norm(cfg["clip"]), optax.adam(sched)), 50)

    recs, prev = {}, None
    for T in Ts:
        t0 = time.time()
        Xn, Un, st = data(T)
        X, U = jnp.asarray(Xn, F32), jnp.asarray(Un, F32)
        model = Model(arm.n, cfg["m"], arm.p, cfg["width"], cfg["depth"], F32)
        model.set_scales(Xn)
        lr_kw = dict(loss=cfg["loss"], reg=cfg["reg"])
        loss_terms = make_loss(model, cfg["lam_fwd"], cfg["lam_reg"], cfg["n_reg"], fwd, **lr_kw)
        fobj = jax.jit(full_objective(model, cfg["lam_fwd"], cfg["lam_reg"], fwd, cfg["n_sel"], **lr_kw))

        def chunk_fn(opt):
            def run_chunk(th, st_, key, X, U):
                def one(c, k):
                    th, st_ = c
                    k1, k2 = jax.random.split(k)
                    idx = jax.random.randint(k1, (cfg["batch"],), 0, X.shape[0])
                    (_, aux), gr = jax.value_and_grad(loss_terms, has_aux=True)(th, X[idx], U[idx], k2)
                    up, st_ = opt.update(gr, st_, th)
                    return (optax.apply_updates(th, up), st_), aux
                (th, st_), aux = jax.lax.scan(one, (th, st_), jax.random.split(key, chunk))
                return th, st_, tree(jnp.mean, aux)
            return run_chunk

        lin = lambda th: model.linearize(tree(np.asarray, th))
        # ---- cold: multi-start burn-in, selection by the full-data objective, train to the end -------
        opt = optimizer(cfg["lr"], cfg["steps"])
        step = chunk_fn(opt)
        train_R, train_1 = jax.jit(jax.vmap(step, in_axes=(0, 0, 0, None, None))), jax.jit(step)
        keys0 = jnp.stack([jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(s), r), T) for r in range(R)])
        th = jax.vmap(model.init)(keys0)
        st_ = jax.vmap(opt.init)(th)
        kb = jnp.stack([jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(10_000 + s), r), T) for r in range(R)])
        for c in range(cfg["burn"] // chunk):
            th, st_, _ = train_R(th, st_, jax.vmap(lambda k: jax.random.fold_in(k, c))(kb), X, U)
        burn_obj = np.array([float(fobj(tree(lambda a: a[r], th), X, U)) for r in range(R)])
        burn_sub = np.array([equilibrium_metrics(*lin(tree(lambda a: a[r], th)), arm.A, arm.B, tg)["sub_dist"]
                             for r in range(R)])
        sel = int(np.argmin(np.where(np.isfinite(burn_obj), burn_obj, np.inf)))
        thc, stc, kc = (tree(lambda a: a[sel], x) for x in (th, st_, kb))
        aux = None
        for c in range(cfg["burn"] // chunk, cfg["steps"] // chunk):
            thc, stc, aux = train_1(thc, stc, jax.random.fold_in(kc, c), X, U)
        cands = {"cold": (thc, float(fobj(thc, X, U)), float(aux["total"]))}
        # ---- warm: the previous horizon's selected model, rescaled to this horizon's data scales ------
        if prev is not None:
            optw = optimizer(cfg["warm_lr"], cfg["warm_steps"])
            train_w = jax.jit(chunk_fn(optw))
            thw = tree(lambda a: jnp.asarray(a, F32), rescale(prev[0], prev[1], model.scales))
            stw = optw.init(thw)
            kw = jax.random.fold_in(jax.random.PRNGKey(20_000 + s), T)
            for c in range(cfg["warm_steps"] // chunk):
                thw, stw, aux = train_w(thw, stw, jax.random.fold_in(kw, c), X, U)
            cands["warm"] = (thw, float(fobj(thw, X, U)), float(aux["total"]))
        branch = min(cands, key=lambda k: cands[k][1] if np.isfinite(cands[k][1]) else np.inf)
        thb, obj, loss = cands[branch]
        JE, JD, Jf, g0 = lin(thb)
        mt = equilibrium_metrics(JE, JD, Jf, g0, arm.A, arm.B, tg)
        th64 = tree(lambda a: jnp.asarray(np.asarray(a), jnp.float64), thb)
        enc64 = jax.jit(lambda th_, x: model.enc(th_, x))
        enc = lambda x: cfg["eps"] * np.asarray(enc64(th64, jnp.asarray(x / cfg["eps"]))) @ mt["K"].T
        mt["sims"] = closed_loops(arm, cfg, mt["K"], JE, enc)
        other = {k: dict(obj=v[1], sub_dist=equilibrium_metrics(*lin(v[0]), arm.A, arm.B, tg)["sub_dist"],
                         params=tree(np.asarray, v[0])) for k, v in cands.items()}
        mt.update(T=T, seed=s, branch=branch, obj=obj, loss=loss, cands=other, burn_obj=burn_obj, burn_sub=burn_sub,
                  burn_sel=sel, scales=model.scales, params=tree(np.asarray, thb), time=time.time() - t0, **st)
        mt.pop("K")
        recs[T] = mt
        report(task, T, mt)
        prev = (tree(np.asarray, thb), model.scales)
    return recs


def worker_linear(arm, tg, data, Ts, s, fwd, cfg, task):
    from latentctl.sweeps import emp_path
    if cfg["loss"] != "bwd" or cfg["reg"] != "orth" or fwd == "tf":
        raise ValueError("the linear-model tasks train L_bwd^(T) with R_orth and J_fwd^(1) or J_fwd^(T) only")
    problems = [(0, cfg["N"], r) for r in range(cfg["lin_inits"])]
    recs = {}
    for T, th, info in emp_path(lambda T: tuple(a[None] for a in data(T)[:2]), cfg["m"], Ts, "bwd", "orth",
                                cfg["lam_reg"], problems, cfg["lin_opt"], lam_fwd=cfg["lam_fwd"], fwd_horizon=fwd,
                                init_seed=1000, data_ids=[s]):
        obj = np.where(np.isfinite(info["obj_direct"]), info["obj_direct"], np.inf)
        i = int(np.argmin(obj))
        E, D, L, H = (np.asarray(x[i]) for x in th)
        mt = equilibrium_metrics(E, D, L, H, arm.A, arm.B, tg)
        mt["sims"] = closed_loops(arm, cfg, mt["K"], E)
        mt.pop("K")
        subs = [equilibrium_metrics(*(np.asarray(x[r]) for x in th), arm.A, arm.B, tg)["sub_dist"] for r in range(len(obj))]
        mt.update(T=T, seed=s, branch="warm" if info["from_warm"][i] else "cold", obj=float(obj[i]), loss=float(obj[i]),
                  inits_obj=np.asarray(info["obj_direct"]), inits_sub=np.array(subs),
                  inits_warm=np.asarray(info["from_warm"]), params=(E, D, L, H), **data(T)[2])
        recs[T] = mt
        report(task, T, mt)
    return recs


def linearization(model_kind, rec, cfg, arm):
    """(J_E(0), J_D(0), J_fz(0), g_z(0)) of a saved model (MLP: float64 copy of its parameters and scales; plus the
    model and parameters for the encoder), or ((E, D, L, H), None, None) for a linear one."""
    if model_kind != "mlp":
        return tuple(np.asarray(x) for x in rec["params"]), None, None
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from latentctl.mlp import Model
    model = Model(arm.n, cfg["m"], arm.p, cfg["width"], cfg["depth"], jnp.float64)
    model.s_x, model.s_z = rec["scales"]
    th64 = jax.tree_util.tree_map(lambda a: jnp.asarray(np.asarray(a), jnp.float64), rec["params"])
    return model.linearize(th64), model, th64


# -----------------------------------------------------------------------------------------
# Figure and tables
# -----------------------------------------------------------------------------------------
def load(profile):
    """{(model, eps, T): records of the data sets, by seed}."""
    import glob
    runs = {}
    for sub in ("nonlinear", "nonlinear_eps"):
        for f in glob.glob(os.path.join(io.results_dir(profile, sub), "*.pkl")):
            r = io.load(f)
            for T, rec in r["recs"].items():
                runs.setdefault((r["model"], round(r["eps"], 6), T), []).append(rec)
    for v in runs.values():
        v.sort(key=lambda rec: rec["seed"])
    return runs


def col(recs, key, captured_only=False):
    v = np.array([np.real(r.get(key, np.nan)) for r in recs], float)
    if captured_only:
        v = np.where(np.array([r["sub_dist"] for r in recs]) < 0.5, v, np.nan)
    return v


def plot(profile):
    import matplotlib.pyplot as plt
    from latentctl import style as S
    from latentctl.systems import flexible_arm
    from latentctl.linalg import alg1_latent
    from latentctl.control import dlqr

    runs = load(profile)
    if not runs:
        return
    cfg = CONFIGS[profile]
    eps0 = round(cfg["eps"], 6)
    arm = flexible_arm()
    E, L, H = alg1_latent(arm.A, arm.B, cfg["m"])
    rho_ref = float(np.max(np.abs(np.linalg.eigvals(arm.A + arm.B @ dlqr(L, H) @ E))))
    COL = {"linear": S.BWD, "mlp": S.MLP}
    MK = {"linear": "o", "mlp": "s"}
    NAME = {"linear": "linear model", "mlp": "MLP model"}      # forward terms (J_fwd^(1) / teacher forcing): in the text

    fig = plt.figure(figsize=(S.TEXT_W, 3.85))
    gs = fig.add_gridspec(2, 3, left=0.085, right=0.985, top=0.93, bottom=0.115, wspace=0.42, hspace=0.62)
    ax = {k: fig.add_subplot(gs[i, j]) for k, (i, j) in zip("abcdef", [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)])}
    Ts = sorted({k[2] for k in runs if k[1] == eps0})
    items = (("a", "sub_dist", True, False), ("b", "lam_err", True, True), ("c", "H0_err", True, True),
             ("d", "rho_cl", False, False))
    span = {key: [] for key, *_ in items}                                # dodged T of the plotted points
    for k, model in enumerate(("linear", "mlp")):
        T = [t for t in Ts if (model, eps0, t) in runs]
        if not T:
            continue
        dodge = 2.0 ** (0.08 * (k - 0.5))                                # the dodge rule of every figure
        for key, c, log, masked in items:
            vals = [col(runs[(model, eps0, t)], c, masked) for t in T]
            stat = lambda f: np.array([f(v) if np.any(np.isfinite(v)) else np.nan for v in vals])
            med = stat(np.nanmedian)
            x = np.array(T, float) * dodge
            if np.any(np.isfinite(med)):
                span[key] += list(x[np.isfinite(med)])
            lohi = np.array([S.spread(v) for v in vals])                      # over data sets (S.SPREAD)
            S.errbar(ax[key], x, lohi[:, 0], lohi[:, 1], COL[model])
            ax[key].plot(x, med, color=COL[model], marker=MK[model], ms=3.4, lw=1.2, mec=S.SURFACE, mew=0.5,
                         zorder=3)
    for key, c, log, masked in items:
        a = ax[key]
        a.set_xscale("log")
        if masked and span[key]:                                         # (b), (c): from the first capture on
            lo, hi = min(span[key]), max(span[key])
            S.log_ticks(a, "x", [t for t in Ts if lo / 1.2 <= t <= hi * 1.2])
            a.set_xlim(lo / 2 ** 0.15, hi * 2 ** 0.15)
        else:
            S.log_ticks(a, "x", Ts, labels=[t for t in Ts if t in (1, 2, 5, 10, 20, 40)])
        a.set_xlabel(r"horizon $T$")
        if log:
            a.set_yscale("log")
            S.pow10_ticks(a)
    ax["a"].set_ylabel(r"$\Delta(\mathcal{G}^{\mathbb{R}}_+(A^\top);\ \mathrm{row}\,J_E(0))$")
    ax["b"].set_ylabel(r"$|\hat\lambda_u - \lambda_u| / \lambda_u$")
    ax["c"].set_ylabel(r"$|\hat h - H_0| / |H_0|$")
    ax["d"].set_ylabel(r"$\rho(A + BKJ_E(0))$")
    ax["d"].axhline(1.0, color=S.AXIS, lw=0.7, zorder=1)
    ax["d"].axhline(rho_ref, color=S.INK, lw=0.8, ls=(0, (4, 1.5, 1, 1.5)), zorder=1)
    ax["d"].set_ylim(0.85, 1.16)
    S.title(ax["a"], "a", "unstable subspace")
    S.title(ax["b"], "b", r"unstable eigenvalue of $J_{a_z}(0)$")
    S.title(ax["c"], "c", r"unstable input gain $g_z(0)$")
    S.title(ax["d"], "d", "linearized closed loop")

    # (e) non-linear arm at the largest T: one dot per data set, the number of initial states from which its
    # controller stabilizes the arm; tied data sets are stacked vertically about their value (translucent)
    Tm = max(Ts)
    ae = ax["e"]
    n_ic = cfg["n_ic"]
    w = 0.22                                                             # lanes flush at each tip displacement
    lanes = ((("mlp", "KE"), -w), (("mlp", "KJE"), 0.0), (("linear", "KE"), w))       # linear: KE(x) = KJ_E(0)x
    dy = 0.24
    for j, tp in enumerate(cfg["tips"]):
        for (model, law), dx in lanes:
            recs = runs.get((model, eps0, Tm))
            if not recs:
                continue
            y = np.array([round(r["sims"].get((law, "nonlinear", tp), 0.0) * n_ic) for r in recs], float)
            yo = y.copy()
            for v in np.unique(y):
                idx = np.flatnonzero(y == v)
                yo[idx] = v + dy * (np.arange(len(idx)) - (len(idx) - 1) / 2)
            filled = law == "KE"
            ae.plot(np.full(len(y), j + dx), yo, ls="none", marker=MK[model], ms=2.9 if filled else 2.6,
                    color=COL[model], mfc=COL[model] if filled else "none", mec=COL[model],
                    mew=0.3 if filled else 0.8, alpha=0.55, zorder=3)
    ae.set_xticks(range(len(cfg["tips"])))
    ae.set_xticklabels([f"{100 * t:g}" for t in cfg["tips"]])
    ae.set_xlim(-0.5, len(cfg["tips"]) - 0.5)
    ae.set_ylim(-0.9, n_ic + 0.9)
    ae.set_yticks([0, n_ic // 2, n_ic])
    ae.grid(axis="x", visible=False)
    ae.set_xlabel("tip displacement [cm]")
    ae.set_ylabel(f"initial states stabilized\n(of {n_ic}, per data set)")
    S.title(ae, "e", f"non-linear stabilization, $T = {Tm}$")

    af = ax["f"]
    af.axis("off")
    h1 = [S.handle(COL[m], marker=MK[m], ms=3.6, lw=1.2, label=NAME[m]) for m in ("linear", "mlp")]
    h3 = [S.handle(S.INK, ls=(0, (4, 1.5, 1, 1.5)), lw=0.9, label="(d) Alg. 1, known $(A, B)$"),
          S.handle(S.INK2, ls="none", marker="o", ms=3.6, mec=S.INK2, mew=0.4, alpha=0.55, label=r"(e) $u = KE(x)$"),
          S.handle(S.INK2, ls="none", marker="o", ms=3.2, mfc="none", mec=S.INK2, mew=0.9, alpha=0.55,
                   label=r"(e) $u = KJ_E(0)x$")]
    l1 = af.legend(handles=h1, loc="upper left", bbox_to_anchor=(-0.2, 1.1), alignment="left", handlelength=2.6)
    af.add_artist(l1)
    af.legend(handles=h3, loc="upper left", bbox_to_anchor=(-0.2, 0.62), alignment="left", handlelength=2.6)
    S.save(fig, io.figure_path(profile, "arm_nonlinear_models"))
    summary(profile, runs, cfg, rho_ref)


def summary(profile, runs, cfg, rho_ref):
    tips = cfg["table_tips"]
    succ = lambda recs, law: "/".join(f"{np.mean([r['sims'].get((law, 'nonlinear', tp), 0.0) for r in recs]):.2f}" for tp in tips)
    lines = ["| model | eps | T | n | captured | locally stable | Delta (median) | rel err lam (captured) | "
             "H0 err (captured) | rho_cl (median) | branch warm | data dev / fallen | KE succ 2/10/30 cm | KJE succ |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(runs):
        recs = runs[key]
        cap = col(recs, "sub_dist") < 0.5
        med = lambda c, m=False: np.nanmedian(col(recs, c, m)) if np.any(np.isfinite(col(recs, c, m))) else np.nan
        lines.append(f"| {key[0]} | {key[1]:g} | {key[2]} | {len(recs)} | {cap.sum()} | "
                     f"{int(np.sum(col(recs, 'rho_cl') < 1))} | {med('sub_dist'):.2e} | {med('lam_err', True):.2e} | "
                     f"{med('H0_err', True):.2e} | {med('rho_cl'):.3f} | {sum(r['branch'] == 'warm' for r in recs)} | "
                     f"{np.mean([r['rel_dev'] for r in recs]):.3f} / {np.mean([r['fallen'] for r in recs]):.2f} | "
                     f"{succ(recs, 'KE')} | {succ(recs, 'KJE')} |")
    burn = ["", "MLP burn-in: restarts with Delta < 0.5 after burn-in per data set, and whether the selected one has it", ""]
    for key in sorted(k for k in runs if k[0] == "mlp"):
        burn.append(f"- eps={key[1]:g} T={key[2]}: " + ", ".join(
            f"{int(np.sum(r['burn_sub'] < 0.5))}/{len(r['burn_sub'])}{'*' if r['burn_sub'][r['burn_sel']] < 0.5 else ''}"
            for r in runs[key]) + "   cold/warm obj: " + ", ".join(
            "/".join(f"{r['cands'][b]['obj']:.1f}" for b in ("cold", "warm") if b in r["cands"]) for r in runs[key]))
    with open(os.path.join(io.results_dir(profile, "nonlinear"), "SUMMARY.md"), "w") as fh:
        fh.write(f"non-linear arm data; forward term: MLP {cfg['fwd_horizon_mlp']}, linear {cfg['fwd_horizon']}; "
                 f"profile {profile}\n\n" + "\n".join(lines + burn)
                 + f"\n\nAlgorithm 1 with known (A, B): rho_cl = {rho_ref:.4f}\n")
    eps_tips_table(profile, runs, cfg)
    conjugacy_table(profile, runs, cfg)


def data_cells(recs):
    """[deviation from the linearized rollouts, fallen fraction] in % of the data sets of ``recs`` (mean), as cells."""
    dev = 100 * np.mean([r["rel_dev"] for r in recs])
    fal = 100 * np.mean([r["fallen"] for r in recs])
    return [f"${dev:.0f}$" if dev >= 10 else (f"${dev:.2g}$" if dev >= 0.1 else "$<0.1$"), f"${fal:.0f}$"]


def eps_tips_table(profile, runs, cfg):
    """tab:exp_eps_tips body: per (eps, T, controller) the success of u = KE(x), u = KJ_E(0) x (MLP) and u = KEx
    (linear) at every tip of ``tips`` (fractions of the n_ic ICs, mean over the data sets)."""
    tip_rows = []
    ctrls = (("mlp", "KE", r"MLP, $u = KE(x)$"), ("mlp", "KJE", r"MLP, $u = KJ_E(0)x$"), ("linear", "KE", r"linear, $u = KEx$"))
    for e in sorted({k[1] for k in runs}):
        for i, T in enumerate(cfg["eps_table_Ts"]):
            ml, li = runs.get(("mlp", e, T)), runs.get(("linear", e, T))
            if not ml or not li:
                continue
            for j, (model, law, name) in enumerate(ctrls):
                recs = ml if model == "mlp" else li
                lab = [f"${e:g}$" if i == j == 0 else "", f"${T}$" if j == 0 else "", name]
                tip_rows.append(" & ".join(lab + [f"{np.mean([r['sims'].get((law, 'nonlinear', tp), 0.0) for r in recs]):.2f}"
                                                  for tp in cfg["tips"]]) + r" \\")
        tip_rows.append(r"\midrule")
    while tip_rows and tip_rows[-1] == r"\midrule":
        tip_rows.pop()
    with open(io.table_path(profile, "nonlinear_eps_tips"), "w") as fh:
        fh.write("% generated by experiments.nonlinear_models (profile " + profile + "): stabilized fraction per tip "
                 + "/".join(f"{round(100 * tp)}" for tp in cfg["tips"]) + " cm\n" + "\n".join(tip_rows) + "\n")


def conjugacy_table(profile, runs, cfg):
    """tab:exp_conjugacy_nonlinear body: forward conjugacy (linalg.conjugacy) of the linearization
    (J_E(0), J_fz(0), g_z(0)) of the selected models, per (eps, T in eps_table_Ts, model): the data's deviation
    from linearized rollouts and fallen fraction (``data_cells``, first row of each (eps, T)), data sets capturing
    the unstable mode, medians over the data sets of Delta, lam_min(J_E J_E^T), r_A = ||E^A - L^E^||_F
    (orthonormal encoder rows), and over the capturing ones of r_u and r_h, and locally stabilizing data sets."""
    from latentctl.systems import flexible_arm
    from latentctl.linalg import canonical_targets, conjugacy
    arm = flexible_arm()
    lam_u = float(np.real(canonical_targets(arm.A, arm.B, cfg["m"])["0"]["L"][0, 0]))

    def sci(x):
        if not np.isfinite(x):
            return "--"
        if abs(x) >= 0.01:
            return f"${x:.2g}$"
        m_, e = f"{x:.1e}".split("e")
        return f"${m_}\\cdot 10^{{{int(e)}}}$"

    rows = []
    for e in sorted({k[1] for k in runs}):
        for i, T in enumerate(cfg["eps_table_Ts"]):
            for j, model in enumerate(("linear", "mlp")):
                recs = runs.get((model, e, T))
                if not recs:
                    continue
                res = []
                for r in recs:
                    (JE, _, Jf, g0), _, _ = linearization(model, r, cfg, arm)
                    res.append(dict(conjugacy(JE, Jf, g0, arm.A, arm.B, lam_u), sub_dist=r["sub_dist"],
                                    EEt_min=r.get("EEt_min", np.nan), rho_cl=r["rho_cl"]))
                cap = [q for q in res if q["sub_dist"] < 0.5]
                med = lambda qs, k: float(np.median([q[k] for q in qs])) if qs else np.nan
                n = len(res)
                cells = [f"${e:g}$" if i == j == 0 else "", f"${T}$" if j == 0 else ""] \
                    + (data_cells(recs) if j == 0 else ["", ""]) + ["MLP" if model == "mlp" else "linear",
                         f"{len(cap)}/{n}", sci(med(res, "sub_dist")), sci(med(res, "EEt_min")), sci(med(res, "res_A")),
                         sci(med(cap, "r_u")), sci(med(cap, "r_h")), f"{int(np.sum([q['rho_cl'] < 1 for q in res]))}/{n}"]
                rows.append(" & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    while rows and rows[-1] == r"\midrule":
        rows.pop()
    with open(io.table_path(profile, "nonlinear_conjugacy"), "w") as fh:
        fh.write("% generated by experiments.nonlinear_models (profile " + profile + "): forward conjugacy of the "
                 "linearizations at the equilibrium (linalg.conjugacy)\n" + "\n".join(rows) + "\n")


def main():
    def extra(ap):
        ap.add_argument("--worker", default=None, help="internal: run one task model:nonlinear:s<seed>[:e<eps>]")
        ap.add_argument("--jobs", type=int, default=8)
    args, cfg = io.cli(__doc__, CONFIGS, extra)
    out = io.results_dir(args.profile, "nonlinear")
    if args.worker:
        return worker(args.worker, cfg, out)
    if args.stage in ("run", "all"):
        io.run_workers("experiments.nonlinear_models", args.profile, tasks(cfg), args.jobs, os.path.join(out, "logs"),
                       args.cpus)
    if args.stage in ("plot", "all"):
        plot(args.profile)


if __name__ == "__main__":
    main()
