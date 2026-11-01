"""Horizon sweeps of the linear latent model (population and finite sample) and their scoring.

pop_path      minimize the population L_bwd^(T) / L_fwd^(T) along a path of horizons for a
              batch of random initializations (``optim.horizon_path``); init s at horizon T is
              drawn from fold_in(PRNGKey(1000 + s), T), a fresh cold start at every T.
emp_path      the same for the empirical losses of a bank of datasets, one problem per
              (dataset, N[, init r]); the dataset of stage T is regenerated with the stage's input
              schedule alpha_T = 1/T (same seed, so the same initial states), so every stage is an
              exact instance of the data model.  Init r of data set d at horizon T is drawn from
              fold_in(fold_in(PRNGKey(init_seed + d), r), T).
score_models  subspace recovery, conditioning, forward residual, and the closed loops of the
              lifted latent LQR and H-infinity gains on the linearized and the nonlinear arm.
mark_selected the lowest-objective run of each group (the reported model).
"""

from __future__ import annotations

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .control import simulate
from .losses import PopLoss, EmpLoss
from .optim import horizon_path, init_batch

OPT_KEYS = ("protocol", "n_adam", "n_lbfgs", "n_warm", "lr", "lr_end")


def pop_path(A, B, m, Ts, loss, reg, lam_reg, seeds, opt, sx=1.0, su=0.5, lam_fwd=1.0, fwd_horizon="T",
             vic_beta=1.0):
    """Yields (T, (E, D, L, H) with a leading seed axis, info incl. the loss terms)."""
    cache = {}

    def loss_at(T):
        pl = PopLoss(A, B, m, T, loss, reg, sx=sx, su=su, lam_fwd=lam_fwd, fwd_horizon=fwd_horizon,
                     vic_beta=vic_beta)
        cache[T] = pl
        return (lambda th, aux, c: pl.total(th, lam_reg, c)), pl.c

    def th0(T):
        keys = jnp.stack([jax.random.fold_in(jax.random.PRNGKey(1000 + s), T) for s in seeds])
        return init_batch(keys, A.shape[0], m, B.shape[1])

    for T, th, info in horizon_path(loss_at, Ts, th0, **{k: opt[k] for k in OPT_KEYS if k in opt}):
        terms = jax.jit(jax.vmap(cache[T].terms, in_axes=(0, None)))(tuple(jnp.asarray(x) for x in th), cache[T].c)
        info.update({k: np.asarray(v) for k, v in terms.items()})
        yield T, th, info


def emp_path(make_data, m, Ts, loss, reg, lam_reg, problems, opt, lam_fwd=1.0, fwd_horizon="T",
             vic_beta=1.0, init_seed=1000, data_ids=None):
    """make_data(T) -> (Xs (S, Nmax, T+1, n), Us (S, Nmax, T, p)); problems: [(sidx, N[, r]), ...]
    (r: initialization index, default 0); data_ids[sidx] names the data set in the init keys
    (default sidx).  Yields (T, params with a leading problem axis, info incl. exact loss terms)."""
    cache = {}

    def loss_at(T):
        Xs, Us = make_data(T)
        el = EmpLoss(Xs, Us, m, loss, reg, lam_fwd=lam_fwd, fwd_horizon=fwd_horizon, vic_beta=vic_beta)
        cache[T] = (el, el.aux_batch(problems))
        return (lambda th, aux, data: el.objective(th, lam_reg, aux, data)), el.data

    def aux_at(T):
        return cache[T][1]

    full = [tuple(q) + (0,) * (3 - len(q)) for q in problems]
    ids = list(range(1 + max(q[0] for q in full))) if data_ids is None else list(data_ids)
    X0, U0 = make_data(Ts[0])
    n, p = X0.shape[-1], U0.shape[-1]

    def th0(T):
        keys = jnp.stack([jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(init_seed + int(ids[s])), r), T)
                          for s, _, r in full])
        return init_batch(keys, n, m, p)

    problems = [(s, N) for s, N, _ in full]
    for T, th, info in horizon_path(loss_at, Ts, th0, aux_at=aux_at, **{k: opt[k] for k in OPT_KEYS if k in opt}):
        el, aux = cache[T]
        thj = tuple(jnp.asarray(x) for x in th)
        terms = jax.jit(jax.vmap(el.terms_direct, in_axes=(0, 0, None)))(thj, aux, el.data)
        info.update({k: np.asarray(v) for k, v in terms.items()})
        info["obj_direct"] = np.asarray(jax.jit(jax.vmap(lambda t, a, d: el.objective_direct(t, lam_reg, a, d),
                                                         in_axes=(0, 0, None)))(thj, aux, el.data))
        cache.pop(T)
        yield T, th, info


def score_models(ev, arm, th, info, meta, nl_ics=None, kinds=("lqr", "hinf")):
    """One row per model: meta + loss info + Evaluator.evaluate + nonlinear success of each design.
    nl_ics: ICs (k, n), or {tip: ICs} (control.standard_ics) -> '<kind>_nl_succ_<tip in cm>' per tip
    and '<kind>_nl_succ' at 10 cm."""
    E, D, L, H = th
    sets = None if nl_ics is None else (nl_ics if isinstance(nl_ics, dict) else {0.10: nl_ics})
    rows, gains = [], []
    for i in range(E.shape[0]):
        r = dict(meta[i])
        r.update({k: float(np.asarray(v)[i]) for k, v in info.items() if np.ndim(v) >= 1})
        res, F = ev.evaluate(E[i], D[i], L[i], H[i], kinds=kinds)
        r.update(res)
        if sets is not None and arm is not None:
            for kind in kinds:
                for tip, X0 in sets.items():
                    ok = F[kind] is not None and res[f"{kind}_stable"]
                    v = float(simulate(arm, F[kind], X0, t_end=12.0)[2].mean()) if ok else 0.0
                    r[f"{kind}_nl_succ_{round(100 * tip)}"] = v
                r[f"{kind}_nl_succ"] = r.get(f"{kind}_nl_succ_10", np.nan)
        rows.append(r)
        gains.append(F)
    return rows, gains


def mark_selected(df, keys, obj="obj"):
    """Column 'selected': True for the lowest-objective row of each group (ties: first)."""
    df = df.copy()
    df["selected"] = False
    o = df[obj].where(np.isfinite(df[obj]), np.inf)
    idx = o.groupby([df[k] for k in keys], dropna=False).idxmin()
    df.loc[idx.values, "selected"] = True
    return df
