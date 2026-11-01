"""Optimizers for the linear latent models: Adam then guarded L-BFGS, vmapped over problems.

The losses are stiff (the T-th terms scale like rho(A)^2T; curvature along the unstable
latent eigenvalue like T^2 rho(A)^2T), so first-order methods stall before a stationary
point once T >~ 16: every solve is Adam (exploration) followed by L-BFGS (polish).

make_solver(f, ...) -> solve(th, *args) for one problem; th = (E, D, L, H).
    hold_L = K      L is held at its initial value for the first K Adam steps (its update
                    is masked; E, D, H train).  The initialization of Sec. 6.4 with L_0 = rho-hat I.
    record_every    store (E, L, objective) every that many iterations (Adam and L-BFGS),
                    for training traces; L-BFGS then runs a fixed number of iterations.
                    Otherwise L-BFGS stops after 20 consecutive iterations with relative
                    decrease < 1e-14 (under vmap, when every problem has stopped).
    A step that produces non-finite values is rejected.

horizon_path(...)   the protocol of the population/finite-sample sweeps over T: at each T,
                    solve from the previous T's solution (L-BFGS) and from a random init
                    (Adam + L-BFGS), keep the lower objective, continue from it.  With
                    ``th0`` a function of T the random init is a fresh draw at every T
                    (as in every experiment); a fixed ``th0`` re-uses one draw.
                    Cold starts alone do not converge at large T; continuation alone is
                    hysteretic (it stays in a stable-mode basin long after the basin that
                    captures the unstable mode has become far better).
"""

from __future__ import annotations

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax


def _finite(tree):
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(tree)]))


def make_solver(f, n_adam=3000, n_lbfgs=2000, lr=1e-2, lr_end=1e-5, clip=None, hold_L=0,
                record_every=0):
    """solve(th, *args) -> (th, info[, frames]) minimizing f(th, *args) from th."""
    tx = ([optax.clip_by_global_norm(clip)] if clip else [])
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, max(1, n_adam // 50), max(n_adam, 2), lr_end)
    adam = optax.apply_if_finite(optax.chain(*tx, optax.adam(sched)), 20)
    lb = optax.lbfgs(memory_size=20)
    rec = int(record_every)

    def solve(th, *args):
        g = lambda t: f(t, *args)
        snap = lambda t: (t[0], t[2], g(t))
        frames = []

        # ---- Adam -------------------------------------------------------------------
        def adam_step(c, i):
            th, st = c
            gr = jax.grad(g)(th)
            up, st = adam.update(gr, st, th)
            if hold_L:
                live = (i >= hold_L).astype(th[2].dtype)
                up = (up[0], up[1], up[2] * live, up[3])
            return (optax.apply_updates(th, up), st), None

        if n_adam > 0:
            st = adam.init(th)
            if rec:
                def chunk(c, i0):
                    c, _ = jax.lax.scan(adam_step, c, i0 + jnp.arange(rec))
                    return c, snap(c[0])
                (th, st), fr = jax.lax.scan(chunk, (th, st), jnp.arange(0, n_adam, rec))
                frames.append(fr)
            else:
                (th, st), _ = jax.lax.scan(adam_step, (th, st), jnp.arange(n_adam))

        # ---- L-BFGS -----------------------------------------------------------------
        vg = optax.value_and_grad_from_state(g)
        stl = lb.init(th)

        def lb_step(th, st):
            v, gr = vg(th, state=st)
            up, st2 = lb.update(gr, st, th, value=v, grad=gr, value_fn=g)
            th2 = optax.apply_updates(th, up)
            ok = _finite((th2, st2))
            th = jax.tree_util.tree_map(lambda a, b: jnp.where(ok, a, b), th2, th)
            st = jax.tree_util.tree_map(lambda a, b: jnp.where(ok, a, b), st2, st)
            return th, st, v

        if n_lbfgs > 0 and rec:
            def chunk(c, _):
                def one(c, _):
                    th, st = c
                    th, st, _ = lb_step(th, st)
                    return (th, st), None
                c, _ = jax.lax.scan(one, c, None, length=rec)
                return c, snap(c[0])
            (th, stl), fr = jax.lax.scan(chunk, (th, stl), None, length=max(1, n_lbfgs // rec))
            frames.append(fr)
            n_it = n_lbfgs
        elif n_lbfgs > 0:
            def cond(c):
                i, _, _, _, stall = c
                return (i < n_lbfgs) & (stall < 20)

            def body(c):
                i, th, st, fprev, stall = c
                th, st, v = lb_step(th, st)
                stall = jnp.where((fprev - v) > 1e-14 * jnp.abs(v), 0, stall + 1)
                return i + 1, th, st, v, stall

            f0 = g(th)
            n_it, th, stl, _, _ = jax.lax.while_loop(cond, body, (0, th, stl, f0 + jnp.abs(f0) + 1.0, 0))
        else:
            n_it = 0
        v, gr = jax.value_and_grad(g)(th)
        gnorm = jnp.sqrt(sum(jnp.sum(x * x) for x in jax.tree_util.tree_leaves(gr)))
        info = dict(obj=v, gnorm=gnorm, lbfgs_iters=jnp.asarray(n_it))
        if rec:
            fr = jax.tree_util.tree_map(lambda *a: jnp.concatenate(a, 0), *frames) if frames else None
            return th, info, fr
        return th, info

    return solve


def batch_solver(f, in_axes=None, **kw):
    """jit(vmap(solve)) over a leading problem axis of th (and of args per in_axes)."""
    solve = make_solver(f, **kw)
    return jax.jit(jax.vmap(solve) if in_axes is None else jax.vmap(solve, in_axes=in_axes))


def pick(choose, a, b):
    """Per-problem selection between two pytrees with a leading problem axis."""
    def sel(x, y):
        return jnp.where(choose.reshape(choose.shape + (1,) * (x.ndim - choose.ndim)), x, y)
    return jax.tree_util.tree_map(sel, a, b)


def horizon_path(loss_at, Ts, th0, aux_at=None, protocol="restart", n_adam=3000, n_lbfgs=2000,
                 n_warm=2000, lr=1e-2, lr_end=1e-5):
    """Generator over T of (T, th, info): th with a leading problem axis.

    th0        initializations with a leading problem axis, or th0(T) -> those of horizon T
               (fresh cold starts at every horizon);
    loss_at(T) -> f(th_raw, aux) for one problem (lam etc. bound by the caller), or a pair
                  (f(th_raw, aux, shared), shared) with data shared by all problems (not vmapped);
    aux_at(T)  -> per-problem pytree with a leading problem axis (or None)."""
    init_at = th0 if callable(th0) else (lambda T: th0)
    prev = None
    for T in Ts:
        th0 = init_at(T)
        f = loss_at(T)
        f, shared = f if isinstance(f, tuple) else ((lambda t, a, s, f=f: f(t, a)), None)
        aux = aux_at(T) if aux_at is not None else jnp.zeros(jax.tree_util.tree_leaves(th0)[0].shape[0])
        solver = lambda **kw: batch_solver(f, in_axes=(0, 0, None), lr=lr, lr_end=lr_end, **kw)
        cold = protocol in ("cold", "restart") or prev is None
        warm = protocol in ("warm", "restart") and prev is not None
        if cold:
            thc, ic = solver(n_adam=n_adam, n_lbfgs=n_lbfgs)(th0, aux, shared)
        if warm:
            thw, iw = solver(n_adam=0, n_lbfgs=n_warm)(prev, aux, shared)
        if cold and warm:
            ok_w = jnp.isfinite(iw["obj"]) & (~jnp.isfinite(ic["obj"]) | (iw["obj"] <= ic["obj"]))
            th = pick(ok_w, thw, thc)
            info = {k: jnp.where(ok_w, iw[k], ic[k]) for k in iw}
            info.update(obj_warm=iw["obj"], obj_cold=ic["obj"], from_warm=ok_w)
        elif warm:
            th, info = thw, dict(iw, from_warm=jnp.ones_like(iw["obj"], dtype=bool))
        else:
            th, info = thc, dict(ic, from_warm=jnp.zeros_like(ic["obj"], dtype=bool), obj_cold=ic["obj"],
                                 obj_warm=jnp.full_like(ic["obj"], jnp.nan))
        prev = th
        yield T, jax.tree_util.tree_map(np.asarray, th), {k: np.asarray(v) for k, v in info.items()}


# -----------------------------------------------------------------------------------------
# Initializations
# -----------------------------------------------------------------------------------------
def init_params(key, n, m, p, L_init="orth", rho=1.0):
    """E, D ~ N(0, 1/n), H ~ 0.1 N(0, 1) and
       L_init = 'orth'  QR of a Gaussian, no sign correction (the arm script's init; not Haar:
                        for m = 3 always a rotation by more than 90 degrees)
                'eye'   rho I."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    E = jax.random.normal(k1, (m, n)) / np.sqrt(n)
    D = jax.random.normal(k2, (n, m)) / np.sqrt(n)
    H = 0.1 * jax.random.normal(k4, (m, p))
    if L_init == "orth":
        L = rho * jnp.linalg.qr(jax.random.normal(k3, (m, m)))[0]
    elif L_init == "eye":
        L = rho * jnp.eye(m)
    else:
        raise ValueError(L_init)
    return (E, D, L, H)


def init_batch(keys, n, m, p, L_init="orth", rho=1.0):
    return jax.vmap(lambda k: init_params(k, n, m, p, L_init, rho))(keys)

