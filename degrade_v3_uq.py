"""Pyomo transcription of the v3 damage model, and reconstruction of ``theta = f_0`` from it.

The v3 counterpart of :mod:`degrade_v2_uq`.  Same contract, same checking discipline: the data
comes from :func:`degrade_v3.simulate` (numpy), so the measurements and the model fitting them
stay two independent implementations, and :func:`check_forward` compares them at the true
solution before any reconstruction is allowed to mean anything.

What is gone relative to v2, and what it buys
---------------------------------------------
No ``u``, no ``eps_sq``, no ``c_elastic``, no ``c_eps``.  Those were 53% of the v2 Jacobian at
grid 20 and the whole of its KKT fill problem.  In their place, the compaction flux enters
``c_mass`` directly over the five-point stencil.

**The decayed field is carried as a variable, not inlined**, and that is the load-bearing
choice.  With ``ft`` a variable the mass balance is *bilinear* in ``(ft, dw)``, so its second
derivatives are constants; inlining ``ft = f exp(-a I - b I^2)`` would put an exponential at
every one of the five stencil positions and the block would stop being bilinear.  This is the
same reasoning v2 used to keep ``dw`` and ``Ipix`` as variables, applied one level further out.
``inline_decay=True`` builds the other version so the difference can be measured rather than
argued.

There is no structural off switch.  v2 had to drop its transport block whenever the flow was
identically zero, because ``sqrt(v^2 + eps^2)`` is a norm of the displacement and has no
derivative at rest.  v3's map is C-infinity everywhere -- no square root, no norm -- so ``I0 = 0``
and ``c_cp = 0`` are ordinary points and the special case is deleted rather than ported.  That is
V3 and V4.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyomo.environ as pyo

from degrade_v2_uq import measurement_rays, ray_walk, solve_with_fallback, _make_solver
from degrade_v3 import V3Params, simulate, _phantom, _disc, _demo_sequence
from degrade_v2 import scale_to_optical_depth


# --- model ------------------------------------------------------------------------------------

def _neighbours(q, res):
    """The four face neighbours of flat index ``q``, as ``(neighbour, )`` tuples that exist."""
    i, j = q // res, q % res
    out = []
    if j > 0:
        out.append(q - 1)
    if j < res - 1:
        out.append(q + 1)
    if i > 0:
        out.append(q - res)
    if i < res - 1:
        out.append(q + res)
    return out


def build_v3_model(theta_ref, seq, p: V3Params, image_res: int, *, f_bounds=None,
                   inline_decay: bool = False):
    """Steps 1-11 of the v3 model as a Pyomo model.  ``theta_ref`` seeds every variable."""
    theta_ref = np.asarray(theta_ref, dtype=float)
    res = int(image_res)
    npix = res * res
    meas = measurement_rays(seq, res)
    K = len(meas)
    if K == 0:
        raise ValueError("no measurements: the sequence is empty")

    m = pyo.ConcreteModel(name="degrade_v3")
    m.res, m.n_steps, m.meas, m.p = res, K, meas, p
    m.inline_decay = bool(inline_decay)

    m.PIX = pyo.RangeSet(0, npix - 1)
    m.T = pyo.RangeSet(0, K)
    m.TM = pyo.RangeSet(0, K - 1)

    flat = theta_ref.ravel()
    m.f = pyo.Var(m.PIX, m.T, bounds=f_bounds, initialize=lambda _m, q, k: float(flat[q]))
    m.Q = pyo.Var(m.PIX, m.T, initialize=0.0)
    for q in m.PIX:
        m.Q[q, 0].fix(0.0)

    # --- 1. photon balance: the cumulative optical depth, bidiagonal along each ray ----------
    chain, ray_id = [], []
    for k, (_ang, rays) in enumerate(meas):
        for j, (_r, walk) in enumerate(rays):
            ray_id.append((k, j, len(walk)))
            chain.extend((k, j, t) for t in range(len(walk) + 1))
    m.CH = pyo.Set(initialize=chain, dimen=3, ordered=True)
    m.S = pyo.Var(m.CH, initialize=0.0)

    def _chain(mm, k, j, t):
        if t == 0:
            return mm.S[k, j, 0] == 0.0
        _pix, chord, shield = meas[k][1][j][1][t - 1]
        return mm.S[k, j, t] == mm.S[k, j, t - 1] + chord * mm.f[shield, k]
    m.c_chain = pyo.Constraint(m.CH, rule=_chain)

    I_terms = {(q, k): [] for q in range(npix) for k in range(K)}
    dQ_terms = {(q, k): [] for q in range(npix) for k in range(K)}
    for k, (_ang, rays) in enumerate(meas):
        for j, (_r, walk) in enumerate(rays):
            for t, (pix, chord, _sh) in enumerate(walk):
                I_terms[(pix, k)].append((k, j, t))
                dQ_terms[(pix, k)].append(((k, j, t), chord))

    m.Ipix = pyo.Var(m.PIX, m.TM, initialize=0.0)

    def _ip(mm, q, k):
        terms = I_terms[(q, k)]
        if not terms:
            return mm.Ipix[q, k] == 0.0
        return mm.Ipix[q, k] == sum(p.I0 * pyo.exp(-mm.S[idx]) for idx in terms)
    m.c_Ipix = pyo.Constraint(m.PIX, m.TM, rule=_ip)

    # --- 2. dose accumulation ---------------------------------------------------------------
    def _dose(mm, q, k):
        terms = dQ_terms[(q, k)]
        if not terms:
            return mm.Q[q, k + 1] == mm.Q[q, k]
        inc = sum(p.c_q * p.I0 * pyo.exp(-mm.S[idx]) * chord for idx, chord in terms)
        return mm.Q[q, k + 1] == mm.Q[q, k] + inc
    m.c_dose = pyo.Constraint(m.PIX, m.TM, rule=_dose)

    # --- 3, 4, 5. response and the RELATIVE converted fraction ------------------------------
    def _om(expr):
        return p.omega_inf + (1.0 - p.omega_inf) * pyo.exp(-expr / p.Q_c)

    m.dw = pyo.Var(m.PIX, m.TM, initialize=0.0)

    def _dwc(mm, q, k):
        # s = exp(-dQ/Q_c) is the per-step survival factor; dw = (om - om_inf)(1 - s)/om.
        # Written multiplicatively, and at omega_inf = 0 there is no division at all.
        s = pyo.exp(-(mm.Q[q, k + 1] - mm.Q[q, k]) / p.Q_c)
        if p.omega_inf == 0.0:
            return mm.dw[q, k] == 1.0 - s
        om = _om(mm.Q[q, k])
        return mm.dw[q, k] * om == (om - p.omega_inf) * (1.0 - s)
    m.c_dw = pyo.Constraint(m.PIX, m.TM, rule=_dwc)

    # --- 9. eq:xd_decay, verbatim -- carried as a VARIABLE so the flux stays bilinear -------
    if inline_decay:
        def _ft(mm, q, k):
            return mm.f[q, k] * pyo.exp(-p.a * mm.Ipix[q, k] - p.b * mm.Ipix[q, k] ** 2)
    else:
        m.ft = pyo.Var(m.PIX, m.TM, initialize=lambda _m, q, k: float(flat[q]))

        def _ftc(mm, q, k):
            return mm.ft[q, k] == mm.f[q, k] * pyo.exp(
                -p.a * mm.Ipix[q, k] - p.b * mm.Ipix[q, k] ** 2)
        m.c_ft = pyo.Constraint(m.PIX, m.TM, rule=_ftc)

        def _ft(mm, q, k):
            return mm.ft[q, k]

    # --- 6, 7. nearest-neighbour compaction flux, then the mass balance ---------------------
    # F_{p->q} = c_cp * 0.5 * (ft_p + ft_q) * (dw_q - dw_p), zero on boundary faces.
    # Written directly into the mass row: no flux variables, one constraint per pixel.
    def _mass(mm, q, k):
        ftp = _ft(mm, q, k)
        rhs = ftp
        if p.c_cp != 0.0:
            for nb in _neighbours(q, res):
                rhs -= p.c_cp * 0.5 * (ftp + _ft(mm, nb, k)) * (mm.dw[nb, k] - mm.dw[q, k])
        return mm.f[q, k + 1] == rhs
    m.c_mass = pyo.Constraint(m.PIX, m.TM, rule=_mass)

    # --- 11. observation: the last link of the chain, no new variable -----------------------
    m.obs_index = [(k, j, n) for (k, j, n) in ray_id]
    return m


# --- pinning and the forward check ------------------------------------------------------------

def numpy_trajectory(theta, seq, p: V3Params, image_res: int):
    """Every variable of the model, taken off a :func:`degrade_v3.simulate` run."""
    res = int(image_res)
    npix = res * res
    meas = measurement_rays(seq, res)
    K = len(meas)
    _s, _Q, _infos, (f_hist, Q_hist) = simulate(theta, seq, p, res, record_trajectory=True)
    f = np.stack([h.ravel() for h in f_hist], axis=1)         # (npix, K+1)
    Q = np.stack([h.ravel() for h in Q_hist], axis=1)

    S, Ipix, dQ = {}, np.zeros((npix, K)), np.zeros((npix, K))
    for k, (_ang, rays) in enumerate(meas):
        for j, (_r, walk) in enumerate(rays):
            acc = 0.0
            S[(k, j, 0)] = 0.0
            for t, (pix, chord, shield) in enumerate(walk):
                loc = p.I0 * np.exp(-acc)
                Ipix[pix, k] += loc
                dQ[pix, k] += p.c_q * loc * chord
                acc += chord * f[shield, k]
                S[(k, j, t + 1)] = acc
    om = p.omega_inf + (1.0 - p.omega_inf) * np.exp(-Q[:, :K] / p.Q_c)
    s = np.exp(-dQ / p.Q_c)
    dw = (1.0 - s) if p.omega_inf == 0.0 else (om - p.omega_inf) * (1.0 - s) / om
    ft = f[:, :K] * np.exp(-p.a * Ipix - p.b * Ipix ** 2)
    return dict(f=f, Q=Q, S=S, Ipix=Ipix, dw=dw, ft=ft)


def pin_model(m, traj, *, fix=True):
    """Set every variable to the numpy trajectory, and optionally fix it there."""
    res, K = m.res, m.n_steps
    for q in m.PIX:
        for k in m.T:
            m.f[q, k].set_value(float(traj["f"][q, k]))
            m.Q[q, k].set_value(float(traj["Q"][q, k]))
            if fix:
                m.f[q, k].fix()
                m.Q[q, k].fix()
        for k in m.TM:
            m.Ipix[q, k].set_value(float(traj["Ipix"][q, k]))
            m.dw[q, k].set_value(float(traj["dw"][q, k]))
            if not m.inline_decay:
                m.ft[q, k].set_value(float(traj["ft"][q, k]))
            if fix:
                m.Ipix[q, k].fix()
                m.dw[q, k].fix()
                if not m.inline_decay:
                    m.ft[q, k].fix()
    for idx in m.CH:
        m.S[idx].set_value(float(traj["S"][idx]))
        if fix:
            m.S[idx].fix()


def max_residual(m):
    """Worst absolute constraint-body violation, and where it is."""
    worst, where = 0.0, ""
    for c in m.component_data_objects(pyo.Constraint, active=True):
        try:
            body = pyo.value(c.body)
            lo = pyo.value(c.lower) if c.lower is not None else body
            r = abs(body - lo)
        except Exception:
            continue
        if r > worst:
            worst, where = r, c.name
    return worst, where


def check_forward(image_res: int = 24, n_steps: int = 3, verbose: bool = True,
                  c_cp: float = 0.3, inline_decay: bool = False, **kw):
    """Does the Pyomo model reproduce :func:`degrade_v3.simulate`?  Residual only, no solver."""
    p = V3Params(c_cp=c_cp, **kw)
    seq = _demo_sequence(n_steps)
    theta = scale_to_optical_depth(_phantom(image_res), 1.1, image_res)
    traj = numpy_trajectory(theta, seq, p, image_res)
    m = build_v3_model(theta, seq, p, image_res, inline_decay=inline_decay)
    pin_model(m, traj)
    r, where = max_residual(m)
    if verbose:
        print("v3 Pyomo model vs numpy simulator: grid %d, %d steps, c_cp = %g%s"
              % (image_res, n_steps, c_cp, ", inlined decay" if inline_decay else ""))
        print("  max constraint residual  %.3e   (%s)" % (r, where or "-"))
        print("  %s" % ("PASS" if r < 1e-10 else "FAIL"))
    return r


def model_size(m):
    """Variable and constraint counts per block, plus Jacobian nnz/row -- the V10 numbers."""
    from pyomo.core.expr.visitor import identify_variables
    rows = []
    for c in m.component_objects(pyo.Constraint, active=True):
        n = len(c)
        nz = sum(len({id(v) for v in identify_variables(cd.body, include_fixed=False)})
                 for cd in c.values())
        rows.append((c.name, n, nz, nz / max(n, 1)))
    nv = sum(1 for _ in m.component_data_objects(pyo.Var))
    nf = sum(1 for v in m.component_data_objects(pyo.Var) if not v.fixed)
    return dict(rows=rows, n_vars=nv, n_free=nf,
                n_cons=sum(r[1] for r in rows), jac_nnz=sum(r[2] for r in rows))


# --- estimation ---------------------------------------------------------------------------

def _tv_expression(m, theta_scale: float):
    """Smoothed isotropic TV of ``f[:, 0]``, smoothing tied to the field as in v2."""
    res = m.res
    eps = (1e-2 * theta_scale) ** 2
    tv = 0.0
    for i in range(res):
        for j in range(res):
            q = i * res + j
            d0 = (m.f[q + res, 0] - m.f[q, 0]) if i < res - 1 else 0.0
            d1 = (m.f[q + 1, 0] - m.f[q, 0]) if j < res - 1 else 0.0
            tv += pyo.sqrt(d0 ** 2 + d1 ** 2 + eps)
    return tv


def add_estimation_objective(m, y_data, tv_weight: float, theta_scale: float):
    """Fit the ray integrals, regularised by TV on ``theta``.  Both terms normalised to O(1)."""
    m.YD = pyo.Set(initialize=[(k, j) for (k, j, _n) in m.obs_index], dimen=2, ordered=True)
    m.y_data = pyo.Var(m.YD, initialize=0.0)
    for (k, j, n) in m.obs_index:
        m.y_data[k, j].set_value(float(y_data[k][j]))
        m.y_data[k, j].fix()
    nobs = max(len(m.obs_index), 1)
    yscale = max(float(np.mean([abs(y_data[k][j]) for (k, j, _n) in m.obs_index])), 1e-12)
    fit = sum((m.S[k, j, n] - m.y_data[k, j]) ** 2 for (k, j, n) in m.obs_index) / (nobs * yscale ** 2)
    npix = m.res * m.res
    m.obj = pyo.Objective(expr=fit + tv_weight * _tv_expression(m, theta_scale) / (npix * theta_scale))
    return m.obj


def run_v3_reconstruction(theta, seq, p: V3Params, image_res: int, *, tv_weight=0.001,
                          linear_solver="ma97", solver_opts=None, inline_decay=False,
                          max_iter=3000, log_callback=None, gate=True):
    """Estimate ``theta`` from v3 dynamics.  Returns a dict of results and timings (V10)."""
    import io, re
    res = int(image_res)
    scale = float(np.abs(theta).max())
    t_sim = time.time()
    _s, _Q, infos, y = simulate(theta, seq, p, res, record_observations=True)
    t_sim = time.time() - t_sim

    gate_resid = float("nan")
    if gate:
        traj = numpy_trajectory(theta, seq, p, res)
        mg = build_v3_model(theta, seq, p, res, inline_decay=inline_decay)
        pin_model(mg, traj)
        gate_resid, where = max_residual(mg)
        del mg
        if gate_resid > 1e-8:
            raise RuntimeError("v3 Pyomo model no longer reproduces degrade_v3.simulate "
                               "(residual %.3e at %s)" % (gate_resid, where))

    t_build = time.time()
    m = build_v3_model(theta, seq, p, res, f_bounds=(0.0, 1.5 * scale), inline_decay=inline_decay)
    # start from a flat field: only theta is unknown, the dynamics are seeded by the simulator
    t0 = numpy_trajectory(np.full_like(theta, float(theta.mean())), seq, p, res)
    pin_model(m, t0, fix=False)
    for q in m.PIX:
        m.Q[q, 0].fix(0.0)
    add_estimation_objective(m, y, tv_weight, scale)
    t_build = time.time() - t_build

    nv = sum(1 for _ in m.component_data_objects(pyo.Var))
    nc = sum(1 for _ in m.component_data_objects(pyo.Constraint, active=True))
    opts = dict(solver_opts or {})
    opts["print_timing_statistics"] = "yes"
    buf = io.StringIO()
    cb = log_callback or buf.write
    t_solve = time.time()
    try:
        r, ls = solve_with_fallback(m, linear_solver=linear_solver, max_iter=max_iter,
                                    log_callback=lambda s: (buf.write(s), cb(s))[0], options=opts)
        status = str(r.solver.termination_condition)
    except Exception as e:
        status, ls = "FAILED: " + str(e).splitlines()[0][:90], linear_solver
    t_solve = time.time() - t_solve

    log = buf.getvalue()
    g = lambda pat: (re.findall(pat, log) or ["-"])[-1]
    th = np.array([pyo.value(m.f[q, 0]) for q in m.PIX]).reshape(res, res)
    return dict(
        theta_hat=th, status=status, linear_solver=ls,
        t_sim=t_sim, t_build=t_build, t_solve=t_solve,
        n_vars=nv, n_cons=nc,
        gate_residual=gate_resid,
        iters=g(r"Number of Iterations\.*:\s*(\S+)"),
        nnz_hess=g(r"Number of nonzeros in Lagrangian Hessian\.*:\s*(\S+)"),
        nnz_jac=g(r"Number of nonzeros in equality constraint Jacobian\.*:\s*(\S+)"),
        ipopt_s=g(r"Total seconds in IPOPT \(w/o function evaluations\)\s*=\s*(\S+)"),
        fev_s=g(r"Total seconds in NLP function evaluations\s*=\s*(\S+)"),
        theta_err=100.0 * float(np.sqrt(np.mean((th - theta) ** 2))) / float(theta.max()),
        courant=max(i.compaction for i in infos),
        mass=float(_s.sum()) / float(theta.sum()),
    )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image-res", type=int, default=24)
    ap.add_argument("--n-steps", type=int, default=4)
    ap.add_argument("--c-cp", type=float, default=0.3)
    ap.add_argument("--tv-weight", type=float, default=0.001)
    ap.add_argument("--linear-solver", default="ma97")
    ap.add_argument("--ma97-order", default="metis")
    ap.add_argument("--inline-decay", action="store_true")
    ap.add_argument("--check", action="store_true", help="forward residual check only")
    a = ap.parse_args()
    if a.check:
        sys.exit(0 if check_forward(a.image_res, a.n_steps, c_cp=a.c_cp,
                                    inline_decay=a.inline_decay) < 1e-10 else 1)
    p = V3Params(c_cp=a.c_cp)
    th = scale_to_optical_depth(_phantom(a.image_res), 1.1, a.image_res)
    opts = {"ma97_order": a.ma97_order} if a.linear_solver == "ma97" else {}
    r = run_v3_reconstruction(th, _demo_sequence(a.n_steps), p, a.image_res,
                              tv_weight=a.tv_weight, linear_solver=a.linear_solver,
                              solver_opts=opts, inline_decay=a.inline_decay)
    print("\nv3 %dx%d, K=%d, %s%s" % (a.image_res, a.image_res, a.n_steps, a.linear_solver,
                                      " (inlined decay)" if a.inline_decay else ""))
    print("  gate (model vs simulator)  %.2e" % r["gate_residual"])
    print("  size                       %d vars / %d cons" % (r["n_vars"], r["n_cons"]))
    print("  nnz Jacobian / Hessian     %s / %s" % (r["nnz_jac"], r["nnz_hess"]))
    print("  iterations                 %s" % r["iters"])
    print("  wall: sim %.1fs  build %.1fs  solve %.1fs  TOTAL %.1fs"
          % (r["t_sim"], r["t_build"], r["t_solve"], r["t_sim"] + r["t_build"] + r["t_solve"]))
    print("  IPOPT(w/o fev) / fev       %s s / %s s" % (r["ipopt_s"], r["fev_s"]))
    print("  theta RMS error            %.2f %% of peak" % r["theta_err"])
    print("  status                     %s (%s)" % (r["status"], r["linear_solver"]))
