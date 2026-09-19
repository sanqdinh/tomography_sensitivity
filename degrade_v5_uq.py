"""Pyomo transcription of v5 -- the reduced model with a NONLOCAL compaction potential.

Same checking discipline as v4: data comes from :func:`degrade_v5.simulate` (numpy), so the
measurements and the model fitting them stay two independent implementations, and
:func:`check_forward` compares them at the true solution before any reconstruction means anything.

What is new relative to v4
--------------------------
One block.  The flux driver is no longer the pointwise ``Pi`` but a potential ``phi`` satisfying

    [varsigma + gamma(1 - sigma_p)] phi_p  -  sum_{q~p} sigma_pq (phi_q - phi_p)  =  Pi_p

with ``sigma_p = 1 - exp(-ft_p/f_ref)`` and ``sigma_pq`` its face average.  **In the NLP this is
not an inner solve.**  ``phi`` is a variable and the row above is a constraint, so the elliptic
problem is handled by the same KKT factorisation as everything else -- there is no nested solver
and the Jacobian stays sparse.  That is the whole reason this is affordable here when it was not
in the forward model's terms.

``Pi``, ``sigma`` and ``phi`` are all carried as VARIABLES rather than inlined.  ``Pi`` and
``sigma`` are what keep the potential row bilinear-with-a-single-exp instead of dragging four
nested nonlinearities into every one of its ~6 entries, and ``phi`` is what keeps the flux row
the same shape it had in v4.

Blocks per stage: ``S`` (chain), ``Ipix``, ``dw``, ``ft``, ``sigma``, ``Pi``, ``phi``, ``f``,
plus ``yobs`` per ray.  Two more scalar fields than v4.
"""

from __future__ import annotations

import io
import re
import sys
import time
from typing import Optional

import numpy as np
import pyomo.environ as pyo

from degrade_v2_uq import measurement_rays, solve_with_fallback
from degrade_v3_uq import _neighbours
from degrade_v5 import V5Params, simulate, resolve, _phantom, _demo_sequence
from degrade_v2 import scale_to_optical_depth


def build_v5_model(theta_ref, seq, p: V5Params, image_res: int, *, f_bounds=None,
                   inline_Ipix: bool = False, inline_dw: bool = False,
                   inline_ft: bool = False):
    """Steps 1-7 of the reduced model as a Pyomo model."""
    theta_ref = np.asarray(theta_ref, dtype=float)
    p = resolve(p, theta_ref)
    res = int(image_res)
    npix = res * res
    meas = measurement_rays(seq, res)
    K = len(meas)
    if K == 0:
        raise ValueError("no measurements: the sequence is empty")

    m = pyo.ConcreteModel(name="degrade_v5")
    m.res, m.n_steps, m.meas, m.p = res, K, meas, p
    m.inline_Ipix, m.inline_dw, m.inline_ft = bool(inline_Ipix), bool(inline_dw), bool(inline_ft)

    m.PIX = pyo.RangeSet(0, npix - 1)
    m.T = pyo.RangeSet(0, K)
    m.TM = pyo.RangeSet(0, K - 1)

    flat = theta_ref.ravel()
    m.f = pyo.Var(m.PIX, m.T, bounds=f_bounds, initialize=lambda _m, q, k: float(flat[q]))

    # --- 1. photon balance: cumulative optical depth, bidiagonal along each ray -------------
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

    # crossing lists: I_terms carries no chord (the fluence sum), dQ_terms carries it (the
    # fluence-path sum that drives dw). Same split v3 used, minus the c_q factor.
    I_terms = {(q, k): [] for q in range(npix) for k in range(K)}
    Id_terms = {(q, k): [] for q in range(npix) for k in range(K)}
    for k, (_ang, rays) in enumerate(meas):
        for j, (_r, walk) in enumerate(rays):
            for t, (pix, chord, _sh) in enumerate(walk):
                I_terms[(pix, k)].append((k, j, t))
                Id_terms[(pix, k)].append(((k, j, t), chord))

    if inline_Ipix:
        def _Ipix(mm, q, k):
            ts = I_terms[(q, k)]
            return sum(p.I0 * pyo.exp(-mm.S[i]) for i in ts) if ts else 0.0
    else:
        m.Ipix = pyo.Var(m.PIX, m.TM, initialize=0.0)

        def _ip(mm, q, k):
            ts = I_terms[(q, k)]
            if not ts:
                return mm.Ipix[q, k] == 0.0
            return mm.Ipix[q, k] == sum(p.I0 * pyo.exp(-mm.S[i]) for i in ts)
        m.c_Ipix = pyo.Constraint(m.PIX, m.TM, rule=_ip)

        def _Ipix(mm, q, k):
            return mm.Ipix[q, k]

    # --- 2. converted fraction, ONE row off the chain. No dose state. ----------------------
    def _dw_expr(mm, q, k):
        ts = Id_terms[(q, k)]
        if not ts:
            return 0.0
        return 1.0 - pyo.exp(-sum(p.c * p.I0 * pyo.exp(-mm.S[i]) * ch for i, ch in ts))

    if inline_dw:
        def _dwv(mm, q, k):
            return _dw_expr(mm, q, k)
    else:
        m.dw = pyo.Var(m.PIX, m.TM, initialize=0.0)

        def _dwc(mm, q, k):
            return mm.dw[q, k] == _dw_expr(mm, q, k)
        m.c_dw = pyo.Constraint(m.PIX, m.TM, rule=_dwc)

        def _dwv(mm, q, k):
            return mm.dw[q, k]

    # --- 3. mass loss, carried as a variable so the flux stays bilinear in (ft, Pi) --------
    if inline_ft:
        def _ft(mm, q, k):
            I = _Ipix(mm, q, k)
            return mm.f[q, k] * pyo.exp(-p.a * I - p.b * I ** 2)
    else:
        m.ft = pyo.Var(m.PIX, m.TM, initialize=lambda _m, q, k: float(flat[q]))

        def _ftc(mm, q, k):
            I = _Ipix(mm, q, k)
            return mm.ft[q, k] == mm.f[q, k] * pyo.exp(-p.a * I - p.b * I ** 2)
        m.c_ft = pyo.Constraint(m.PIX, m.TM, rule=_ftc)

        def _ft(mm, q, k):
            return mm.ft[q, k]

    # --- 4a. the compaction potential: Pi, sigma, then the elliptic row -------------------
    fm = float(p.f_max)
    eh = float(p.eps_h) * max(float(np.abs(theta_ref).max()), 1e-300)
    vsig = float(p.varsigma())
    gam = float(p.gamma)
    f_ref = (None if p.f_ref_frac is None else float(p.f_ref_frac) * fm)

    m.Pi = pyo.Var(m.PIX, m.TM, initialize=0.0)

    def _pic(mm, q, k):
        return mm.Pi[q, k] * fm == _dwv(mm, q, k) * _ft(mm, q, k)
    m.c_Pi = pyo.Constraint(m.PIX, m.TM, rule=_pic)

    # sigma < 1 is structural (1 - exp(-x) < 1); the lower bound is left open because ft can be
    # marginally negative from the logistic residual and a hard 0 would make pinning the gate
    # trajectory an out-of-bounds write.
    m.sig = pyo.Var(m.PIX, m.TM, bounds=(None, 1.0), initialize=0.0)

    def _sigc(mm, q, k):
        if f_ref is None:
            return mm.sig[q, k] * fm == _ft(mm, q, k)
        return mm.sig[q, k] == 1.0 - pyo.exp(-_ft(mm, q, k) / f_ref)
    m.c_sig = pyo.Constraint(m.PIX, m.TM, rule=_sigc)

    m.phi = pyo.Var(m.PIX, m.TM, initialize=0.0)

    def _phic(mm, q, k):
        # [varsigma + gamma(1-sigma_p)] phi_p - sum_q sigma_pq (phi_q - phi_p) = Pi_p.
        # Written with sigma_pq = (sigma_p + sigma_q)/2 inline: bilinear in (sigma, phi), so the
        # second derivatives are constants and the row has ~6 variables on a 5-point stencil.
        acc = (vsig + gam * (1.0 - mm.sig[q, k])) * mm.phi[q, k]
        for nb in _neighbours(q, res):
            acc -= 0.5 * (mm.sig[q, k] + mm.sig[nb, k]) * (mm.phi[nb, k] - mm.phi[q, k])
        return acc == mm.Pi[q, k]
    m.c_phi = pyo.Constraint(m.PIX, m.TM, rule=_phic)

    # --- 4b, 5. the flux, driven by phi, and the mass balance. Same shape as v4. ----------
    def _mass(mm, q, k):
        rhs = _ft(mm, q, k)
        if p.c_cp != 0.0:
            ftp = _ft(mm, q, k)
            for nb in _neighbours(q, res):
                g = mm.phi[nb, k] - mm.phi[q, k]
                ftq = _ft(mm, nb, k)
                if p.flux == "harmonic":
                    rhs -= p.c_cp * (2.0 * ftp * ftq / (ftp + ftq + eh)) * g
                elif p.flux == "upwind":
                    chi = 1.0 / (1.0 + pyo.exp(-p.beta * g))
                    rhs -= p.c_cp * (chi * ftp + (1.0 - chi) * ftq) * g
                else:
                    rhs -= p.c_cp * 0.5 * (ftp + ftq) * g
        return mm.f[q, k + 1] == rhs
    m.c_mass = pyo.Constraint(m.PIX, m.TM, rule=_mass)

    # --- 6. observation, index-shifted: y_{k+1} = C_{u_k} f_{k+1} --------------------------
    m.obs_index = [(k, j, n) for (k, j, n) in ray_id]
    m.RAY = pyo.Set(initialize=[(k, j) for (k, j, _n) in ray_id], dimen=2, ordered=True)
    m.yobs = pyo.Var(m.RAY, initialize=0.0)
    chords = {}
    for k, (_ang, rays) in enumerate(meas):
        for j, (_r, walk) in enumerate(rays):
            acc = {}
            for _pix, chord, owner in walk:
                acc[owner] = acc.get(owner, 0.0) + chord
            chords[(k, j)] = sorted(acc.items())
    m.chords = chords

    def _obs(mm, k, j):
        return mm.yobs[k, j] == sum(c * mm.f[q, k + 1] for q, c in chords[(k, j)])
    m.c_obs = pyo.Constraint(m.RAY, rule=_obs)
    return m


# --- pinning and the gate ---------------------------------------------------------------

def numpy_trajectory(theta, seq, p: V5Params, image_res: int):
    """Every variable of the model, taken off a :func:`degrade_v5.simulate` run."""
    from degrade_v2 import accumulate_dose
    from dose_response import bundle_r_values
    p = resolve(p, theta)
    res = int(image_res)
    npix = res * res
    meas = measurement_rays(seq, res)
    K = len(meas)
    _f, _infos, hist = simulate(theta, seq, p, res, record_trajectory=True)
    f = np.stack([h.ravel() for h in hist], axis=1)

    S, Ipix, cIdelta = {}, np.zeros((npix, K)), np.zeros((npix, K))
    for k, (_ang, rays) in enumerate(meas):
        for j, (_r, walk) in enumerate(rays):
            acc = 0.0
            S[(k, j, 0)] = 0.0
            for t, (pix, chord, shield) in enumerate(walk):
                loc = p.I0 * np.exp(-acc)
                Ipix[pix, k] += loc
                cIdelta[pix, k] += p.c * loc * chord
                acc += chord * f[shield, k]
                S[(k, j, t + 1)] = acc
    dw = 1.0 - np.exp(-cIdelta)
    ft = f[:, :K] * np.exp(-p.a * Ipix - p.b * Ipix ** 2)
    from degrade_v5 import material_indicator, potential_operator
    import scipy.sparse.linalg as _spla
    fmv = float(p.f_max)
    Pi = dw * ft / fmv
    sig = np.zeros_like(ft)
    phi = np.zeros_like(ft)
    for k in range(K):
        s_k = material_indicator(ft[:, k].reshape(res, res), fmv, p.f_ref_frac)
        sig[:, k] = s_k.ravel()
        A = potential_operator(s_k, p.varsigma(), p.gamma)
        phi[:, k] = _spla.spsolve(A, Pi[:, k])
    yobs = {}
    for k, (_ang, rays) in enumerate(meas):
        for j, (_r, walk) in enumerate(rays):
            acc = {}
            for _pix, chord, owner in walk:
                acc[owner] = acc.get(owner, 0.0) + chord
            yobs[(k, j)] = float(sum(c * f[q, k + 1] for q, c in acc.items()))
    return dict(f=f, S=S, Ipix=Ipix, dw=dw, ft=ft, yobs=yobs, Pi=Pi, sig=sig, phi=phi)


def pin_model(m, traj, *, fix=True):
    for q in m.PIX:
        for k in m.T:
            m.f[q, k].set_value(float(traj["f"][q, k]))
            if fix:
                m.f[q, k].fix()
        for k in m.TM:
            if not m.inline_Ipix:
                m.Ipix[q, k].set_value(float(traj["Ipix"][q, k]))
            if not m.inline_dw:
                m.dw[q, k].set_value(float(traj["dw"][q, k]))
            if not m.inline_ft:
                m.ft[q, k].set_value(float(traj["ft"][q, k]))
            m.Pi[q, k].set_value(float(traj["Pi"][q, k]))
            m.sig[q, k].set_value(float(traj["sig"][q, k]))
            m.phi[q, k].set_value(float(traj["phi"][q, k]))
            if fix:
                if not m.inline_Ipix:
                    m.Ipix[q, k].fix()
                if not m.inline_dw:
                    m.dw[q, k].fix()
                if not m.inline_ft:
                    m.ft[q, k].fix()
                m.Pi[q, k].fix()
                m.sig[q, k].fix()
                m.phi[q, k].fix()
    for idx in m.CH:
        m.S[idx].set_value(float(traj["S"][idx]))
        if fix:
            m.S[idx].fix()
    for idx in m.RAY:
        m.yobs[idx].set_value(float(traj["yobs"][idx]))
        if fix:
            m.yobs[idx].fix()


def max_residual(m):
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


def check_forward(image_res: int = 24, n_steps: int = 3, verbose: bool = True, **kw):
    """G1: does the Pyomo model reproduce degrade_v5.simulate? Residual only, no solver."""
    inline = {k: kw.pop(k) for k in ("inline_Ipix", "inline_dw", "inline_ft") if k in kw}
    p = V5Params(**kw)
    seq = _demo_sequence(n_steps)
    theta = scale_to_optical_depth(_phantom(image_res), 1.1, image_res)
    traj = numpy_trajectory(theta, seq, p, image_res)
    m = build_v5_model(theta, seq, p, image_res, **inline)
    pin_model(m, traj)
    r, where = max_residual(m)
    if verbose:
        tags = ",".join(k.replace("inline_", "") for k, v in inline.items() if v) or "none"
        print("  v5 Pyomo vs numpy: grid %d, %d steps, c_cp=%g, inlined %s -> %.3e (%s)  %s"
              % (image_res, n_steps, p.c_cp, tags, r, where or "-",
                 "PASS" if r < 1e-10 else "FAIL"))
    return r


# --- estimation ---------------------------------------------------------------------------

def _tv_expression(m, theta_scale: float):
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
    m.YD = pyo.Set(initialize=[(k, j) for (k, j, _n) in m.obs_index], dimen=2, ordered=True)
    m.y_data = pyo.Var(m.YD, initialize=0.0)
    for (k, j, n) in m.obs_index:
        m.y_data[k, j].set_value(float(y_data[k][j]))
        m.y_data[k, j].fix()
    nobs = max(len(m.obs_index), 1)
    yscale = max(float(np.mean([abs(y_data[k][j]) for (k, j, _n) in m.obs_index])), 1e-12)
    fit = sum((m.yobs[k, j] - m.y_data[k, j]) ** 2
              for (k, j, _n) in m.obs_index) / (nobs * yscale ** 2)
    npix = m.res * m.res
    m.obj = pyo.Objective(expr=fit
                          + tv_weight * _tv_expression(m, theta_scale) / (npix * theta_scale))
    return m.obj


def run_v5_reconstruction(theta, seq, p: V5Params, image_res: int, *, tv_weight=0.001,
                          linear_solver="ma97", solver_opts=None, max_iter=3000,
                          gate=True, continuation=True, **inline):
    """Estimate theta from the reduced dynamics. Returns a dict of results and timings."""
    res = int(image_res)
    p = resolve(p, theta)
    scale = float(np.abs(theta).max())
    t_sim = time.time()
    _f, infos, y = simulate(theta, seq, p, res, record_observations=True)
    t_sim = time.time() - t_sim

    gate_resid = float("nan")
    if gate:
        traj = numpy_trajectory(theta, seq, p, res)
        mg = build_v5_model(theta, seq, p, res, **inline)
        pin_model(mg, traj)
        gate_resid, where = max_residual(mg)
        del mg
        if gate_resid > 1e-8:
            raise RuntimeError("v5 Pyomo model no longer reproduces degrade_v5.simulate "
                               "(residual %.3e at %s)" % (gate_resid, where))

    theta0 = np.full_like(theta, float(theta.mean()))
    t_cont, cont_iters, cont_status = 0.0, "-", "skipped"
    if continuation:
        tc = time.time()
        from dataclasses import replace
        p0 = replace(p, I0=0.0, c_cp=0.0)
        m0 = build_v5_model(theta, seq, p0, res, f_bounds=(0.0, 1.5 * scale), **inline)
        for q in m0.PIX:
            m0.f[q, 0].set_value(float(theta0.ravel()[q]))
        add_estimation_objective(m0, y, tv_weight, scale)
        b0 = io.StringIO()
        try:
            r0, _ls = solve_with_fallback(m0, linear_solver=linear_solver, max_iter=max_iter,
                                          log_callback=b0.write, options=dict(solver_opts or {}))
            cont_status = str(r0.solver.termination_condition)
            theta0 = np.array([pyo.value(m0.f[q, 0]) for q in m0.PIX]).reshape(res, res)
        except Exception as e:
            cont_status = "FAILED: " + str(e).splitlines()[0][:60]
        cont_iters = (re.findall(r"Number of Iterations\.*:\s*(\S+)", b0.getvalue()) or ["-"])[-1]
        del m0
        t_cont = time.time() - tc

    t_build = time.time()
    m = build_v5_model(theta, seq, p, res, f_bounds=(0.0, 1.5 * scale), **inline)
    t0 = numpy_trajectory(theta0, seq, p, res)
    pin_model(m, t0, fix=False)
    add_estimation_objective(m, y, tv_weight, scale)
    t_build = time.time() - t_build

    nv = sum(1 for _ in m.component_data_objects(pyo.Var))
    nc = sum(1 for _ in m.component_data_objects(pyo.Constraint, active=True))
    opts = dict(solver_opts or {})
    opts["print_timing_statistics"] = "yes"
    buf = io.StringIO()
    t_solve = time.time()
    try:
        r, ls = solve_with_fallback(m, linear_solver=linear_solver, max_iter=max_iter,
                                    log_callback=buf.write, options=opts)
        status = str(r.solver.termination_condition)
    except Exception as e:
        status, ls = "FAILED: " + str(e).splitlines()[0][:90], linear_solver
    t_solve = time.time() - t_solve

    log = buf.getvalue()
    g = lambda pat: (re.findall(pat, log) or ["-"])[-1]
    th = np.array([pyo.value(m.f[q, 0]) for q in m.PIX]).reshape(res, res)
    return dict(theta_hat=th, status=status, linear_solver=ls, gate_residual=gate_resid,
                t_sim=t_sim, t_build=t_build, t_solve=t_solve, t_cont=t_cont,
                cont_iters=cont_iters, cont_status=cont_status, n_vars=nv, n_cons=nc,
                iters=g(r"Number of Iterations\.*:\s*(\S+)"),
                nnz_hess=g(r"Number of nonzeros in Lagrangian Hessian\.*:\s*(\S+)"),
                nnz_jac=g(r"Number of nonzeros in equality constraint Jacobian\.*:\s*(\S+)"),
                ipopt_s=g(r"Total seconds in IPOPT \(w/o function evaluations\)\s*=\s*(\S+)"),
                fev_s=g(r"Total seconds in NLP function evaluations\s*=\s*(\S+)"),
                fact_s=g(r"LinearSystemFactorization\.*:\s*(\S+)"),
                theta_err=100.0 * float(np.sqrt(np.mean((th - theta) ** 2))) / float(theta.max()))
