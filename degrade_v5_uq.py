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
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyomo.environ as pyo

from degrade_v2_uq import measurement_rays, solve_with_fallback
from degrade_v3_uq import _neighbours
from degrade_v5 import V5Params, simulate, resolve, _phantom, _demo_sequence
from degrade_v2 import scale_to_optical_depth


def build_v5_model(theta_ref, seq, p: V5Params, image_res: int, *, f_bounds=None,
                   inline_Ipix: bool = False, inline_dw: bool = False,
                   inline_ft: bool = False, sigma_fixed=None, potential: bool = True):
    """Steps 1-7 of the reduced model as a Pyomo model.

    ``potential=False`` omits the whole ``Pi``/``sigma``/``phi`` block.  Legal ONLY at
    ``c_cp == 0``, where nothing reads ``phi``: the flux terms of :func:`_mass` are the only
    other consumer.  It exists for the ``I0 = 0, c_cp = 0`` continuation solve, where the block
    is provably inert -- ``Ipix = 0`` gives ``dw = 0`` and hence ``Pi = 0``, a finite reach makes
    ``A`` nonsingular so ``A phi = 0`` forces ``phi = 0`` exactly, and with ``phi`` appearing in
    no other row its multiplier is zero, so even the bilinear ``(sigma, phi)`` cross block
    contributes nothing.  Inert but not free: measured at grid 32 / K=4 it is 12,288 of the
    model's 34,520 variables, 36%, which made the continuation the same size as the problem it
    is supposed to cheaply initialise.
    """
    theta_ref = np.asarray(theta_ref, dtype=float)
    p = resolve(p, theta_ref)
    res = int(image_res)
    npix = res * res
    meas = measurement_rays(seq, res)
    K = len(meas)
    if K == 0:
        raise ValueError("no measurements: the sequence is empty")
    if not potential and p.c_cp != 0.0:
        raise ValueError(
            "potential=False needs c_cp == 0 (got %r): the flux terms of the mass balance read "
            "phi, so dropping the potential block at c_cp != 0 would silently build a DIFFERENT "
            "model rather than a cheaper one." % (p.c_cp,))

    m = pyo.ConcreteModel(name="degrade_v5")
    m.res, m.n_steps, m.meas, m.p = res, K, meas, p
    # carried so initialize_from_numpy is self-contained: it re-runs the numpy model
    # through the SAME fixed measurement sequence this model was built around.
    m.seq, m.theta_ref = tuple(tuple(x) for x in seq), np.array(theta_ref, dtype=float)
    m.inline_Ipix, m.inline_dw, m.inline_ft = bool(inline_Ipix), bool(inline_dw), bool(inline_ft)
    m.has_potential = bool(potential)

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

    if potential:
        m.Pi = pyo.Var(m.PIX, m.TM, initialize=0.0)

        def _pic(mm, q, k):
            return mm.Pi[q, k] * fm == _dwv(mm, q, k) * _ft(mm, q, k)
        m.c_Pi = pyo.Constraint(m.PIX, m.TM, rule=_pic)

        # sigma: a VARIABLE by default, or FROZEN to supplied constants.
        #
        # Frozen is the Picard iteration's frozen-sigma problem -- see v5_picard_demo.py. It
        # matters because A(ftilde) phi = Pi is BILINEAR in (sigma, phi), and that cross term is
        # what spoils the projected Hessian: it is off-diagonal with zero diagonal, so it
        # contributes an indefinite block at every iterate. Measured at grid 32 / K=4, sigma free
        # regularised on 62% of iterations against 3% for v4's pointwise driver. (An earlier
        # "201 of 200" figure was a parser defect, not a measurement: it counted the header and
        # the restoration lines. The correct test splits the iteration line and checks that
        # field 6, lg(rg), is not "-".) Holding sigma constant makes c_phi LINEAR in phi, so it
        # contributes nothing to the Hessian and the indefiniteness is gone by construction.
        m.sigma_frozen = sigma_fixed is not None
        if m.sigma_frozen:
            sf = np.asarray(sigma_fixed, dtype=float).reshape(npix, K)
            m.sig = pyo.Param(m.PIX, m.TM, initialize=lambda _m, q, k: float(sf[q, k]),
                              mutable=True, within=pyo.Reals)
        else:
            # sigma < 1 is structural (1 - exp(-x) < 1); the lower bound is left open because ft can
            # be marginally negative from the logistic residual and a hard 0 would make pinning the
            # gate trajectory an out-of-bounds write.
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
    else:
        # No sigma Var and no sigma Param: nothing downstream may read either, which is
        # what has_potential tells pin_model and initialize_from_numpy.
        m.sigma_frozen = False

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
            if m.has_potential:
                m.Pi[q, k].set_value(float(traj["Pi"][q, k]))
                if not m.sigma_frozen:
                    m.sig[q, k].set_value(float(traj["sig"][q, k]))
                m.phi[q, k].set_value(float(traj["phi"][q, k]))
            if fix:
                if not m.inline_Ipix:
                    m.Ipix[q, k].fix()
                if not m.inline_dw:
                    m.dw[q, k].fix()
                if not m.inline_ft:
                    m.ft[q, k].fix()
                if m.has_potential:
                    m.Pi[q, k].fix()
                    if not m.sigma_frozen:
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


def _reg_fraction(log: str):
    """``(regularised, total)`` IPOPT iterations.

    Column 6 of an iteration line is ``lg(rg)``; ``"-"`` means no Hessian regularisation was
    applied on that iteration.  Counting any other way is how the bogus "201 of 200" figure
    arose -- a substring test that also matched the header and the restoration lines.
    """
    n = r = 0
    for ln in log.splitlines():
        f = ln.split()
        if len(f) < 10 or not re.fullmatch(r"\d+r?", f[0]):
            continue
        n += 1
        if f[6] != "-":
            r += 1
    return r, n


def _shape_diagnostics(theta, f):
    """``(support_pct, half_pct, flips)`` -- what the damage did to the body's shape.

    Radii are measured about the **fixed initial centroid**, not a moving one, so a body that
    translates does not read as a body that contracted.  ``flips`` counts sign changes in the
    radially banded mass difference: ONE is coherent condensation (mass leaves the outside and
    arrives inside), several means mass is shuffling between neighbours.

    Kept identical to the forward tab's readout in ``app.py`` so the two cannot disagree.
    """
    theta = np.asarray(theta, dtype=float)
    f = np.asarray(f, dtype=float)
    nr, nc = theta.shape
    yy, xx = np.mgrid[0:nr, 0:nc]
    m0 = float(theta.sum())
    if m0 <= 0.0:
        return float("nan"), float("nan"), 0
    cx0, cy0 = (xx * theta).sum() / m0, (yy * theta).sum() / m0
    rad = np.sqrt((xx - cx0) ** 2 + (yy - cy0) ** 2).ravel()
    order = np.argsort(rad)
    rsort = rad[order]

    def _q(img, frac):
        w = np.clip(np.asarray(img, float).ravel(), 0.0, None)[order]
        c = np.cumsum(w)
        return float(rsort[np.searchsorted(c, frac * c[-1])]) if c[-1] > 0 else float("nan")

    r0, r1 = _q(theta, 0.99), _q(f, 0.99)
    h0, h1 = _q(theta, 0.50), _q(f, 0.50)
    d = (f - theta).reshape(nr, nc)
    r2 = rad.reshape(nr, nc)
    bands = [float(d[(r2 >= lo) & (r2 < lo + 2)].sum()) for lo in range(0, int(0.55 * nr), 2)]
    flips = sum(1 for i in range(len(bands) - 1) if bands[i] * bands[i + 1] < 0)
    sup = (100.0 * (r1 - r0) / r0) if r0 > 0 else float("nan")
    half = (100.0 * (h1 - h0) / h0) if h0 > 0 else float("nan")
    return sup, half, flips


@dataclass
class V5UQParams:
    """Inputs to :func:`run_v5_reconstruction`.  Physics defaults match the v5 tab's seeds."""

    image_res: int = 32
    optical_depth: float = 1.1
    beam_steps: tuple = ()             # (angle_deg, offset, n_beams) triples, _table_to_seq form
    phantom: Optional[np.ndarray] = None

    # --- v5 physics (V5Params) ---
    I0: float = 1.0
    c: float = 0.1
    a: float = 0.05
    b: float = 0.0
    c_cp: float = 0.3
    reach: Optional[float] = 7.0
    gamma: float = 100.0
    f_ref_frac: Optional[float] = 0.002
    beta: float = 1000.0
    flux: str = "upwind"
    eps_h: float = 1e-12
    dx: float = 1.0

    # --- estimation ---
    tv_weight: float = 0.001
    noise_sigma: float = 0.0           # 0 = noiseless data, as v1/v2 do
    continuation: bool = True          # seed from the I0 = 0 linear-tomography + TV solve
    gate: bool = True
    ipopt_max_iter: int = 3000
    linear_solver: str = "ma97"
    solver_opts: Optional[dict] = None

    def physics(self, **over) -> V5Params:
        kw = dict(I0=self.I0, c=self.c, a=self.a, b=self.b, c_cp=self.c_cp, reach=self.reach,
                  gamma=self.gamma, f_ref_frac=self.f_ref_frac, beta=self.beta, flux=self.flux,
                  eps_h=self.eps_h, dx=self.dx)
        kw.update(over)
        return V5Params(**kw)


@dataclass
class V5UQResults:
    """Arrays and scalars, not matplotlib figures -- the caller draws.

    No covariance and no D-optimality: v5 is scoped to the damage model and its solve, so the
    k_aug step v2 carries is deliberately absent rather than merely unimplemented.
    """

    theta_true: np.ndarray
    theta_hat: np.ndarray
    f_final_true: np.ndarray
    f_final_hat: np.ndarray

    # --- the solve ---
    status: str = ""
    linear_solver: str = ""
    iters: str = "-"
    regularised: int = 0               # IPOPT iterations that needed Hessian regularisation
    n_iter_lines: int = 0              # ... out of this many
    ipopt_s: str = "-"
    fev_s: str = "-"
    n_vars: int = 0
    n_cons: int = 0
    t_sim: float = float("nan")
    t_cont: float = float("nan")
    t_solve: float = float("nan")

    # --- how good the start was, and whether the model still means anything ---
    forward_residual: float = float("nan")   # the drift gate, on THIS geometry
    init_residual: float = float("nan")      # worst dynamic residual at the starting point
    continuation_status: str = "skipped"
    continuation_iters: str = "-"
    theta_rms_cont: float = float("nan")     # error of the continuation estimate alone

    # --- the answer ---
    obs_rms: float = float("nan")            # fit residual, RMS over all rays
    theta_rms: float = float("nan")          # ||theta_hat - theta_true|| RMS, synthetic only
    theta_pct_peak: float = float("nan")     # the same as a % of peak theta -- the quotable one
    n_theta_at_lower: int = 0
    n_theta_at_upper: int = 0
    n_theta_interior: int = 0

    # --- what the damage actually did, so the answer can be read in context ---
    mass_true: float = float("nan")
    mass_hat: float = float("nan")
    ck_max: float = float("nan")             # C_k positivity number; sufficient bound is <= 1
    phi_max: float = float("nan")
    support_pct: float = float("nan")
    half_pct: float = float("nan")
    flips: int = 0

    @property
    def regularised_pct(self) -> float:
        return 100.0 * self.regularised / max(self.n_iter_lines, 1)


def run_v5_reconstruction(params: V5UQParams, log_callback=None) -> V5UQResults:
    """Estimate ``theta`` from the v5 dynamics.  ONE monolithic solve, no Picard iteration.

    If IPOPT does not converge this reports that and stops -- it does not retry on a better
    start and it does not fall back.  The three numbers that separate the two explanations are
    on the result: ``init_residual`` (was the start dynamically feasible?), ``regularised`` /
    ``n_iter_lines`` (was the Hessian indefinite throughout?) and ``iters``.  A low init
    residual with a high regularisation fraction points at the bilinear ``(sigma, phi)`` block;
    a high init residual points at the start.
    """
    def say(msg):
        if log_callback:
            log_callback(msg)

    res = int(params.image_res)
    seq = tuple(params.beam_steps)
    if not seq:
        raise ValueError("no measurements: take at least one before reconstructing")
    p = params.physics()
    base = _phantom(res) if params.phantom is None else np.asarray(params.phantom, dtype=float)
    theta = scale_to_optical_depth(base, params.optical_depth, res)
    p = resolve(p, theta)
    scale = float(np.abs(theta).max())

    # --- data, from the numpy simulator: an implementation independent of the NLP ----------
    t_sim = time.time()
    f_true, infos, y = simulate(theta, seq, p, res, record_observations=True)
    t_sim = time.time() - t_sim
    if params.noise_sigma > 0.0:
        rng = np.random.default_rng(0)      # fixed seed: a rerun must be comparable
        y = [np.asarray(v, float) + params.noise_sigma * rng.standard_normal(np.shape(v))
             for v in y]

    # --- the gate: does the Pyomo model still reproduce the simulator HERE? -----------------
    gate_resid = float("nan")
    if params.gate:
        say("Checking the Pyomo model against the simulator on this geometry...\n")
        mg = build_v5_model(theta, seq, p, res)
        pin_model(mg, numpy_trajectory(theta, seq, p, res))
        gate_resid, where = max_residual(mg)
        del mg
        say("    max constraint residual %.3e  (%s)\n" % (gate_resid, where))
        if gate_resid > 1e-8:
            raise RuntimeError(
                "The Pyomo model no longer reproduces degrade_v5.simulate on this geometry "
                "(residual %.3e at %s). Reconstructing against it would not mean anything; "
                "run degrade_v5_uq.check_forward() to localise the disagreement."
                % (gate_resid, where))

    opts = dict(params.solver_opts or {})
    opts.setdefault("ma97_order", "metis")

    # --- continuation: I0 = 0, c_cp = 0 is linear tomography + TV ---------------------------
    # The seed matters more than its amplitude: a mis-SCALED theta converges in a handful of
    # iterations, a structurally wrong field does not (a flat mean field dies in restoration).
    # The continuation estimate is structurally close by construction, which is the point.
    theta0 = np.full_like(theta, float(theta.mean()))
    cont_status, cont_iters, t_cont = "skipped", "-", 0.0
    if params.continuation:
        say("Continuation solve at I0 = 0 (linear tomography + TV)...\n")
        tc = time.time()
        p0 = params.physics(I0=0.0, c_cp=0.0)
        # potential=False: at I0 = 0 the whole Pi/sigma/phi block is inert, and carrying it made
        # the continuation the same size as the problem it is supposed to cheaply initialise.
        m0 = build_v5_model(theta, seq, resolve(p0, theta), res,
                            f_bounds=(0.0, 1.5 * scale), potential=False)
        for q in m0.PIX:
            m0.f[q, 0].set_value(float(theta0.ravel()[q]))
        add_estimation_objective(m0, y, params.tv_weight, scale)
        b0 = io.StringIO()

        def _cont_log(chunk):
            b0.write(chunk)
            if log_callback:
                log_callback(chunk)
        try:
            r0, _ls0 = solve_with_fallback(m0, linear_solver=params.linear_solver,
                                           max_iter=params.ipopt_max_iter,
                                           log_callback=_cont_log, options=opts)
            cont_status = str(r0.solver.termination_condition)
            theta0 = np.array([pyo.value(m0.f[q, 0]) for q in m0.PIX]).reshape(res, res)
        except Exception as exc:
            cont_status = "FAILED: " + str(exc).splitlines()[0][:60]
        cont_iters = (re.findall(r"Number of Iterations\.*:\s*(\S+)", b0.getvalue()) or ["-"])[-1]
        del m0
        t_cont = time.time() - tc
        say("    %s (%s iterations)\n" % (cont_status, cont_iters))
    e_cont = 100.0 * float(np.sqrt(np.mean((theta0 - theta) ** 2))) / scale

    # --- the monolithic estimation NLP ------------------------------------------------------
    say("Building the v5 estimation NLP...\n")
    m = build_v5_model(theta, seq, p, res, f_bounds=(0.0, 1.5 * scale))
    init = initialize_from_numpy(m, theta0)
    add_estimation_objective(m, y, params.tv_weight, scale)
    n_v = sum(1 for _ in m.component_data_objects(pyo.Var))
    n_c = sum(1 for _ in m.component_data_objects(pyo.Constraint, active=True))
    say("    %d variables, %d constraints; start residual %.3e (%s)\n"
        % (n_v, n_c, init["residual"], init["worst_row"]))

    buf = io.StringIO()

    def _log(chunk):
        buf.write(chunk)
        if log_callback:
            log_callback(chunk)

    t_solve = time.time()
    try:
        r, ls = solve_with_fallback(m, linear_solver=params.linear_solver,
                                    max_iter=params.ipopt_max_iter, log_callback=_log,
                                    options=dict(opts, print_timing_statistics="yes"))
        status = str(r.solver.termination_condition)
    except Exception as exc:
        status, ls = "FAILED: " + str(exc).splitlines()[0][:90], params.linear_solver
    t_solve = time.time() - t_solve

    log = buf.getvalue()
    grab = lambda pat: (re.findall(pat, log) or ["-"])[-1]
    reg, n_lines = _reg_fraction(log)

    theta_hat = np.array([pyo.value(m.f[q, 0]) for q in m.PIX]).reshape(res, res)
    f_hat = np.array([pyo.value(m.f[q, m.n_steps]) for q in m.PIX]).reshape(res, res)
    resid = [pyo.value(m.yobs[k, j]) - float(y[k][j]) for (k, j, _n) in m.obs_index]

    # eq:xd_box active set, counted before anything else reads the solution
    lo, hi, tol = 0.0, 1.5 * scale, 1e-9 * max(scale, 1.0)
    flat_hat = theta_hat.ravel()
    at_lo = int(np.sum(flat_hat <= lo + tol))
    at_hi = int(np.sum(flat_hat >= hi - tol))

    sup, half, flips = _shape_diagnostics(theta, f_true)
    out = V5UQResults(
        theta_true=theta, theta_hat=theta_hat, f_final_true=f_true, f_final_hat=f_hat,
        status=status, linear_solver=ls, iters=grab(r"Number of Iterations\.*:\s*(\S+)"),
        regularised=reg, n_iter_lines=n_lines,
        ipopt_s=grab(r"Total seconds in IPOPT \(w/o function evaluations\)\s*=\s*(\S+)"),
        fev_s=grab(r"Total seconds in NLP function evaluations\s*=\s*(\S+)"),
        n_vars=n_v, n_cons=n_c, t_sim=t_sim, t_cont=t_cont, t_solve=t_solve,
        forward_residual=gate_resid, init_residual=float(init["residual"]),
        continuation_status=cont_status, continuation_iters=cont_iters, theta_rms_cont=e_cont,
        obs_rms=float(np.sqrt(np.mean(np.square(resid)))) if resid else float("nan"),
        theta_rms=float(np.sqrt(np.mean((theta_hat - theta) ** 2))),
        n_theta_at_lower=at_lo, n_theta_at_upper=at_hi,
        n_theta_interior=int(flat_hat.size - at_lo - at_hi),
        mass_true=float(f_true.sum()), mass_hat=float(f_hat.sum()),
        ck_max=max((i.compaction for i in infos), default=float("nan")),
        phi_max=max((i.phi_max for i in infos), default=float("nan")),
        support_pct=sup, half_pct=half, flips=flips,
    )
    out.theta_pct_peak = 100.0 * out.theta_rms / scale
    say("    %s, %s iterations, %d/%d regularised, theta error %.2f%% of peak\n"
        % (status, out.iters, reg, n_lines, out.theta_pct_peak))
    return out

# --- the FORWARD solve ----------------------------------------------------------------------

def forward_solve(theta, seq, p: V5Params, image_res: int, *, linear_solver="ma97",
                  solver_opts=None, max_iter=3000, start="undamaged", tee=False,
                  sigma_fixed=None):
    """Fix ``f[:,0] = theta`` and let IPOPT find the whole trajectory.

    Strictly stronger than :func:`check_forward`, which pins every variable to a
    :func:`degrade_v5.simulate` trajectory and evaluates residuals.  That check can only say the
    equations were transcribed correctly at a point it was handed.  This one starts IPOPT
    somewhere else and asks it to *find* the trajectory, so it additionally says the model is
    square, solvable, and scaled well enough to converge -- which for v5 is a real question,
    since the ESTIMATION NLP does not converge monolithically at this grid.

    ``start="undamaged"`` initialises every stage at ``theta`` and the undamaged potential, so
    convergence to the damaged trajectory is genuinely found rather than handed over.
    ``start="true"`` starts at the answer and only confirms it is a fixed point.

    Returns a dict with the solved trajectory and the comparison against the numpy simulator.
    """
    res = int(image_res)
    p = resolve(p, theta)
    npix = res * res
    K = len(measurement_rays(seq, res))

    m = build_v5_model(theta, seq, p, res, sigma_fixed=sigma_fixed)
    # The forward problem is SQUARE: fixing f[:,0] removes the only degrees of freedom the
    # estimation problem has. A constant objective keeps IPOPT solving a feasibility problem
    # rather than optimising anything.
    flat = np.asarray(theta, dtype=float).ravel()
    for q in m.PIX:
        m.f[q, 0].fix(float(flat[q]))
    m.obj = pyo.Objective(expr=0.0)

    if start == "true":
        pin_model(m, numpy_trajectory(theta, seq, p, res), fix=False)
    else:
        und = numpy_trajectory(theta, [], p, res) if False else None
        # undamaged start: every stage at theta, zero fluence, zero dose, zero potential
        for q in m.PIX:
            for k in m.T:
                if k:
                    m.f[q, k].set_value(float(flat[q]))
            for k in m.TM:
                if not m.inline_Ipix:
                    m.Ipix[q, k].set_value(0.0)
                if not m.inline_dw:
                    m.dw[q, k].set_value(0.0)
                if not m.inline_ft:
                    m.ft[q, k].set_value(float(flat[q]))
                m.Pi[q, k].set_value(0.0)
                m.phi[q, k].set_value(0.0)
                if not m.sigma_frozen:
                    m.sig[q, k].set_value(0.0)
        for idx in m.CH:
            m.S[idx].set_value(0.0)
        for idx in m.RAY:
            m.yobs[idx].set_value(0.0)

    nv = sum(1 for v in m.component_data_objects(pyo.Var) if not v.fixed)
    nc = sum(1 for _ in m.component_data_objects(pyo.Constraint, active=True))
    buf = io.StringIO()
    t0 = time.time()
    try:
        r, ls = solve_with_fallback(m, linear_solver=linear_solver, max_iter=max_iter,
                                    log_callback=buf.write, tee=tee,
                                    options=dict(solver_opts or {}))
        status = str(r.solver.termination_condition)
    except Exception as exc:
        status, ls = "FAILED: " + str(exc).splitlines()[0][:90], linear_solver
    wall = time.time() - t0

    log = buf.getvalue()
    g = lambda pat: (re.findall(pat, log) or ["-"])[-1]
    f_py = np.array([[pyo.value(m.f[q, k]) for k in range(K + 1)] for q in m.PIX])
    ref = numpy_trajectory(theta, seq, p, res)
    f_np = ref["f"]
    scale = max(float(np.abs(f_np).max()), 1e-300)
    return dict(
        status=status, linear_solver=ls, wall=wall, n_vars=nv, n_cons=nc,
        dof=nv - nc,
        iters=g(r"Number of Iterations\.*:\s*(\S+)"),
        nnz_jac=g(r"Number of nonzeros in equality constraint Jacobian\.*:\s*(\S+)"),
        nnz_hess=g(r"Number of nonzeros in Lagrangian Hessian\.*:\s*(\S+)"),
        f_pyomo=f_py, f_numpy=f_np,
        err_abs=float(np.abs(f_py - f_np).max()),
        err_rel=float(np.abs(f_py - f_np).max()) / scale,
        err_final_rel=float(np.abs(f_py[:, -1] - f_np[:, -1]).max()) / scale,
        phi_rel=float(np.abs(np.array([[pyo.value(m.phi[q, k]) for k in range(K)]
                                       for q in m.PIX]) - ref["phi"]).max())
        / max(float(np.abs(ref["phi"]).max()), 1e-300),
    )


def forward_solve_staged(theta, seq, p: V5Params, image_res: int, *, linear_solver="ma97",
                         solver_opts=None, max_iter=3000, warmstart_duals=True, verbose=False):
    """Solve the horizon one step at a time, then warm-start the full horizon from the result.

    The undamaged start is a poor guess for a long horizon: every stage begins at ``theta`` while
    the true field decays monotonically, so the initial error grows with k and the last stages
    start furthest from their answer.  Solving stage by stage costs K small square solves and
    hands the full model a trajectory that is already nearly feasible everywhere.

    Duals are carried too, not just primals.  Each single-step solve produces multipliers for the
    same rows the full model has at that stage, so they transfer directly, and IPOPT can start
    from them instead of rediscovering them -- which is what ``warm_start_init_point`` wants.

    Returns the same dict as :func:`forward_solve`, plus the staging cost.
    """
    res = int(image_res)
    p = resolve(p, theta)
    npix = res * res
    K = len(measurement_rays(seq, res))
    flat = np.asarray(theta, dtype=float).ravel()

    # --- stage-by-stage --------------------------------------------------------------------
    acc = {"f": np.zeros((npix, K + 1))}
    for nm in ("Ipix", "dw", "ft", "Pi", "sig", "phi"):
        acc[nm] = np.zeros((npix, K))
    acc["f"][:, 0] = flat
    acc["S"], acc["yobs"] = {}, {}
    duals = {}
    field = np.asarray(theta, dtype=float).copy()
    t_stage = time.time()
    for k in range(K):
        sub = build_v5_model(field, seq[k:k + 1], p, res)
        for q in sub.PIX:
            sub.f[q, 0].fix(float(field.ravel()[q]))
        sub.obj = pyo.Objective(expr=0.0)
        sub.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        solve_with_fallback(sub, linear_solver=linear_solver, max_iter=max_iter,
                            log_callback=lambda _s: None, options=dict(solver_opts or {}))
        for q in sub.PIX:
            acc["f"][q, k + 1] = pyo.value(sub.f[q, 1])
            for nm in ("Ipix", "dw", "ft", "Pi", "phi"):
                acc[nm][q, k] = pyo.value(getattr(sub, nm)[q, 0])
            acc["sig"][q, k] = float(pyo.value(sub.sig[q, 0]))
        for (kk, j, t) in sub.CH:
            acc["S"][(k, j, t)] = pyo.value(sub.S[kk, j, t])
        for (kk, j) in sub.RAY:
            acc["yobs"][(k, j)] = pyo.value(sub.yobs[kk, j])
        for c in sub.component_data_objects(pyo.Constraint, active=True):
            idx = c.index()
            comp = c.parent_component().name
            if isinstance(idx, tuple) and len(idx) >= 2:
                duals[(comp, idx[0], k)] = sub.dual.get(c, 0.0)
        field = acc["f"][:, k + 1].reshape(res, res)
        if verbose:
            print("    stage %d done" % (k + 1)); sys.stdout.flush()
        del sub
    t_stage = time.time() - t_stage

    # --- full horizon, warm-started from the staged trajectory -------------------------------
    m = build_v5_model(theta, seq, p, res)
    for q in m.PIX:
        m.f[q, 0].fix(float(flat[q]))
    m.obj = pyo.Objective(expr=0.0)
    for q in m.PIX:
        for k in m.T:
            if k:
                m.f[q, k].set_value(float(acc["f"][q, k]))
        for k in m.TM:
            for nm in ("Ipix", "dw", "ft", "Pi", "phi"):
                getattr(m, nm)[q, k].set_value(float(acc[nm][q, k]))
            m.sig[q, k].set_value(float(acc["sig"][q, k]))
    for idx in m.CH:
        m.S[idx].set_value(float(acc["S"].get(idx, 0.0)))
    for idx in m.RAY:
        m.yobs[idx].set_value(float(acc["yobs"].get(idx, 0.0)))

    opts = dict(solver_opts or {})
    if warmstart_duals:
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)
        for c in m.component_data_objects(pyo.Constraint, active=True):
            idx = c.index()
            if isinstance(idx, tuple) and len(idx) >= 2:
                v = duals.get((c.parent_component().name, idx[0], idx[-1]))
                if v is not None:
                    m.dual[c] = float(v)
        opts.setdefault("warm_start_init_point", "yes")
        opts.setdefault("warm_start_bound_push", 1e-9)
        opts.setdefault("warm_start_mult_bound_push", 1e-9)

    nv = sum(1 for v in m.component_data_objects(pyo.Var) if not v.fixed)
    nc = sum(1 for _ in m.component_data_objects(pyo.Constraint, active=True))
    buf = io.StringIO()
    t0 = time.time()
    try:
        r, ls = solve_with_fallback(m, linear_solver=linear_solver, max_iter=max_iter,
                                    log_callback=buf.write, options=opts)
        status = str(r.solver.termination_condition)
    except Exception as exc:
        status, ls = "FAILED: " + str(exc).splitlines()[0][:90], linear_solver
    wall = time.time() - t0

    log = buf.getvalue()
    g = lambda pat: (re.findall(pat, log) or ["-"])[-1]
    f_py = np.array([[pyo.value(m.f[q, k]) for k in range(K + 1)] for q in m.PIX])
    ref = numpy_trajectory(theta, seq, p, res)
    scale = max(float(np.abs(ref["f"]).max()), 1e-300)
    return dict(status=status, linear_solver=ls, wall=wall, t_stage=t_stage,
                n_vars=nv, n_cons=nc, dof=nv - nc,
                iters=g(r"Number of Iterations\.*:\s*(\S+)"),
                f_pyomo=f_py, f_numpy=ref["f"],
                err_rel=float(np.abs(f_py - ref["f"]).max()) / scale,
                staged_err_rel=float(np.abs(acc["f"] - ref["f"]).max()) / scale)


# --- initialisation from the numpy model ------------------------------------------------

def initialize_from_numpy(m, theta_seed=None, *, fix_theta=False, verbose=False):
    """Initialise every variable of a v5 Pyomo model from a :mod:`degrade_v5` run.

    **Spec-agnostic by construction.**  The forward model and the estimation model differ only
    in whether ``f[:,0]`` is fixed; the initialisation is the same operation in both, because in
    both the measurement sequence is *given*.  So: hold the angles fixed, run the numpy model
    through them to get the field at every time step, and copy the result across.  Nothing here
    inspects which spec it was handed.

    ``theta_seed`` is the field the numpy run starts from -- the quantity the Pyomo model is
    being initialised *about*, not necessarily the truth:

    * forward     -- the true ``theta``.  The trajectory is then the answer, and the model starts
                     feasible to ~1e-16.
    * estimation  -- the current estimate (a flat field, a continuation solution, a previous
                     outer iterate).  Every dynamic row then starts at residual ~0 and only the
                     data-fit rows are wrong, which is the initialisation the v2/v3 drivers used
                     and the reason they began dynamically feasible.

    Defaults to the ``theta_ref`` the model was built with.

    ``fix_theta`` additionally fixes ``f[:,0]``, which is what turns the estimation model into
    the forward one.  Left False the caller keeps whatever the model already had.

    Returns a dict with the seed, the worst constraint residual after initialising, and which
    blocks were written -- the residual is the useful number, since it says how good the start
    actually is rather than asserting that it is good.
    """
    # Recorded BEFORE the reshape below: `reshape` returns a view, not the same object, so an
    # `is` test against m.theta_ref afterwards is always False even on the default path.
    default_seed = theta_seed is None
    theta_seed = m.theta_ref if default_seed else np.asarray(theta_seed, dtype=float)
    theta_seed = np.asarray(theta_seed, dtype=float).reshape(m.res, m.res)

    traj = numpy_trajectory(theta_seed, m.seq, m.p, m.res)
    written = []

    # f and the per-stage scalar fields. Blocks absent through inlining, or frozen to Params,
    # are skipped rather than special-cased at the call site.
    for q in m.PIX:
        for k in m.T:
            m.f[q, k].set_value(float(traj["f"][q, k]))
        for k in m.TM:
            if not m.inline_Ipix:
                m.Ipix[q, k].set_value(float(traj["Ipix"][q, k]))
            if not m.inline_dw:
                m.dw[q, k].set_value(float(traj["dw"][q, k]))
            if not m.inline_ft:
                m.ft[q, k].set_value(float(traj["ft"][q, k]))
            if m.has_potential:
                m.Pi[q, k].set_value(float(traj["Pi"][q, k]))
                m.phi[q, k].set_value(float(traj["phi"][q, k]))
                if not m.sigma_frozen:
                    m.sig[q, k].set_value(float(traj["sig"][q, k]))
    written += ["f"] + [n for n, on in (("Ipix", m.inline_Ipix), ("dw", m.inline_dw),
                                        ("ft", m.inline_ft)) if not on]
    if m.has_potential:
        written += ["Pi", "phi"] + ([] if m.sigma_frozen else ["sig"])

    for idx in m.CH:
        m.S[idx].set_value(float(traj["S"][idx]))
    for idx in m.RAY:
        m.yobs[idx].set_value(float(traj["yobs"][idx]))
    written += ["S", "yobs"]

    if fix_theta:
        for q in m.PIX:
            m.f[q, 0].fix(float(theta_seed.ravel()[q]))

    resid, where = max_residual(m)
    out = dict(seed_is_theta_ref=default_seed, residual=resid, worst_row=where,
               blocks=written, n_steps=m.n_steps, res=m.res,
               sigma_frozen=m.sigma_frozen, theta_fixed=bool(fix_theta))
    if verbose:
        print("  initialised %d blocks from the numpy model over %d steps; worst residual "
              "%.3e (%s)" % (len(written), m.n_steps, resid, where))
    return out
