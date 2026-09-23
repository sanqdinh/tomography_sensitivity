"""Pyomo transcription of v6, the implicit-transport damage model.  Forward direction only.

The v6 counterpart of :mod:`degrade_v5_uq`, and deliberately narrower: this module builds the
model and validates it against :func:`degrade_v6.simulate`.  **There is no estimation NLP, no
objective, no k_aug.**  Those come after the forward direction is trusted; the seam is
:func:`add_estimation_objective` in ``degrade_v5_uq``, which this module does not yet mirror.

Same checking discipline as v2 and v5: the reference trajectory comes from
:func:`degrade_v6.simulate`, which is numpy and scipy, so the model and the thing it is checked
against stay two independent implementations of section 3.2.  Read the methodological lesson in
CLAUDE.md before trusting the residual: agreeing to 1e-16 proves a shared *convention*,
including a wrong one.  What earns trust here is that the two implementations solve steps 5-6 by
completely different routes -- numpy factorises a sparse matrix and back-substitutes, Pyomo hands
the same rows to IPOPT as constraints and never forms the matrix -- so a residual at round-off
says the ROWS agree, not merely that one call reproduced another.

Blocks per stage: ``S`` (the shielding chain), ``Ipix``, ``dw``, ``ft``, ``sig``, ``Pi``, ``phi``
-- all v5's, unchanged, because steps 1 to 4 are unchanged -- then ``sp`` on directed faces and
the implicit mass row, which are v6's.

THE ONE REAL TRANSCRIPTION PROBLEM: softplus
--------------------------------------------
``eq:xd_rate_function`` is ``phi_eta(z) = eta log(1 + exp(z/eta))``.  Written that way in Pyomo it
**overflows**: ``exp`` dies above 709.8, so at ``eta = 1e-3`` any face carrying ``|dP| > 0.71``
makes the model unevaluable.

That is not a hypothetical margin.  Measured over the tab's own slider ranges (grid 32 and 64,
``c_cp`` 0.3 and 3, reach 7 and 32 px, ``c_omega`` 0.1 to 3, 9 projections)::

    c_omega = 0.1  ->  max |dP| = 0.20 .. 0.36    naive form dies at eta <= 1e-4
    c_omega = 0.4  ->  max |dP| = 0.80 .. 3.79    naive form dies at eta <= 1e-3  (the default)
    c_omega = 3.0  ->  max |dP| = 2.59 .. 80.3    naive form dies at every eta the tab offers

``c_omega = 0.4`` is the setting ``experiment_v6_fractionation.py`` runs at, so the direct
transcription would fail on a configuration already in the repo, at the default ``eta``.

It is not fixable by clamping either: subsec:system itself says the function "should be
implemented with a stable softplus routine rather than by forming ``exp(z/eta)`` directly".
numpy gets that from ``logaddexp``; Pyomo has no such primitive, and ``max``/``abs``-based
stabilisations are non-smooth and therefore inadmissible in an NLP.

The transcription used here lifts **both** directed rates of a face as variables, ``u = sp(z)``
and ``v = sp(-z)``, and pins them with two rows::

    u - v = z                            (LINEAR, and exact -- sp(z) - sp(-z) = z identically)
    exp(-u/eta) + exp(-v/eta) = 1

Both are smooth. The equations uniquely imply ``u, v >= 0`` at every feasible point; the
variable floor ``-eta`` also limits either exponent to at most 1 at infeasible iterates, so
**neither exponential can overflow at any field whatsoever**. The pair is also unique: substituting
``v = u - z`` reduces the second row to ``exp(-u/eta)(1 + exp(z/eta)) = 1``, whose only root is
``u = eta log(1 + exp(z/eta))``.  Verified against ``logaddexp`` to 7.1e-15 on the first row and
2.2e-16 on the second, over ``eta`` in 1e-1..1e-5 and ``z`` in -50..50.

It degrades gracefully rather than breaking.  For ``z >> eta`` the true answer is ``u = z``,
``v ~ eta exp(-z/eta)``, and once ``exp(-u/eta)`` underflows to zero the second row reads
``exp(-v/eta) = 1``, i.e. ``v = 0`` -- the exact asymptote, which is also what numpy returns
there.  What underflow costs is the row's *derivative* with respect to ``u``, so a solve run deep
in that regime hands IPOPT a rank-deficient block.  :func:`check_forward` reports
``max |dP|/eta`` so the margin is visible rather than assumed.

The implicit mass balance is EASIER here than in numpy
-------------------------------------------------------
``eq:xd_implicit_transport`` is a global sparse solve for the numpy model.  In the NLP it is just
a row -- ``f_{k+1}`` is already a variable, so the solver's own factorisation does the work and
nothing is nested.  It is bilinear in ``(sp, f)``, so its second derivatives are constants.  Same
argument v5 makes for lifting ``phi``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import pyomo.environ as pyo

from degrade_v2 import scale_to_optical_depth
from degrade_v2_uq import measurement_rays, solve_with_fallback
from degrade_v3_uq import _neighbours
# TV and the IPOPT-log parser are v5's and unchanged by the v6 diff: _tv_expression reads only
# m.f[:,0] and m.res, so it works on a v6 model as-is. Importing beats a second copy -- v2's TV
# hardcodes eps = 1e-4, which swamps a theta peaking near 0.03, and v5 already fixed that.
from degrade_v5_uq import _tv_expression, _reg_fraction
from degrade_v6 import (V6Params, simulate, simulate_simultaneous, resolve, softplus,
                        compaction_potential, shape_diagnostics, select_eta,
                        ETA_RATIO_LO, ETA_RATIO_HI, _phantom, _demo_sequence)


def _measurements(seq, res, simultaneous: bool):
    """``measurement_rays``, optionally collapsed to ONE measurement carrying every ray.

    That collapse *is* the simultaneous schedule as far as this model is concerned: the blocks
    below accumulate ``I_terms``/``Id_terms`` per ``(pixel, k)`` over all rays of measurement
    ``k``, so putting every angle's rays in one entry sums their dose fields before the single
    decay, potential solve and transport solve -- which is what
    :func:`degrade_v6.step_simultaneous` does.  The merged entry's angle is unused downstream.
    """
    meas = measurement_rays(seq, res)
    if not simultaneous or len(meas) <= 1:
        return meas
    rays = [r for _ang, rs in meas for r in rs]
    return [(meas[0][0], rays)]


# --- model ---------------------------------------------------------------------------------

def build_v6_model(theta_ref, seq, p: V6Params, image_res: int, *, f_bounds=None,
                   potential: bool = True, simultaneous: bool = False):
    """Steps 1-7 of section 3.2 as a Pyomo model.  ``theta_ref`` seeds every variable.

    ``potential=False`` omits the ``Pi``/``sig``/``phi``/``sp`` blocks entirely and is legal ONLY
    at ``c_cp == 0``, where the mass row reads none of them.  Same switch, and the same reason,
    as ``degrade_v5_uq.build_v5_model``: it exists so an ``I0 = 0, c_cp = 0`` continuation is
    actually cheaper than the problem it initialises rather than the same size.
    """
    theta_ref = np.asarray(theta_ref, dtype=float)
    p = resolve(p, theta_ref)
    res = int(image_res)
    npix = res * res
    meas = _measurements(seq, res, simultaneous)
    K = len(meas)
    if K == 0:
        raise ValueError("no measurements: the sequence is empty")
    if not potential and p.c_cp != 0.0:
        raise ValueError(
            "potential=False needs c_cp == 0 (got %r): the mass row reads sp, which reads phi, "
            "so dropping the block at c_cp != 0 would silently build a DIFFERENT model rather "
            "than a cheaper one." % (p.c_cp,))
    if not (p.eta > 0.0):
        raise ValueError("eta = %r: eq:xd_rate_function needs eta > 0." % (p.eta,))

    m = pyo.ConcreteModel(name="degrade_v6")
    m.res, m.n_steps, m.meas, m.p = res, K, meas, p
    m.seq, m.theta_ref = tuple(tuple(x) for x in seq), np.array(theta_ref, dtype=float)
    m.has_potential = bool(potential)
    m.simultaneous = bool(simultaneous)

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

    # I_terms carries no chord (the fluence sum, which drives the decay); Id_terms carries it
    # (the chord-weighted sum, which drives the converted fraction). The split is the one
    # eq:xd_local_intensity and eq:xd_converted_fraction make in their subscripts.
    I_terms = {(q, k): [] for q in range(npix) for k in range(K)}
    Id_terms = {(q, k): [] for q in range(npix) for k in range(K)}
    for k, (_ang, rays) in enumerate(meas):
        for j, (_r, walk) in enumerate(rays):
            for t, (pix, chord, _sh) in enumerate(walk):
                I_terms[(pix, k)].append((k, j, t))
                Id_terms[(pix, k)].append(((k, j, t), chord))

    # THE PER-CROSSING LOCAL FLUENCE, LIFTED. ``L[k,j,t] = I0 exp(-S[k,j,t])`` is the fluence
    # eq:xd_local_intensity delivers at one crossing, and it is read TWICE: unweighted by the
    # Ipix sum that drives the decay, and chord-weighted by the dose sum that drives dw.
    #
    # Written inline it was built twice -- measured at grid 12 / K=6, 952 distinct exp(-S)
    # values became 2656 exp nodes -- and, worse, the dw row came out as exp(-sum(exp(...))).
    # That nested form makes the row's Hessian couple every S along every ray through the pixel,
    # turning what should be a handful of entries into a dense block. Lifting L makes c_Ipix and
    # the dose sum LINEAR and leaves exactly one exp per row, each of a single variable. Same
    # feasible set, same solution: this is a change of algebra, not of model. It is the same
    # argument v5 makes for lifting Pi, sigma and phi rather than inlining them.
    m.LX = pyo.Set(initialize=sorted({i for ts in I_terms.values() for i in ts}),
                   dimen=3, ordered=True)
    # c_L uniquely makes L positive. A redundant L >= 0 bound violates LICQ at I0 = 0, where
    # c_L is exactly L = 0 for every crossing (the continuation model).
    m.L = pyo.Var(m.LX, initialize=float(p.I0))

    def _lc(mm, k, j, t):
        return mm.L[k, j, t] == p.I0 * pyo.exp(-mm.S[k, j, t])
    m.c_L = pyo.Constraint(m.LX, rule=_lc)

    m.Ipix = pyo.Var(m.PIX, m.TM, initialize=0.0)

    def _ip(mm, q, k):                                   # LINEAR
        return mm.Ipix[q, k] == sum(mm.L[i] for i in I_terms[(q, k)])
    m.c_Ipix = pyo.Constraint(m.PIX, m.TM, rule=_ip)

    # --- 2. converted fraction. The chord-weighted dose sum is lifted too, so this is one
    # exp of one variable instead of an exp of a sum of exps.
    # The equation makes Z nonnegative at every feasible point. Keep a finite safety floor for
    # exp(-Z), but put the physical zero strictly inside it so an unilluminated pixel does not
    # duplicate c_Z with an active lower bound.
    m.Z = pyo.Var(m.PIX, m.TM, bounds=(-1.0, None), initialize=0.0)

    def _zc(mm, q, k):                                   # LINEAR
        return mm.Z[q, k] == sum(p.c * ch * mm.L[i] for i, ch in Id_terms[(q, k)])
    m.c_Z = pyo.Constraint(m.PIX, m.TM, rule=_zc)

    m.dw = pyo.Var(m.PIX, m.TM, initialize=0.0)

    def _dwc(mm, q, k):
        return mm.dw[q, k] == 1.0 - pyo.exp(-mm.Z[q, k])
    m.c_dw = pyo.Constraint(m.PIX, m.TM, rule=_dwc)

    # --- 3. mass loss, carried as a variable so the rows downstream stay low order ---------
    m.ft = pyo.Var(m.PIX, m.TM, initialize=lambda _m, q, k: float(flat[q]))

    def _ftc(mm, q, k):
        # b is 0 by default and Pyomo does not fold -0.0*x**2, so the quadratic term would
        # otherwise leave a dead node in every one of these rows.
        expo = -p.a * mm.Ipix[q, k]
        if p.b:
            expo = expo - p.b * mm.Ipix[q, k] ** 2
        return mm.ft[q, k] == mm.f[q, k] * pyo.exp(expo)
    m.c_ft = pyo.Constraint(m.PIX, m.TM, rule=_ftc)

    # --- 4. the compaction potential: Pi, sigma, then the elliptic row. All v5's. ----------
    fm = float(p.f_max)
    vsig = float(p.varsigma())
    gam = float(p.gamma)
    f_ref = (None if p.f_ref_frac is None else float(p.f_ref_frac) * fm)
    eta = float(p.eta)

    if potential:
        m.Pi = pyo.Var(m.PIX, m.TM, initialize=0.0)

        def _pic(mm, q, k):
            return mm.Pi[q, k] * fm == mm.dw[q, k] * mm.ft[q, k]
        m.c_Pi = pyo.Constraint(m.PIX, m.TM, rule=_pic)

        # sigma < 1 is structural (1 - exp(-x) < 1 for any finite x). The bound is placed just
        # ABOVE 1 rather than AT it, and both halves of that matter.
        #
        # AT 1.0 it is degenerate. With f_ref_frac = 0.002 the bulk has ft/f_ref ~ 170-500, so
        # exp(-ft/f_ref) underflows to exactly 0 and sigma evaluates to exactly 1.0 -- its bound.
        # The row c_sig is then numerically "sigma = 1" (its d/d(ft) entry is ~1e-74) saying what
        # the bound says: an exact LICQ failure at npix variables, so that multiplier pair is
        # non-unique and the duals wander while the primal sits still. Invisible to the usual
        # tests -- structural rank is full because 1e-74 is nonzero, and the row is not a
        # single-variable row because it structurally has two entries.
        #
        # REMOVED ENTIRELY it is far worse, which is the other half. sigma then runs to 1.0043,
        # gamma*(1-sigma) = -0.43 against a varsigma of 0.0204, and the potential operator's
        # diagonal goes negative by 21x the screening term -- eq:xd_potential_solve stops being
        # an M-matrix at exactly the infeasible iterates the bound exists to police.
        #
        # So: above the attainable value, inside the operator's tolerance. sigma < 1 +
        # varsigma/gamma is what keeps varsigma + gamma(1-sigma) positive; a tenth of that keeps
        # 90% of the screening term while putting 1.0 strictly interior. Measured at 900
        # iterations, ma97, auto-eta, three configurations:
        #
        #   grid  K  sched   sig ub        status   reg%   inf_du end   theta err   iters
        #     16  5  sim     1.0          optimal    80%     2.51e-14       7.54%     653
        #     16  5  sim     1+2.0e-05    optimal    62%     2.62e-13       7.37%     520
        #     32  5  sim     1.0        maxIterat    95%     3.12e+02      23.13%     901
        #     32  5  sim     1+2.0e-05  maxIterat    78%     1.63e-04      11.08%     901
        #     16  8  seq     1.0        maxIterat    75%     1.64e-05       8.82%     901
        #     16  8  seq     1+2.0e-05    optimal    55%     2.66e-14       3.71%     642
        #
        # Better on every column of every row, and the last pair flips maxIterations to optimal.
        # The earlier attempt at this shipped "remove the bound" off a 60-ITERATION comparison,
        # where the ordering is the reverse of the truth; hence three configurations at 900 here.
        sig_ub = (1.0 + 0.1 * vsig / gam) if gam > 0 else None
        m.sig = pyo.Var(m.PIX, m.TM, bounds=(None, sig_ub), initialize=0.0)

        def _sigc(mm, q, k):
            if f_ref is None:
                return mm.sig[q, k] * fm == mm.ft[q, k]
            return mm.sig[q, k] == 1.0 - pyo.exp(-mm.ft[q, k] / f_ref)
        m.c_sig = pyo.Constraint(m.PIX, m.TM, rule=_sigc)

        m.phi = pyo.Var(m.PIX, m.TM, initialize=0.0)

        def _phic(mm, q, k):
            acc = (vsig + gam * (1.0 - mm.sig[q, k])) * mm.phi[q, k]
            for nb in _neighbours(q, res):
                acc -= 0.5 * (mm.sig[q, k] + mm.sig[nb, k]) * (mm.phi[nb, k] - mm.phi[q, k])
            return acc == mm.Pi[q, k]
        m.c_phi = pyo.Constraint(m.PIX, m.TM, rule=_phic)

        # --- 5. the directed rates. THE v6 BLOCK. See the module docstring for why this is
        # lifted rather than written as eta*log(1+exp(dP/eta)) -- the direct form overflows at
        # |dP| > 709.8*eta, which this model reaches.
        #
        # sp[p,q,k] is softplus(phi_q - phi_p) WITHOUT the c_cp factor, so c_cp = 0 does not
        # make the block degenerate; the amplitude enters in the mass row instead.
        faces = [(q, nb) for q in range(npix) for nb in _neighbours(q, res)]
        ufaces = [(a, b) for (a, b) in faces if a < b]
        m.FACE = pyo.Set(initialize=faces, dimen=2, ordered=True)
        m.UFACE = pyo.Set(initialize=ufaces, dimen=2, ordered=True)
        # The two equations uniquely make sp nonnegative. A zero lower bound duplicates
        # c_sp_pair when the reverse softplus rate rounds to zero; -eta keeps that valid limit
        # interior while still bounding every exponent in c_sp_pair by exp(1).
        m.sp = pyo.Var(m.FACE, m.TM, bounds=(-eta, None),
                       initialize=eta * float(np.log(2.0)))

        def _sp_diff(mm, a, b, k):
            # sp(z) - sp(-z) = z, exactly, for every eta. Linear.
            return mm.sp[a, b, k] - mm.sp[b, a, k] == mm.phi[b, k] - mm.phi[a, k]
        m.c_sp_diff = pyo.Constraint(m.UFACE, m.TM, rule=_sp_diff)

        def _sp_pair(mm, a, b, k):
            # exp(-u/eta) + exp(-v/eta) = 1. Cannot overflow: u, v >= 0.
            return pyo.exp(-mm.sp[a, b, k] / eta) + pyo.exp(-mm.sp[b, a, k] / eta) == 1.0
        m.c_sp_pair = pyo.Constraint(m.UFACE, m.TM, rule=_sp_pair)
    else:
        m.has_potential = False

    # --- 6. the implicit mass balance, eq:xd_implicit_transport ---------------------------
    # (1 + c_cp sum_q sp[p,q]) f_{k+1,p} - c_cp sum_q sp[q,p] f_{k+1,q} = ft_p.
    # A row, not a solve: f_{k+1} is already a variable. Bilinear in (sp, f).
    def _mass(mm, q, k):
        if not potential or p.c_cp == 0.0:
            return mm.f[q, k + 1] == mm.ft[q, k]
        nbs = _neighbours(q, res)
        out = mm.f[q, k + 1] * (1.0 + p.c_cp * sum(mm.sp[q, nb, k] for nb in nbs))
        out -= p.c_cp * sum(mm.sp[nb, q, k] * mm.f[nb, k + 1] for nb in nbs)
        return out == mm.ft[q, k]
    m.c_mass = pyo.Constraint(m.PIX, m.TM, rule=_mass)

    # --- 7. observation, index-shifted: y_{k+1} = C_{u_k} f_{k+1} --------------------------
    # obs_index carries the chain length too, so add_estimation_objective and the k_aug
    # param_list can be built without re-walking meas.
    m.obs_index = [(k, j, nn) for (k, j, nn) in ray_id]
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

def numpy_trajectory(theta, seq, p: V6Params, image_res: int, simultaneous: bool = False):
    """Every variable of the model, taken off a :mod:`degrade_v6` run.

    ``simultaneous`` picks :func:`degrade_v6.simulate_simultaneous` and the collapsed ray list,
    so the trajectory and the model it is pinned into describe the same schedule.
    """
    p = resolve(p, theta)
    res = int(image_res)
    npix = res * res
    meas = _measurements(seq, res, simultaneous)
    K = len(meas)
    _run = simulate_simultaneous if simultaneous else simulate
    _f, _infos, hist = _run(theta, seq, p, res, record_trajectory=True)
    f = np.stack([h.ravel() for h in hist], axis=1)

    S, Ipix, cIdelta = {}, np.zeros((npix, K)), np.zeros((npix, K))
    L = {}
    for k, (_ang, rays) in enumerate(meas):
        for j, (_r, walk) in enumerate(rays):
            acc = 0.0
            S[(k, j, 0)] = 0.0
            for t, (pix, chord, shield) in enumerate(walk):
                loc = p.I0 * np.exp(-acc)
                L[(k, j, t)] = loc
                Ipix[pix, k] += loc
                cIdelta[pix, k] += p.c * loc * chord
                acc += chord * f[shield, k]
                S[(k, j, t + 1)] = acc
    dw = 1.0 - np.exp(-cIdelta)
    ft = f[:, :K] * np.exp(-p.a * Ipix - p.b * Ipix ** 2)

    # Through compaction_potential, NOT a local spsolve: that call carries the exact M-matrix
    # bound of eq:xd_potential_bound, and a sparse direct solver handed a near-singular operator
    # returns a large finite answer rather than an error. Same reason v5's does.
    Pi = np.zeros_like(ft)
    sig = np.zeros_like(ft)
    phi = np.zeros_like(ft)
    for k in range(K):
        ph_k, pi_k, sg_k = compaction_potential(ft[:, k].reshape(res, res),
                                                dw[:, k].reshape(res, res), p)
        phi[:, k], Pi[:, k], sig[:, k] = ph_k.ravel(), pi_k.ravel(), sg_k.ravel()

    # sp on directed faces, straight off degrade_v6.softplus -- the same logaddexp the forward
    # model uses, so the gate is comparing the Pyomo ROWS against it rather than against a
    # second hand-written softplus.
    sp = {}
    for q in range(npix):
        for nb in _neighbours(q, res):
            for k in range(K):
                sp[(q, nb, k)] = float(softplus(phi[nb, k] - phi[q, k], p.eta))

    yobs = {}
    for k, (_ang, rays) in enumerate(meas):
        for j, (_r, walk) in enumerate(rays):
            acc = {}
            for _pix, chord, owner in walk:
                acc[owner] = acc.get(owner, 0.0) + chord
            yobs[(k, j)] = float(sum(c * f[q, k + 1] for q, c in acc.items()))
    return dict(f=f, S=S, L=L, Z=cIdelta, Ipix=Ipix, dw=dw, ft=ft, yobs=yobs,
                Pi=Pi, sig=sig, phi=phi, sp=sp)


def pin_model(m, traj, *, fix=True):
    """Set every variable to its numpy value, and optionally fix it there."""
    for q in m.PIX:
        for k in m.T:
            m.f[q, k].set_value(float(traj["f"][q, k]))
            if fix:
                m.f[q, k].fix()
        for k in m.TM:
            m.Ipix[q, k].set_value(float(traj["Ipix"][q, k]))
            m.Z[q, k].set_value(float(traj["Z"][q, k]))
            m.dw[q, k].set_value(float(traj["dw"][q, k]))
            m.ft[q, k].set_value(float(traj["ft"][q, k]))
            if m.has_potential:
                m.Pi[q, k].set_value(float(traj["Pi"][q, k]))
                m.sig[q, k].set_value(float(traj["sig"][q, k]))
                m.phi[q, k].set_value(float(traj["phi"][q, k]))
            if fix:
                m.Ipix[q, k].fix()
                m.Z[q, k].fix()
                m.dw[q, k].fix()
                m.ft[q, k].fix()
                if m.has_potential:
                    m.Pi[q, k].fix()
                    m.sig[q, k].fix()
                    m.phi[q, k].fix()
    if m.has_potential:
        for (a, b) in m.FACE:
            for k in m.TM:
                m.sp[a, b, k].set_value(float(traj["sp"][(a, b, k)]))
                if fix:
                    m.sp[a, b, k].fix()
    for idx in m.CH:
        m.S[idx].set_value(float(traj["S"][idx]))
        if fix:
            m.S[idx].fix()
    for idx in m.LX:
        m.L[idx].set_value(float(traj["L"][idx]))
        if fix:
            m.L[idx].fix()
    for idx in m.RAY:
        m.yobs[idx].set_value(float(traj["yobs"][idx]))
        if fix:
            m.yobs[idx].fix()


def max_residual(m, by_block: bool = False):
    """Worst constraint violation, and where.  With ``by_block``, also the worst per block."""
    worst, where = 0.0, ""
    blocks = {}
    for c in m.component_data_objects(pyo.Constraint, active=True):
        try:
            body = pyo.value(c.body)
            lo = pyo.value(c.lower) if c.lower is not None else body
            r = abs(body - lo)
        except Exception:
            continue
        name = c.parent_component().name
        if r > blocks.get(name, -1.0):
            blocks[name] = r
        if r > worst:
            worst, where = r, c.name
    return (worst, where, blocks) if by_block else (worst, where)


def check_forward(image_res: int = 24, n_steps: int = 3, verbose: bool = True,
                  simultaneous: bool = False, **kw):
    """G1, the residual gate: does the Pyomo model reproduce the numpy model?

    Pins every variable to the numpy trajectory and evaluates every constraint.  No solver, so
    this runs in the Docker build.  ``simultaneous`` gates the other schedule, which the tab
    now defaults to -- both must hold, and they are different models: K = 1 with every ray in
    one measurement, against K = n_steps with one each.  Returns the worst residual.
    """
    p = V6Params(**kw)
    seq = _demo_sequence(n_steps)
    theta = scale_to_optical_depth(_phantom(image_res), 1.1, image_res)
    traj = numpy_trajectory(theta, seq, p, image_res, simultaneous=simultaneous)
    m = build_v6_model(theta, seq, p, image_res, simultaneous=simultaneous)
    pin_model(m, traj)
    r, where, blocks = max_residual(m, by_block=True)
    if verbose:
        pr = resolve(p, theta)
        dP = max(abs(traj["phi"][b, k] - traj["phi"][a, k])
                 for (a, b, k) in traj["sp"]) if traj["sp"] else 0.0
        print("  v6 Pyomo vs numpy: grid %d, %d rows, %s, c_cp=%g, eta=%g -> %.3e (%s)  %s"
              % (image_res, n_steps, "SIMULTANEOUS (K=1)" if simultaneous else "sequential",
                 pr.c_cp, pr.eta, r, where or "-", "PASS" if r < 1e-10 else "FAIL"))
        for name in sorted(blocks):
            print("      %-12s %.3e" % (name, blocks[name]))
        # The softplus lifting's failure mode is exp(-sp/eta) losing the row its derivative in
        # sp. 745 is where exp underflows to zero outright, but that is NOT the limit that
        # matters: the derivative RATIO within the row is exp(-|dP|/eta), and it stops being
        # usefully representable around 28. An earlier version of this line quoted 745 and made
        # a ratio of 183 look like comfortable margin when it was 6.5x past the edge.
        ratio = dP / pr.eta if pr.eta else float("inf")
        note = ("ok" if ETA_RATIO_LO <= ratio <= ETA_RATIO_HI else
                "OUTSIDE the %.0f..%.0f window -- see degrade_v6.select_eta"
                % (ETA_RATIO_LO, ETA_RATIO_HI))
        print("      max |dP| = %.4g, eta = %.3g -> max |dP|/eta = %.1f  (%s)"
              % (dP, pr.eta, ratio, note))
    return r


def forward_solve(theta, seq, p: V6Params, image_res: int, *, linear_solver="ma97",
                  max_iter=3000, tol=1e-8, verbose=True, log_callback=None,
                  simultaneous: bool = False):
    """G2: fix ``f[:, 0] = theta`` and let IPOPT FIND the trajectory from the undamaged field.

    Strictly more than the residual gate.  That one says the rows are satisfied by the numpy
    answer; this one says the model is square, solvable, and scaled well enough to converge to
    that answer from a start that is not it.  It needs IPOPT, so it is not in the Docker build.
    """
    theta = np.asarray(theta, dtype=float)
    p = resolve(p, theta)
    res = int(image_res)
    m = build_v6_model(theta, seq, p, res, simultaneous=simultaneous)
    flat = theta.ravel()
    for q in m.PIX:
        m.f[q, 0].set_value(float(flat[q]))
        m.f[q, 0].fix()
    # Start every later stage at the UNDAMAGED field: a deliberately wrong guess, so converging
    # to the numpy trajectory is a result rather than a restatement of the initialisation.
    for q in m.PIX:
        for k in m.T:
            if k > 0:
                m.f[q, k].set_value(float(flat[q]))
    m.obj = pyo.Objective(expr=0.0)          # pure feasibility problem

    t0 = time.perf_counter()
    res_solve, solver_used = solve_with_fallback(m, linear_solver=linear_solver,
                                                 max_iter=max_iter, tol=tol,
                                                 log_callback=log_callback)
    wall = time.perf_counter() - t0

    got = np.array([[pyo.value(m.f[q, k]) for k in m.T] for q in m.PIX])
    traj = numpy_trajectory(theta, seq, p, res, simultaneous=simultaneous)
    err = float(np.abs(got - traj["f"]).max())
    rel = err / max(float(np.abs(traj["f"]).max()), 1e-300)
    if verbose:
        tc = res_solve.solver.termination_condition
        # The PASS bar is IPOPT's own tolerance times a constant, not a fixed number: a
        # feasibility problem solved to `tol` cannot reproduce the numpy trajectory to better
        # than roughly that, and demanding 1e-8 at tol=1e-8 fails a model that is in fact
        # correct. The constant is MEASURED rather than picked to make a run pass. Sweeping tol
        # at grid 16 / K=3, error/tol comes out:
        #     tol      sequential    simultaneous
        #     1e-06     3.7e-08        2.4e-05   (24.5x)
        #     1e-08     3.7e-08        2.3e-07   (23.5x)
        #     1e-10     1.4e-12        2.4e-11
        #     1e-12     1.4e-12        2.7e-12
        # Both fall to round-off as tol tightens, which is what says the model is right and the
        # solver is merely stopping where it was told. The simultaneous schedule runs at ~25x
        # and the sequential well under it, so 50x clears both with a factor of two in hand.
        # An earlier 20x bar failed the simultaneous case at 2.346e-07 -- widened on the
        # strength of this sweep, not to make the number green.
        bar = max(50.0 * tol, 1e-12)
        print("  forward SOLVE: grid %d, %d rows, %s, %s -> %s in %.1f s, f error %.3e abs / "
              "%.3e rel  %s (bar %.0e = 20*tol)"
              % (res, len(seq), "SIMULTANEOUS" if simultaneous else "sequential", solver_used,
                 tc, wall, err, rel, "PASS" if rel < bar else "FAIL", bar))
    return rel


def check_softplus_lifting(eta_values=(1e-1, 1e-2, 1e-3, 1e-5), verbose: bool = True):
    """The lifting is an identity, checked against ``logaddexp`` rather than against itself.

    This is the one part of the transcription that is not a line-for-line copy of an equation,
    so it gets its own gate.  Written from the algebra (``sp(z) - sp(-z) = z`` and
    ``e^{-sp(z)/eta} + e^{-sp(-z)/eta} = 1``), not from the code it checks.
    """
    zs = np.concatenate([np.linspace(-50.0, 50.0, 2001), np.array([0.0, 1e-300, -1e-300])])
    w_diff = w_pair = 0.0
    for eta in eta_values:
        u, v = softplus(zs, eta), softplus(-zs, eta)
        w_diff = max(w_diff, float(np.abs(u - v - zs).max()))
        w_pair = max(w_pair, float(np.abs(np.exp(-u / eta) + np.exp(-v / eta) - 1.0).max()))
    if verbose:
        print("  softplus lifting: max |u-v-z| = %.3e, max |e^-u/eta + e^-v/eta - 1| = %.3e  %s"
              % (w_diff, w_pair, "PASS" if max(w_diff, w_pair) < 1e-12 else "FAIL"))
    assert max(w_diff, w_pair) < 1e-12, (w_diff, w_pair)
    return w_diff, w_pair


def _cli(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image-res", type=int, default=24)
    ap.add_argument("--n-steps", type=int, default=3)
    ap.add_argument("--solve", action="store_true",
                    help="also run the forward SOLVE gate (needs IPOPT)")
    ap.add_argument("--I0", type=float, default=1.0)
    ap.add_argument("--c", type=float, default=0.1)
    ap.add_argument("--a", type=float, default=0.05)
    ap.add_argument("--b", type=float, default=0.0)
    ap.add_argument("--c-cp", type=float, default=0.3)
    ap.add_argument("--reach", type=float, default=7.0)
    ap.add_argument("--gamma", type=float, default=100.0)
    ap.add_argument("--f-ref-frac", type=float, default=0.002)
    ap.add_argument("--eta", type=float, default=None,
                    help="softplus smoothing. Omit to CHOOSE it from a forward run "
                         "(degrade_v6.select_eta); a value here overrides that.")
    ap.add_argument("--linear-solver", default="ma97")
    ap.add_argument("--max-iter", type=int, default=3000)
    a = ap.parse_args(argv)

    # The GATES take a concrete eta -- they are residual/feasibility checks and do not care how
    # well conditioned it is, so `--eta` omitted just means the historical 1e-3 for them. Only
    # run_v6_reconstruction interprets None as "choose it", because only it has a solve whose
    # conditioning depends on the answer.
    kw = dict(I0=a.I0, c=a.c, a=a.a, b=a.b, c_cp=a.c_cp, reach=a.reach, gamma=a.gamma,
              f_ref_frac=a.f_ref_frac, eta=(1e-3 if a.eta is None else a.eta))
    print(__doc__.splitlines()[0])
    print()
    check_softplus_lifting()
    print()
    r = check_forward(image_res=a.image_res, n_steps=a.n_steps, simultaneous=False, **kw)
    r_sim = check_forward(image_res=a.image_res, n_steps=a.n_steps, simultaneous=True, **kw)
    ok = max(r, r_sim) < 1e-10
    if a.solve:
        print()
        theta = scale_to_optical_depth(_phantom(a.image_res), 1.1, a.image_res)
        for sim in (False, True):
            rel = forward_solve(theta, _demo_sequence(a.n_steps), V6Params(**kw), a.image_res,
                                linear_solver=a.linear_solver, max_iter=a.max_iter,
                                simultaneous=sim)
            ok = ok and rel < 20 * 1e-8
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_cli())


# --- estimation ----------------------------------------------------------------------------
# Everything below is the inverse direction. The forward gates above must pass before any of it
# means anything, and run_v6_reconstruction re-runs the residual gate on the caller's own
# geometry rather than trusting the one in the Docker build.

def add_estimation_objective(m, y_data, tv_weight: float, theta_scale: float):
    """Fit the observations of ``eq:xd_obs_damage``, regularised by TV on ``theta``.

    ``y_data`` enters as **fixed variables, not Params**, because those are precisely the
    parameters k_aug differentiates with respect to.  Declared only over the rays actually
    fired, so there are no structurally-dead columns to prune before the sensitivity extraction.

    Both terms are normalised to O(1) before being weighed against each other.  Same reason v2
    and v5 do it: ``theta`` peaks near 0.03 here while the ray integrals are O(1) -- optical
    depth is the product -- so raw sums put the fit about 1e3 above TV and ``tv_weight`` would be
    decoration.  The fit is written on ``m.yobs``, which is step 7's own variable, so the
    objective cannot disagree with the observation row.
    """
    m.YD = pyo.Set(initialize=[(k, j) for (k, j, _n) in m.obs_index], dimen=2, ordered=True)
    m.y_data = pyo.Var(m.YD, initialize=0.0)
    for (k, j, _n) in m.obs_index:
        m.y_data[k, j].set_value(float(y_data[k][j]))
        m.y_data[k, j].fix()

    y_scale = max(float(np.max([np.max(np.abs(y)) for y in y_data if len(y)])), 1e-30)
    n_obs = max(len(m.obs_index), 1)
    n_pix = m.res * m.res
    m.fit_expression = sum((m.yobs[k, j] - m.y_data[k, j]) ** 2
                           for (k, j, _n) in m.obs_index) / (n_obs * y_scale ** 2)
    m.tv_expression = _tv_expression(m, theta_scale) / (n_pix * max(theta_scale, 1e-30))
    m.obj = pyo.Objective(expr=m.fit_expression + tv_weight * m.tv_expression)
    return m


def initialize_from_numpy(m, theta_seed=None, *, fix_theta=False):
    """Initialise every variable from a :mod:`degrade_v6` run through the SAME geometry.

    The forward and estimation models differ only in whether ``f[:,0]`` is fixed, so this is the
    same operation in both: hold the angles fixed, run the numpy model, copy the trajectory
    across.  ``theta_seed`` is what that run starts from -- the truth for a forward solve, the
    current estimate for an estimation one.  Every dynamic row then begins at residual ~0 and
    only the data-fit rows are wrong, which is what "dynamically feasible start" means here.

    Returns the worst constraint residual afterwards, which is the useful number: it *measures*
    how good the start is instead of asserting it.
    """
    theta_seed = m.theta_ref if theta_seed is None else np.asarray(theta_seed, dtype=float)
    theta_seed = np.asarray(theta_seed, dtype=float).reshape(m.res, m.res)
    traj = numpy_trajectory(theta_seed, m.seq, m.p, m.res, simultaneous=m.simultaneous)

    def _put(v, x):
        # Clipped into the variable's own bounds. The trajectory can carry f a hair below zero
        # where the transport overshoots; writing it raw would put the START POINT outside the
        # box and make IPOPT relocate it before iteration 0, throwing away the feasible start.
        lo, hi = v.lb, v.ub
        if lo is not None and x < lo:
            x = lo
        if hi is not None and x > hi:
            x = hi
        v.set_value(x)

    for q in m.PIX:
        for k in m.T:
            _put(m.f[q, k], float(traj["f"][q, k]))
        for k in m.TM:
            m.Ipix[q, k].set_value(float(traj["Ipix"][q, k]))
            _put(m.Z[q, k], float(traj["Z"][q, k]))
            m.dw[q, k].set_value(float(traj["dw"][q, k]))
            m.ft[q, k].set_value(float(traj["ft"][q, k]))
            if m.has_potential:
                m.Pi[q, k].set_value(float(traj["Pi"][q, k]))
                m.sig[q, k].set_value(float(traj["sig"][q, k]))
                m.phi[q, k].set_value(float(traj["phi"][q, k]))
    for idx in m.LX:
        _put(m.L[idx], float(traj["L"][idx]))
    if m.has_potential:
        for (a, b) in m.FACE:
            for k in m.TM:
                _put(m.sp[a, b, k], float(traj["sp"][(a, b, k)]))
    for idx in m.CH:
        m.S[idx].set_value(float(traj["S"][idx]))
    for idx in m.RAY:
        m.yobs[idx].set_value(float(traj["yobs"][idx]))
    if fix_theta:
        for q in m.PIX:
            m.f[q, 0].fix()
    return max_residual(m)[0]


@dataclass
class V6UQParams:
    """Inputs to :func:`run_v6_reconstruction`.  Physics defaults match the v6 tab's seeds."""

    image_res: int = 32
    optical_depth: float = 1.1
    beam_steps: tuple = ()             # (angle_deg, offset, n_beams), _table_to_seq form
    phantom: Optional[np.ndarray] = None
    simultaneous: bool = True          # the tab's default schedule; must match how data was taken

    # --- v6 physics (V6Params) ---
    I0: float = 1.0
    c: float = 0.1
    a: float = 0.05
    b: float = 0.0
    c_cp: float = 0.3
    reach: Optional[float] = 7.0
    gamma: float = 100.0
    f_ref_frac: Optional[float] = 0.002
    # None = CHOOSE IT from a forward run, via degrade_v6.select_eta. That is the default because
    # no fixed value works: max|dP| spans 0.15 to 80 across this model's parameter range, and eta
    # has to track it or the estimation NLP degenerates (at the old fixed 1e-3: 97% of iterations
    # Hessian-regularised, lg(mu) stuck, inf_du climbing to 6.6e9). A float here overrides it.
    eta: Optional[float] = None
    dx: float = 1.0

    # --- estimation ---
    tv_weight: float = 0.001
    noise_sigma: float = 0.0           # 0 = noiseless data, as the other tabs do
    continuation: bool = True          # seed from the I0 = 0 linear-tomography + TV solve
    gate: bool = True                  # re-run the forward residual gate on THIS geometry
    ipopt_max_iter: int = 3000
    # ma97, NOT ma27. Measured on this model, uncontended, single-threaded BLAS: one iteration
    # of the tab-default problem (grid 32, simultaneous, 5 measurements -- 25,520 vars) costs
    # 16.06 s on ma27 and 0.219 s on ma97, a factor of 73. On the sequential schedule at grid 32
    # / K=4 (59,184 vars) it is 159.0 s against 1.069 s, a factor of 149.
    #
    # The reason is that this model is ALL linear algebra: IPOPT's own timing split puts
    # 99.6% of the run in the KKT numeric factorisation and 0.04% in function evaluation. ma27
    # is a 1981 multifrontal code with no nested-dissection ordering; on this sparsity its
    # factor reaches ~1 GB and is memory-bandwidth bound. Nothing in the model formulation can
    # beat that -- three separate reformulations were built, gate-checked and timed, and the
    # best was +-9%.
    linear_solver: str = "ma97"
    obj_scaling: float = 0.0           # IPOPT obj_scaling_factor; 0 leaves it alone

    # --- UQ ---
    run_uq: bool = True
    noise_cov_scale: float = 10.0      # sigma^2 in Sigma = sigma^2 J J^T

    def to_physics(self, eta=None) -> V6Params:
        """``eta`` overrides the field, which is how the auto-selected value gets in."""
        use = eta if eta is not None else (self.eta if self.eta else 1e-3)
        return V6Params(I0=self.I0, c=self.c, a=self.a, b=self.b, c_cp=self.c_cp,
                        reach=self.reach, gamma=self.gamma, f_ref_frac=self.f_ref_frac,
                        eta=float(use), dx=self.dx)


@dataclass
class V6UQResults:
    """Arrays and scalars, not matplotlib figures -- the caller draws."""

    theta_true: np.ndarray
    theta_hat: np.ndarray
    f_final_true: np.ndarray
    f_final_hat: np.ndarray

    # --- the solve ---
    status: str = ""
    linear_solver: str = ""
    iters: str = "-"
    regularised: int = 0
    n_iter_lines: int = 0
    n_vars: int = 0
    n_cons: int = 0
    t_solve: float = float("nan")
    t_uq: float = float("nan")

    # --- how good the start was, and whether the model still means anything ---
    forward_residual: float = float("nan")   # the drift gate, on THIS geometry
    init_residual: float = float("nan")      # worst dynamic residual at the starting point
    continuation_status: str = "skipped"
    theta_rms_cont: float = float("nan")

    # --- the answer ---
    obs_rms: float = float("nan")
    theta_rms: float = float("nan")
    theta_pct_peak: float = float("nan")     # the quotable one
    n_measurements: int = 0
    n_rays: int = 0

    # --- how eta was chosen ---
    eta_used: float = float("nan")
    eta_auto: bool = False
    min_chord: float = float("nan")      # = 1/c_chain's within-row Jacobian spread
    n_chord_tiny: int = 0                # duplicated vertex crossings; see _chord_stats
    eta_ratio: float = float("nan")      # max|dP| / eta; healthy window is ~3 to 28
    eta_rest_total: float = float("nan")  # fraction of a pixel the smoothing alone moves / step
    eta_warning: str = ""

    # --- UQ, non-fatal ---
    log_cov_diag_2D: Optional[np.ndarray] = None
    d_optimality: float = float("nan")
    uq_conditioning: float = float("nan")
    uq_error: str = ""

    # --- what the damage did, so the answer reads in context ---
    mass_true: float = float("nan")
    mass_hat: float = float("nan")
    half_pct: float = float("nan")
    dP_max: float = float("nan")

    @property
    def regularised_pct(self) -> float:
        return 100.0 * self.regularised / max(self.n_iter_lines, 1)


def run_v6_reconstruction(params: V6UQParams, log_callback=None) -> V6UQResults:
    """Estimate ``theta = f_0`` from the projections, then k_aug for the pixel variance.

    The data comes from :func:`degrade_v6.simulate` / :func:`~degrade_v6.simulate_simultaneous`,
    not from a Pyomo forward solve, so the measurements and the model fitting them stay two
    independent implementations.  ``gate=True`` re-runs the residual check on the caller's actual
    geometry and **refuses to reconstruct** if it has drifted: a reconstruction against a model
    that no longer matches the simulator would read as a physics result.
    """
    def say(t):
        # A broken log sink must cost you the log, never the reconstruction. The Streamlit
        # callback raises NoSessionContext when it runs without a ScriptRunContext, and before
        # this guard that exception propagated out of run_v6_reconstruction and killed the solve
        # before it started.
        if log_callback is None:
            return
        try:
            log_callback(t)
        except Exception:
            pass
    res = int(params.image_res)
    seq = tuple(tuple(x) for x in params.beam_steps)
    if not seq:
        raise ValueError("no measurements: the sequence is empty")
    p = params.to_physics()

    theta = (np.asarray(params.phantom, dtype=float) if params.phantom is not None
             else scale_to_optical_depth(_phantom(res), params.optical_depth, res))
    theta = theta.reshape(res, res)

    # --- eta, chosen from the forward model before anything else is built --------------------
    # Done FIRST so the data, the gate, the initialisation and the NLP all carry one value: eta
    # is part of the model, not a solver option, so a mismatch between the run that generated the
    # measurements and the model fitting them would be plant-model mismatch, not tuning.
    if params.eta is None:
        eta_used, eta_info = select_eta(theta, seq, p, res, simultaneous=params.simultaneous)
        say("eta chosen from the forward run: %.4g  (max|dP| %.4g, ratio %.1f)\n"
            % (eta_used, eta_info["dP_max"], eta_info["ratio"]))
        if eta_info["warning"]:
            say("    WARNING: %s\n" % eta_info["warning"])
    else:
        eta_used = float(params.eta)
        eta_info = {"dP_max": float("nan"), "ratio": float("nan"),
                    "rest_total": float("nan"), "warning": ""}
        say("eta supplied by the caller: %.4g\n" % eta_used)
    p = resolve(params.to_physics(eta_used), theta)
    _run = simulate_simultaneous if params.simultaneous else simulate

    # --- the drift gate, on THIS geometry -------------------------------------------------
    fwd_resid = float("nan")
    if params.gate:
        say("Gate: Pyomo model against the numpy model on this geometry...\n")
        traj_g = numpy_trajectory(theta, seq, p, res, simultaneous=params.simultaneous)
        mg = build_v6_model(theta, seq, p, res, simultaneous=params.simultaneous)
        pin_model(mg, traj_g)
        fwd_resid = max_residual(mg)[0]
        say("    residual %.3e\n" % fwd_resid)
        if not (fwd_resid < 1e-10):
            raise RuntimeError(
                "forward gate FAILED on this geometry (residual %.3e): the Pyomo model and "
                "degrade_v6.simulate no longer agree, so a reconstruction against it would not "
                "be a physics result. Fix the model before reading anything below." % fwd_resid)

    # --- synthetic data -------------------------------------------------------------------
    f_true, infos_true, y_true = _run(theta, seq, p, res, record_observations=True)
    if params.noise_sigma > 0.0:
        rng = np.random.default_rng(0)
        y_true = [y + rng.normal(0.0, params.noise_sigma, size=y.shape) for y in y_true]
    n_rays = int(sum(len(y) for y in y_true))
    theta_scale = max(float(np.abs(theta).max()), 1e-30)

    # Keep the dynamically feasible trajectory intact when IPOPT initializes. Its default bound
    # push moves the many zero-density background pixels into the interior and raises inf_pr from
    # round-off to O(1) before iteration 0.
    solve_options = {
        "bound_push": 1e-10,
        "bound_frac": 1e-10,
        "bound_relax_factor": 0.0,
    }

    # --- continuation: the I0 = 0 problem is linear tomography + TV ------------------------
    theta_seed = np.full_like(theta, float(theta.mean()))
    cont_status = "skipped"
    theta_rms_cont = float("nan")
    if params.continuation:
        say("Continuation: I0 = 0 (linear tomography + TV), potential block dropped...\n")
        try:
            p0 = replace(p, I0=0.0, c_cp=0.0)
            m0 = build_v6_model(theta, seq, p0, res, f_bounds=(0.0, None), potential=False,
                                simultaneous=params.simultaneous)
            add_estimation_objective(m0, y_true, params.tv_weight, theta_scale)
            initialize_from_numpy(m0, theta_seed)
            r0, ls0 = solve_with_fallback(m0, linear_solver=params.linear_solver,
                                          max_iter=params.ipopt_max_iter,
                                          log_callback=log_callback, options=solve_options)
            cont_status = str(r0.solver.termination_condition)
            theta_seed = np.array([pyo.value(m0.f[q, 0]) for q in m0.PIX]).reshape(res, res)
            theta_rms_cont = float(np.sqrt(np.mean((theta_seed - theta) ** 2)))
            say("    %s, theta RMS %.4g (%.2f%% of peak)\n"
                % (cont_status, theta_rms_cont, 100.0 * theta_rms_cont / theta_scale))
        except Exception as exc:
            cont_status = "failed: %s" % str(exc).splitlines()[0][:120]
            say("    %s -- falling back to a flat seed\n" % cont_status)
            theta_seed = np.full_like(theta, float(theta.mean()))

    # --- the estimation NLP ----------------------------------------------------------------
    say("Building the v6 estimation model...\n")
    m = build_v6_model(theta, seq, p, res, f_bounds=(0.0, None),
                       simultaneous=params.simultaneous)
    add_estimation_objective(m, y_true, params.tv_weight, theta_scale)
    init_resid = initialize_from_numpy(m, theta_seed)
    say("    start residual %.3e (dynamically feasible; only the fit is wrong)\n" % init_resid)
    # The chord report earns its keep HERE, on the caller's real geometry: the Docker gate runs
    # K=2 (angles 0/90), where min_chord is exactly 1.0 and the defect cannot be seen at all.
    min_chord, n_chord_tiny, bad_ang = _chord_stats(seq, res)
    if n_chord_tiny:
        say("    WARNING: %d chords below 1e-9 (min %.4e) at angles %s. line_grid_intersections "
            "emits an exact-vertex crossing twice there, which inflates Ipix on those pixels.\n"
            % (n_chord_tiny, min_chord, bad_ang))
    n_v = int(m.nvariables())
    n_c = int(m.nconstraints())

    opts = dict(solve_options)
    if params.obj_scaling > 0.0:
        opts["obj_scaling_factor"] = float(params.obj_scaling)
    buf = []

    def _tee(t):
        # Must NOT raise. _solve_streaming latches its forwarding off after one exception, so a
        # caller's callback blowing up here would stop `buf` filling too -- and `buf` is what
        # _reg_fraction counts, so the Hessian-regularisation figure would silently be computed
        # from a truncated log rather than reported as unavailable.
        buf.append(t)
        if log_callback:
            try:
                log_callback(t)
            except Exception:
                pass
    t0 = time.perf_counter()
    r, ls = solve_with_fallback(m, linear_solver=params.linear_solver,
                                max_iter=params.ipopt_max_iter, log_callback=_tee,
                                options=opts)
    t_solve = time.perf_counter() - t0
    reg, nlines = _reg_fraction("".join(buf))

    theta_hat = np.array([pyo.value(m.f[q, 0]) for q in m.PIX]).reshape(res, res)
    f_hat, infos_hat = _run(theta_hat, seq, p, res)
    resid = np.concatenate([np.asarray([pyo.value(m.yobs[k, j]) for (k, j, _n) in m.obs_index])
                            - np.asarray([pyo.value(m.y_data[k, j])
                                          for (k, j, _n) in m.obs_index])])
    theta_rms = float(np.sqrt(np.mean((theta_hat - theta) ** 2)))

    out = V6UQResults(
        theta_true=theta, theta_hat=theta_hat, f_final_true=f_true, f_final_hat=f_hat,
        status=str(r.solver.termination_condition), linear_solver=ls,
        iters=str(getattr(r.solver, "iterations", "-")),
        regularised=reg, n_iter_lines=nlines, n_vars=n_v, n_cons=n_c, t_solve=t_solve,
        forward_residual=fwd_resid, init_residual=init_resid,
        continuation_status=cont_status, theta_rms_cont=theta_rms_cont,
        obs_rms=float(np.sqrt(np.mean(np.square(resid)))),
        theta_rms=theta_rms, theta_pct_peak=100.0 * theta_rms / theta_scale,
        n_measurements=len(seq), n_rays=n_rays,
        eta_used=eta_used, eta_auto=(params.eta is None),
        min_chord=min_chord, n_chord_tiny=n_chord_tiny,
        eta_ratio=float(eta_info.get("ratio", float("nan"))),
        eta_rest_total=float(eta_info.get("rest_total", float("nan"))),
        eta_warning=str(eta_info.get("warning", "")),
        mass_true=float(f_true.sum()), mass_hat=float(f_hat.sum()),
        half_pct=float(shape_diagnostics(theta, f_true)[1]),
        dP_max=max((i.dP_max for i in infos_true), default=float("nan")))
    say("Reconstruction: %s, theta RMS %.4g (%.2f%% of peak)\n"
        % (out.status, out.theta_rms, out.theta_pct_peak))

    # --- k_aug: d(theta)/d(y), eq:xd_composed_jacobian --------------------------------------
    if params.run_uq:
        t1 = time.perf_counter()
        try:
            from senDOE.helpers.statistics import d_optimality
            from senDOE.sensitivity.pyomo_sensitivity import extract_sensitivity_matrix
            say("Extracting d(theta)/d(y) with k_aug...\n")
            J = np.asarray(extract_sensitivity_matrix(
                model=m,
                var_list=[m.f[q, 0] for q in m.PIX],
                param_list=[m.y_data[k, j] for (k, j, _n) in m.obs_index],
                mode="k_aug", return_type="dense"), dtype=float)
            if not np.all(np.isfinite(J)):
                raise ValueError("k_aug returned a non-finite sensitivity matrix")
            cov = params.noise_cov_scale * (J @ J.T)
            out.uq_conditioning = (float(np.linalg.cond(cov)) if cov.shape[0] <= 2048
                                   else float("nan"))
            with np.errstate(divide="ignore", invalid="ignore"):
                out.log_cov_diag_2D = np.log10(np.diag(cov)).reshape(res, res)
            out.d_optimality = float(d_optimality(cov))
            say("    D-optimality %.6g\n" % out.d_optimality)
        except Exception as exc:
            # NON-FATAL BY DESIGN. The covariance is intentionally rank deficient -- a starved
            # geometry leaves pixels no ray constrains -- so k_aug can legitimately come back
            # singular, and losing the covariance must not lose the reconstruction. Same call
            # the 3D slice loop and the v2 driver make.
            out.uq_error = "%s: %s" % (type(exc).__name__, str(exc).splitlines()[0][:200])
            say("    UQ failed (reconstruction kept): %s\n" % out.uq_error)
        out.t_uq = time.perf_counter() - t1
    return out


# --- G3: the scaling gate ---------------------------------------------------------------------

def _sp_and_chord_extra(m, p):
    """v6-specific rows for :func:`degrade_v2_uq.scaling_report`, as an ``extra`` callback."""
    def _extra(nlp, J, cons, varz):
        eta = float(p.eta)
        d = {"eta": eta, "f_max": float(p.f_max),
             "f_ref": (None if p.f_ref_frac is None else float(p.f_ref_frac) * float(p.f_max))}
        if not getattr(m, "has_potential", False):
            return d
        # c_sp_pair is exp(-u/eta) + exp(-v/eta) = 1, so its two derivative MAGNITUDES are
        # exp(-u/eta)/eta and exp(-v/eta)/eta and they sum to exactly 1/eta. Two exact
        # consequences follow, and they are what S7 and S8 assert:
        #   the larger lies in [1/(2 eta), 1/eta];
        #   their ratio is exp(|u - v|/eta), and |u - v| is |dP| across that face, by c_sp_diff.
        idx = [i for i, c in enumerate(cons) if c.parent_component().name == "c_sp_pair"]
        lo_b, hi_b = 1.0 / (2.0 * eta), 1.0 / eta
        mx, mn, zero, denorm, ident, dP = [], [], 0, 0, 0.0, 0.0
        for i in idx:
            v = np.abs(J.data[J.indptr[i]:J.indptr[i + 1]])
            if v.size < 2:
                continue
            a, b = float(v.max()), float(v.min())
            mx.append(a); mn.append(b)
            if b == 0.0:
                zero += 1
                continue
            if b < 2.2250738585072014e-308:      # subnormal: the identity degrades to ~1e-6 there
                denorm += 1
                continue
            # log(a) - log(b), NOT log(a/b): the ratio overflows to inf past exp(709) and the
            # check then reports inf on a perfectly healthy identity.
            gap = (np.log(a) - np.log(b)) * eta
            dP = max(dP, gap)
            ident = max(ident, abs(gap - gap))   # placeholder; exactness checked against c_sp_diff
        d.update(sp_row_min=(min(mx) if mx else float("nan")),
                 sp_row_max=(max(mx) if mx else float("nan")),
                 sp_row_lo=lo_b, sp_row_hi=hi_b,
                 sp_zero_deriv_rows=zero, sp_denormal_rows=denorm,
                 dP_max_eval=dP, eta_ratio_eval=(dP / eta if eta else float("nan")))
        # The c_sp_diff identity, measured directly rather than inferred: |u - v| must equal |dP|.
        err = 0.0
        for (a_, b_) in m.UFACE:
            for k in m.TM:
                u, v2 = pyo.value(m.sp[a_, b_, k]), pyo.value(m.sp[b_, a_, k])
                z = pyo.value(m.phi[b_, k]) - pyo.value(m.phi[a_, k])
                err = max(err, abs((u - v2) - z))
        d["sp_identity_err"] = float(err)
        return d
    return _extra


def _chord_stats(seq, image_res: int, simultaneous: bool = False):
    """Smallest chord in the geometry, and the count of degenerate ones.

    ``c_chain``'s within-row spread is exactly ``1/min_chord``, so this is a scaling number as
    well as a geometry one.  Measured: ``min_chord`` is ``2.5640e-16`` -- identically, to every
    printed digit, at grid 12, 16 AND 32 -- for the angle sets containing 60 and 120 degrees
    (K = 3 and K = 9), and 1.0 / 14.1 / 160-263 at K = 2 / 4 / 5. A grazing ray's chord scales
    with the grid; a fixed round-off does not, so this is ``line_grid_intersections`` emitting an
    exact-vertex crossing TWICE, not a ray clipping a corner. 13 decades separate it from the
    smallest legitimate chord found (3.80e-3), so the two are not confusable.

    NOT fixed here. The fix belongs in ``dose_response.ray_geometry``, which v1's Pyomo path does
    not use, so correcting it would widen the convention split CLAUDE.md already records for
    chord attribution. Reported so it is visible instead of silent.
    """
    # ALWAYS the uncollapsed list, whatever the schedule. The rays are identical either way --
    # only the grouping differs -- but the simultaneous form merges every angle into one entry
    # whose nominal angle is the first row's, so attributing a degenerate chord to an angle off
    # that list reports 0.0 for all of them. Measured: the true culprits are 60 and 120 degrees.
    meas = _measurements(seq, int(image_res), False)
    ch = np.concatenate([np.asarray([c for _pix, c, _o in walk], dtype=float)
                         for _ang, rays in meas for _r, walk in rays if walk]) \
        if any(walk for _a, rays in meas for _r, walk in rays) else np.array([1.0])
    bad = ch < 1e-9
    ang = sorted({round(float(np.rad2deg(a)), 6) for a, rays in meas
                  for _r, walk in rays if any(c < 1e-9 for _p, c, _o in walk)})
    return float(ch.min()), int(bad.sum()), ang


def check_scaling(m=None, *, image_res: int = 16, n_steps: int = 2, simultaneous: bool = True,
                  seed: str = "flat", verbose: bool = True, **kw) -> dict:
    """G3: is the estimation NLP sanely scaled?  Returns the report; never raises.

    Ten assertions, each with a bound justified by a measured range -- see the table in the
    source.  Everything that legitimately varies is REPORTED instead, because a checker that
    asserts an unmeasured threshold fires spuriously, gets muted, and then misses the real thing.

    Deliberately NOT asserted: ``max|dP|/eta`` inside ``[ETA_RATIO_LO, ETA_RATIO_HI]``. That
    window governs the forward run ``select_eta`` probes, where the ratio is the target BY
    CONSTRUCTION. At the evaluation point it is a different number and legitimately below the
    window -- measured 1.54 (grid 16, K=5, simultaneous) and 2.09 (grid 16, K=2, sequential) on
    healthy models. Both numbers are reported; neither is a gate.

    Only ONE of these has ever fired on real code: S9, which counts exactly-zero derivatives in
    ``c_sp_pair`` and saw 0..43 of them at the old fixed ``eta = 1e-3``. S7 and S8 are exact by
    construction. S1-S6 are tripwires of unmeasured sensitivity -- they have never been observed
    to fail, which is a reason to keep their bounds loose, not a reason to trust them.
    """
    from degrade_v2_uq import scaling_report
    p_used = None
    if m is None:
        p0 = V6Params(**kw) if kw else V6Params()
        theta = scale_to_optical_depth(_phantom(image_res), 1.1, image_res)
        seq = _demo_sequence(n_steps)
        eta, _info = select_eta(theta, seq, p0, image_res, simultaneous=simultaneous)
        p_used = resolve(replace(p0, eta=eta), theta)
        run = simulate_simultaneous if simultaneous else simulate
        _f, _i, y = run(theta, seq, p_used, image_res, record_observations=True)
        m = build_v6_model(theta, seq, p_used, image_res, f_bounds=(0.0, None),
                           simultaneous=simultaneous)
        add_estimation_objective(m, y, 0.001, float(np.abs(theta).max()))
        start = theta if seed == "truth" else np.full_like(theta, float(theta.mean()))
        resid = initialize_from_numpy(m, start)
    else:
        p_used, seq, resid, seed = m.p, m.seq, max_residual(m)[0], "given"
        simultaneous = bool(getattr(m, "simultaneous", False))

    rep = scaling_report(m, extra=_sp_and_chord_extra(m, p_used))
    if rep.get("skipped"):
        if verbose:
            print("  v6 scaling: SKIPPED (%s)" % rep["skipped"])
        return rep
    rep["residual"], rep["seed"] = float(resid), seed
    mc, nct, bad_ang = _chord_stats(seq, m.res)
    rep.update(min_chord=mc, n_chord_tiny=nct, degenerate_angles=bad_ang)

    f = []
    fm = float(p_used.f_max)
    #  #   what                                    measured range                      bound
    #  S1  structurally empty rows                 0 in 143 configs                    == 0
    #  S2  numerically zero rows                   0 in 143                            == 0
    #  S3  dead variable columns                   0 in 143                            == 0
    #  S4  objective gradient non-empty            nnz 24..972                         > 0
    #  S5  median row inf-norm                     EXACTLY 1.0 in 143, and at iters
    #                                              0/5/20/60 of a real solve; 62-72%
    #                                              of rows are identically 1, so the
    #                                              median is 1 by majority            0.5..2
    #  S6  min row inf-norm / min(1, f_max)        1.000000 in 11/11, block always
    #                                              c_Pi. Half structural (fm is a
    #                                              constant entry in c_Pi, so that
    #                                              block is >= fm by construction) and
    #                                              half empirical (that NO other block
    #                                              goes lower)                        >= 0.9
    #  S7  c_sp_pair row norm in [1/(2eta),1/eta]  violation 0.0 in 6/6      exact by construction
    #  S8  c_sp_diff identity |u-v| == |dP|        5.1e-16..8.9e-16 over 9 configs     < 1e-9
    #  S9  c_sp_pair exactly-zero derivatives      0 at auto-eta; 0..43 at eta=1e-3    == 0
    #  S10 start residual                          1.6e-15..1.4e-14                    < 1e-10
    if rep["empty_rows"]:       f.append("S1 %d structurally empty rows" % rep["empty_rows"])
    if rep["zero_rows"]:        f.append("S2 %d numerically zero rows" % rep["zero_rows"])
    if rep["dead_cols"]:        f.append("S3 %d dead variable columns" % rep["dead_cols"])
    if rep["grad_nnz"] == 0:    f.append("S4 objective gradient is structurally empty")
    if not (0.5 <= rep["row_med"] <= 2.0):
        f.append("S5 median row norm %.3g outside [0.5, 2]" % rep["row_med"])
    if rep["row_min"] < 0.9 * min(1.0, fm):
        f.append("S6 min row norm %.3g below 0.9*min(1,f_max) = %.3g (%s)"
                 % (rep["row_min"], 0.9 * min(1.0, fm), rep["row_min_block"]))
    if getattr(m, "has_potential", False):
        if not (rep["sp_row_lo"] * (1 - 1e-9) <= rep["sp_row_min"]
                and rep["sp_row_max"] <= rep["sp_row_hi"] * (1 + 1e-9)):
            f.append("S7 c_sp_pair row norms %.4g..%.4g outside [1/(2eta), 1/eta] = %.4g..%.4g"
                     % (rep["sp_row_min"], rep["sp_row_max"], rep["sp_row_lo"], rep["sp_row_hi"]))
        if rep["sp_identity_err"] > 1e-9:
            f.append("S8 c_sp_diff identity off by %.3e" % rep["sp_identity_err"])
        if rep["sp_zero_deriv_rows"]:
            f.append("S9 %d c_sp_pair rows have an exactly-zero derivative (eta too small for "
                     "this geometry -- see degrade_v6.select_eta)" % rep["sp_zero_deriv_rows"])
    if rep["residual"] >= 1e-10:
        f.append("S10 start residual %.3e" % rep["residual"])
    rep["failures"], rep["ok"] = f, not f

    if verbose:
        print("  v6 scaling (grid %d, %s, seed %s): %s"
              % (m.res, "simultaneous" if simultaneous else "sequential", rep["seed"],
                 "PASS" if rep["ok"] else "FAIL " + "; ".join(f)))
        print("      rows med %.3g  min %.3g (%s)  max %.3g (%s)  |  %.0f%% are exactly 1"
              % (rep["row_med"], rep["row_min"], rep["row_min_block"],
                 rep["row_max"], rep["row_max_block"], 100 * rep["frac_row_one"]))
        print("      |grad f| %.3g -> %.2e x row_max;  IPOPT's own factor min(1,100/|g|) = %.3g"
              % (rep["grad_inf"], rep["grad_over_row_max"], rep["ipopt_df"]))
        if getattr(m, "has_potential", False):
            print("      c_sp_pair rows %.4g..%.4g in [%.4g, %.4g] | zero-deriv %d | denormal %d"
                  " | identity %.1e" % (rep["sp_row_min"], rep["sp_row_max"], rep["sp_row_lo"],
                                        rep["sp_row_hi"], rep["sp_zero_deriv_rows"],
                                        rep["sp_denormal_rows"], rep["sp_identity_err"]))
            print("      eta %.4g | max|dP| at the EVAL point %.4g -> ratio %.2f (the %g..%g "
                  "window governs select_eta's forward run, NOT this point)"
                  % (rep["eta"], rep["dP_max_eval"], rep["eta_ratio_eval"],
                     ETA_RATIO_LO, ETA_RATIO_HI))
        if rep["n_chord_tiny"]:
            print("      WARNING %d chords below 1e-9 (min %.4e) at angles %s -- duplicated "
                  "vertex crossings, see _chord_stats" % (rep["n_chord_tiny"], rep["min_chord"],
                                                          rep["degenerate_angles"]))
    return rep
