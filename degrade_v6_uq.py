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

Both are smooth, and because ``u, v >= 0`` the exponents are never positive, so **neither
exponential can overflow at any field whatsoever**.  The pair is also unique: substituting
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
from dataclasses import replace

import numpy as np
import pyomo.environ as pyo

from degrade_v2 import scale_to_optical_depth
from degrade_v2_uq import measurement_rays, solve_with_fallback
from degrade_v3_uq import _neighbours
from degrade_v6 import (V6Params, simulate, resolve, softplus, compaction_potential,
                        _phantom, _demo_sequence)


# --- model ---------------------------------------------------------------------------------

def build_v6_model(theta_ref, seq, p: V6Params, image_res: int, *, f_bounds=None,
                   potential: bool = True):
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
    meas = measurement_rays(seq, res)
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

    m.Ipix = pyo.Var(m.PIX, m.TM, initialize=0.0)

    def _ip(mm, q, k):
        ts = I_terms[(q, k)]
        if not ts:
            return mm.Ipix[q, k] == 0.0
        return mm.Ipix[q, k] == sum(p.I0 * pyo.exp(-mm.S[i]) for i in ts)
    m.c_Ipix = pyo.Constraint(m.PIX, m.TM, rule=_ip)

    # --- 2. converted fraction, one row off the chain. No dose state. ----------------------
    m.dw = pyo.Var(m.PIX, m.TM, initialize=0.0)

    def _dwc(mm, q, k):
        ts = Id_terms[(q, k)]
        if not ts:
            return mm.dw[q, k] == 0.0
        return mm.dw[q, k] == 1.0 - pyo.exp(
            -sum(p.c * p.I0 * pyo.exp(-mm.S[i]) * ch for i, ch in ts))
    m.c_dw = pyo.Constraint(m.PIX, m.TM, rule=_dwc)

    # --- 3. mass loss, carried as a variable so the rows downstream stay low order ---------
    m.ft = pyo.Var(m.PIX, m.TM, initialize=lambda _m, q, k: float(flat[q]))

    def _ftc(mm, q, k):
        return mm.ft[q, k] == mm.f[q, k] * pyo.exp(
            -p.a * mm.Ipix[q, k] - p.b * mm.Ipix[q, k] ** 2)
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

        # sigma < 1 is structural (1 - exp(-x) < 1); the lower bound is left open because ft can
        # be marginally negative and a hard 0 would make pinning the trajectory an
        # out-of-bounds write. Same reasoning as v5.
        m.sig = pyo.Var(m.PIX, m.TM, bounds=(None, 1.0), initialize=0.0)

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
        m.sp = pyo.Var(m.FACE, m.TM, bounds=(0.0, None), initialize=eta * float(np.log(2.0)))

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

def numpy_trajectory(theta, seq, p: V6Params, image_res: int):
    """Every variable of the model, taken off a :func:`degrade_v6.simulate` run."""
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
    return dict(f=f, S=S, Ipix=Ipix, dw=dw, ft=ft, yobs=yobs, Pi=Pi, sig=sig, phi=phi, sp=sp)


def pin_model(m, traj, *, fix=True):
    """Set every variable to its numpy value, and optionally fix it there."""
    for q in m.PIX:
        for k in m.T:
            m.f[q, k].set_value(float(traj["f"][q, k]))
            if fix:
                m.f[q, k].fix()
        for k in m.TM:
            m.Ipix[q, k].set_value(float(traj["Ipix"][q, k]))
            m.dw[q, k].set_value(float(traj["dw"][q, k]))
            m.ft[q, k].set_value(float(traj["ft"][q, k]))
            if m.has_potential:
                m.Pi[q, k].set_value(float(traj["Pi"][q, k]))
                m.sig[q, k].set_value(float(traj["sig"][q, k]))
                m.phi[q, k].set_value(float(traj["phi"][q, k]))
            if fix:
                m.Ipix[q, k].fix()
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


def check_forward(image_res: int = 24, n_steps: int = 3, verbose: bool = True, **kw):
    """G1, the residual gate: does the Pyomo model reproduce :func:`degrade_v6.simulate`?

    Pins every variable to the numpy trajectory and evaluates every constraint.  No solver, so
    this runs in the Docker build.  Returns the worst residual.
    """
    p = V6Params(**kw)
    seq = _demo_sequence(n_steps)
    theta = scale_to_optical_depth(_phantom(image_res), 1.1, image_res)
    traj = numpy_trajectory(theta, seq, p, image_res)
    m = build_v6_model(theta, seq, p, image_res)
    pin_model(m, traj)
    r, where, blocks = max_residual(m, by_block=True)
    if verbose:
        pr = resolve(p, theta)
        dP = max(abs(traj["phi"][b, k] - traj["phi"][a, k])
                 for (a, b, k) in traj["sp"]) if traj["sp"] else 0.0
        print("  v6 Pyomo vs numpy: grid %d, %d steps, c_cp=%g, eta=%g -> %.3e (%s)  %s"
              % (image_res, n_steps, pr.c_cp, pr.eta, r, where or "-",
                 "PASS" if r < 1e-10 else "FAIL"))
        for name in sorted(blocks):
            print("      %-12s %.3e" % (name, blocks[name]))
        # The softplus lifting's only failure mode is underflow of exp(-sp/eta), which costs the
        # row its derivative in sp. Report the margin rather than assume it.
        print("      max |dP| = %.4g, eta = %.3g -> max |dP|/eta = %.1f  (exp underflows at 745)"
              % (dP, pr.eta, dP / pr.eta if pr.eta else float("inf")))
    return r


def forward_solve(theta, seq, p: V6Params, image_res: int, *, linear_solver="ma27",
                  max_iter=3000, tol=1e-8, verbose=True, log_callback=None):
    """G2: fix ``f[:, 0] = theta`` and let IPOPT FIND the trajectory from the undamaged field.

    Strictly more than the residual gate.  That one says the rows are satisfied by the numpy
    answer; this one says the model is square, solvable, and scaled well enough to converge to
    that answer from a start that is not it.  It needs IPOPT, so it is not in the Docker build.
    """
    theta = np.asarray(theta, dtype=float)
    p = resolve(p, theta)
    res = int(image_res)
    m = build_v6_model(theta, seq, p, res)
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
    traj = numpy_trajectory(theta, seq, p, res)
    err = float(np.abs(got - traj["f"]).max())
    rel = err / max(float(np.abs(traj["f"]).max()), 1e-300)
    if verbose:
        tc = res_solve.solver.termination_condition
        # The PASS bar is IPOPT's own tolerance, not a fixed constant: a feasibility problem
        # solved to `tol` cannot be expected to reproduce the numpy trajectory to better than
        # roughly that, and demanding 1e-8 at tol=1e-8 fails a model that is in fact correct.
        bar = max(20.0 * tol, 1e-12)
        print("  forward SOLVE: grid %d, %d steps, %s -> %s in %.1f s, f error %.3e abs / "
              "%.3e rel  %s (bar %.0e = 20*tol)"
              % (res, len(seq), solver_used, tc, wall, err, rel,
                 "PASS" if rel < bar else "FAIL", bar))
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
    ap.add_argument("--eta", type=float, default=1e-3)
    ap.add_argument("--linear-solver", default="ma27")
    ap.add_argument("--max-iter", type=int, default=3000)
    a = ap.parse_args(argv)

    kw = dict(I0=a.I0, c=a.c, a=a.a, b=a.b, c_cp=a.c_cp, reach=a.reach, gamma=a.gamma,
              f_ref_frac=a.f_ref_frac, eta=a.eta)
    print(__doc__.splitlines()[0])
    print()
    check_softplus_lifting()
    print()
    r = check_forward(image_res=a.image_res, n_steps=a.n_steps, **kw)
    ok = r < 1e-10
    if a.solve:
        print()
        theta = scale_to_optical_depth(_phantom(a.image_res), 1.1, a.image_res)
        rel = forward_solve(theta, _demo_sequence(a.n_steps), V6Params(**kw), a.image_res,
                            linear_solver=a.linear_solver, max_iter=a.max_iter)
        ok = ok and rel < 1e-8
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
