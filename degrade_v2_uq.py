"""Pyomo transcription of the v2 damage model, and reconstruction of ``theta = f_0`` from it.

:mod:`degrade_v2` is the forward simulator: numpy, fast, and validated against the manuscript
by ``check_invariants`` / ``check_reference_numbers``.  This module writes the *same* ten steps
as algebraic constraints so IPOPT can run them backwards -- estimate the undamaged reference
field from the projections -- and so k_aug can differentiate the result, which is
eq:xd_composed_jacobian.

It is the v2 counterpart of :mod:`tomography_uq`, and deliberately not a modification of it:
``senDOE/`` is a verbatim vendored snapshot (see ``SENDOE_VENDOR.md``) and v1's Pyomo model
carries only ``image[ix, iy, time]``, where v2 needs a dose state, a displacement field and a
mass balance as well.

What makes this checkable
-------------------------
The measured data comes from the numpy simulator, not from a Pyomo forward solve, so the two
implementations stay independent and can be compared.  :func:`check_forward` does exactly that,
at two levels:

* **residual** -- pin every variable to a ``degrade_v2.simulate`` trajectory and evaluate every
  constraint body.  Catches any transcription error at the true solution, needs no solver.
* **forward solve** -- fix ``f[:, 0] = theta``, start IPOPT somewhere else, and let it find the
  trajectory.  Compare against numpy.  This additionally proves the model is square, solvable
  and well enough scaled to converge, which the residual check cannot.

Both must pass before a reconstruction means anything.

Transcription notes, where a choice had to be made
--------------------------------------------------
* **The observation costs nothing.**  Step 1 needs a running sum of ``chord * f`` along each ray
  in travel order; its final value *is* the ray integral of eq:xd_obs_damage (checked: 7.1e-15
  over 17 angles and a full fan).  So ``y_k`` is the last element of the shielding chain rather
  than a variable of its own, and the observation cannot disagree with the photon balance.
* **``K`` and ``B`` are constants, from an explicit reference density.**  Taken literally,
  ``ElasticSolver(theta, ...)`` makes the stiffness a function of the estimation target and
  eq:xd_elastic_discrete bilinear.  The spec's own variable table lists ``K, B`` as operators of
  "grid, E, nu, BCs" -- state independent -- and (S2) has ``K`` "assembled once on the initial
  support".  The ersatz density is a device so the free surface needs no explicit meshing, not a
  dependence on the unknown *values*.  So the caller supplies ``reference_density`` and it enters
  as a constant, which also keeps (D2): the elasticity block stays sparse linear rows.  This
  idealises the stiffness skeleton as known.  It is the same convention v1 uses -- there the
  forward and inverse solves are literally one Pyomo object with ``image[:, :, 0]`` freed -- and
  the manuscript's note that the modulus multiplies the eigenstrain load as well as the
  stiffness ("for a uniform modulus the two cancel exactly, and for a varying one the
  sensitivity is second order") is why it is cheap rather than a fudge.
* **``dw`` and ``Ipix`` are variables, not expressions.**  Holding them as variables is what
  keeps ``K u = B dw`` *linear*, which is the whole content of (D2); folded in as expressions
  every elasticity row would carry four ``exp``s and the block would stop being linear.
* **``H = I`` is half of a pair, and only the forward-model half of the Jacobian is claimed.**
  eq:xd_obs_damage is ``H(Qbar_k) C^loc f_k``; the spec sanctions ``H = I`` but as a *variant*
  that inflates the noise covariance instead, ``Sigma_eps -> Sigma_eps(Qbar_k)``.  This model
  takes ``H = I`` with a CONSTANT ``Sigma_eps``, so it has not taken the variant -- it has
  dropped the operator.  What survives is the state dependence of the Fisher information, since
  ``Phi_k`` is dose-driven through steps 1-10, so the pivot is intact; what is lost is the
  resolution-loss channel.  So what may be claimed from this is the forward-model half of
  eq:xd_composed_jacobian, not the composed Jacobian.
* **``eps_up`` must be the relative form.**  ``sqrt(v^2)`` has no derivative at ``v = 0``, so the
  tab's default ``eps_up = 0`` cannot be handed to a solver.  ``eps_rel`` scales the smoothing to
  the flow, which is differentiable *and* leaves the collapse and the ``I0 = 0`` identity exact
  -- unlike a constant ``eps_up``, which leaks ~3e-7 into both (see
  ``degrade_v2.check_invariants``).  It is carried squared, as one variable per step, because
  the ``sqrt`` of a sum over the whole grid inside all ~8k face splits would otherwise be one
  enormous shared subexpression.
* **``c_cp = 0`` drops the transport block entirely.**  Not an optimisation: it is the spec's own
  off switch ("the eigenstrain vanishes, so Delta x = 0 and every flux with it"), and taking it
  literally avoids handing the solver ``sqrt(0)``, whose derivative does not exist, at every
  face at once.
* **The travel-order walk is mirrored exactly, including a defect.**  Travelling against the
  vendored ascending-(x, y) crossing order, ``accumulate_dose`` deposits dose into the pixel at
  crossing ``i`` while the chord it uses, and the shielding increment it then adds, belong to the
  pixel at crossing ``i-1``.  So on those rays **each pixel is shielded by its own chord**.  This
  is a violation of a written equation, not a filled-in gap: eq:xd_dose_state is
  ``Q_{k+1,p} = Q_{k,p} + c_q I_p delta_p`` -- deposit pixel and chord owner are the same symbol
  ``p``, in one line -- and eq:xd_local_intensity gives ``f_{p_m}`` and ``delta_{p_m}`` the same
  subscript.  The fix is one index (deposit into ``rows[s]``, not ``rows[i]``); the walk order is
  already correct and must not be touched.  Measured by :func:`check_photon_balance`: forward
  rays 0.0, antiparallel rays 1.0, which is exactly ``I0``.
  It is reproduced rather than corrected because the decision is not this module's to take --
  ``dose_response.degradation_dose_response`` shares the split, so it moves the 2D live picture
  and the 3D simulator, and because ``degrade_v2`` turns out to BE the "independent reproduction"
  quoted at the note's line 609 (it returns -0.93 / -3.63 / -7.95 for Rg, the printed digits
  exactly), the fix also edits three numbers already written into section 3.4.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyomo.environ as pyo

from dose_response import bundle_r_values, ray_geometry
from degrade_v2 import ElasticSolver, V2Params, scale_to_optical_depth, simulate


# --- geometry ------------------------------------------------------------------------------

def ray_walk(r: float, angle_rad: float, res: int):
    """``accumulate_dose``'s travel-order walk, as a list of records.

    One record per crossing that actually deposits dose:
    ``(dose_pixel, chord, shield_pixel)``, with pixels flattened to ``row * res + col``.
    The shielding after record ``t`` is ``sum_{s <= t} chord_s * f[shield_pixel_s]``, and its
    final value is the ray integral -- step 11.

    Returns ``None`` if the ray never enters the grid, matching ``ray_geometry``.
    """
    g = ray_geometry(float(r), float(angle_rad), int(res), int(res))
    if g is None:
        return None
    rows, cols, seg, forward = g
    n, n_seg = len(rows), len(seg)
    walk = []
    for i in (range(n) if forward else range(n - 1, -1, -1)):
        s = i if forward else i - 1          # chord traversed on leaving crossing i
        if 0 <= s < n_seg:
            # Deposit pixel and shielding pixel are BOTH the chord's owner, rows[s]: that is
            # eq:xd_dose_state, where they are the same symbol p. They are kept as separate
            # fields only because the constraint builder reads them separately.
            dst = i if forward else s
            walk.append((int(rows[dst]) * res + int(cols[dst]),    # receives the dose
                         float(seg[s]),                            # chord for that dose
                         int(rows[s]) * res + int(cols[s])))       # shields the rest of the ray
    return walk or None


def measurement_rays(seq, image_res: int):
    """``[(angle_rad, [(r, walk), ...]), ...]`` -- one entry per measurement, rays that hit."""
    out = []
    for angle_deg, offset, n_beams in seq:
        ang = float(np.deg2rad(float(angle_deg)))
        rays = []
        for r in bundle_r_values(float(offset), int(n_beams), int(image_res)):
            w = ray_walk(r, ang, image_res)
            if w is not None:
                rays.append((float(r), w))
        out.append((ang, rays))
    return out


# --- model ---------------------------------------------------------------------------------

class NonDifferentiableModel(ValueError):
    """Raised when the requested settings would hand IPOPT a derivative that does not exist."""


def build_v2_model(theta_ref, seq, p: V2Params, image_res: int, *,
                   reference_density=None, f_bounds=None, freeze_mechanics=False,
                   frozen_velocity=None, allow_nondifferentiable=False):
    """Steps 1-11 of section 3.2 as a Pyomo model.  ``theta_ref`` seeds every variable.

    ``reference_density`` is what ``K``/``B`` are assembled from (see the module docstring);
    it defaults to ``theta_ref``.  ``f_bounds`` applies eq:xd_box -- leave it ``None`` for a
    forward check, where the numpy trajectory may legitimately overshoot below zero and a bound
    would hide the disagreement rather than reveal it.

    ``freeze_mechanics`` replaces the elasticity block with velocities supplied in
    ``frozen_velocity`` (a list of ``(cx, cy)`` per step): the manuscript's "frozen-transport
    approximation", kept as the escape hatch and as its own open item 6.
    """
    theta_ref = np.asarray(theta_ref, dtype=float)
    res = int(image_res)
    npix = res * res
    meas = measurement_rays(seq, res)
    K = len(meas)
    if K == 0:
        raise ValueError("no measurements: the sequence is empty")

    dens = theta_ref if reference_density is None else np.asarray(reference_density, float)
    solver = ElasticSolver(dens, p.nu, p.E0, p.e_min_ratio, p.dx, p.clamp_bottom)

    # The transport block is dropped whenever the flow is identically zero, which is BOTH of the
    # spec's nested off switches, not just one of them:
    #   c_cp = 0  -- "the eigenstrain vanishes, so Delta x = 0 and every flux with it"
    #   I0   = 0  -- "M = id": no fluence, so no dose, no dw, no eigenstrain, no displacement
    # Dropping it is faithful (the fluxes really are zero), and it is also the only way to keep
    # the model differentiable there.  The relative smoothing of eq:xd_upwind is
    # eps_up^2 = eps_rel^2 * mean|Delta x|^2, so it vanishes WITH the flow -- which is the
    # property that restores the exact collapse, and the price is that sqrt(v^2 + eps_up^2)
    # becomes a norm of Delta x and has no derivative at Delta x = 0.  A constant eps_up did not
    # have that failure mode (it paid in resting diffusion instead), so this is a genuine
    # trade-off the spec presents as a clean win and is not one.  Measured: with I0 = 0 and
    # c_cp = 0.3 left in the transport block, IPOPT dies with
    # "Error evaluating ... can't evaluate sqrt'(0)".
    transport = (p.c_cp != 0.0) and (p.I0 != 0.0)

    # eq:xd_upwind with no smoothing is sqrt(v^2) = |v|, and IPOPT says so in as many words
    # ("Error evaluating constraint N: can't evaluate sqrt'(0)") the moment any face is at rest.
    # The numpy simulator is fine with it -- it never differentiates -- which is exactly why the
    # tab defaults to 0 and the NLP cannot.  Caught here rather than in the solver log.
    differentiable = (not transport) or p.eps_rel > 0.0 or p.eps_up > 0.0
    if not differentiable and not allow_nondifferentiable:
        raise NonDifferentiableModel(
            "eps_up = eps_rel = 0 with c_cp = %g: eq:xd_upwind reduces to |v|, which has no "
            "derivative at v = 0, so this model cannot be solved. Set eps_rel > 0 (1e-3 is the "
            "spec's value, and leaves the collapse and the I0 = 0 identity exact -- see "
            "degrade_v2.check_invariants), or set c_cp = 0 to switch transport off." % p.c_cp)

    m = pyo.ConcreteModel(name="degrade_v2")
    # Plain attributes only.  k_aug clones the model, and an ElasticSolver carries a SuperLU
    # factorisation that cannot be deep-copied -- stashing it here made every sensitivity
    # extraction print "Unable to clone Pyomo component attribute", which looks like a failure
    # and is not one.  Nothing downstream needs the solver object anyway.
    m.res, m.n_steps, m.meas = res, K, meas
    m.p = p
    m.transport, m.frozen = transport, bool(freeze_mechanics)
    m.differentiable = differentiable

    # --- state ------------------------------------------------------------------------
    m.PIX = pyo.RangeSet(0, npix - 1)
    m.T = pyo.RangeSet(0, K)             # f, Q live on 0..K
    m.TM = pyo.RangeSet(0, K - 1)        # measurements on 0..K-1

    flat = theta_ref.ravel()
    m.f = pyo.Var(m.PIX, m.T, bounds=f_bounds,
                  initialize=lambda _m, q, k: float(flat[q]))
    m.Q = pyo.Var(m.PIX, m.T, initialize=0.0)
    for q in m.PIX:
        m.Q[q, 0].fix(0.0)               # Q_0 = 0

    # --- 1. photon balance: the shielding chain, in travel order -----------------------
    chain, ray_id = [], []
    for k, (_ang, rays) in enumerate(meas):
        for j, (_r, walk) in enumerate(rays):
            ray_id.append((k, j, len(walk)))
            chain.extend((k, j, t) for t in range(len(walk) + 1))
    m.CH = pyo.Set(initialize=chain, dimen=3, ordered=True)
    m.S = pyo.Var(m.CH, initialize=0.0)
    m.RAY = pyo.Set(initialize=[(k, j) for (k, j, _n) in ray_id], dimen=2, ordered=True)

    def _chain(mm, k, j, t):
        if t == 0:
            return mm.S[k, j, 0] == 0.0
        _pix, chord, shield = meas[k][1][j][1][t - 1]
        return mm.S[k, j, t] == mm.S[k, j, t - 1] + chord * mm.f[shield, k]
    m.c_chain = pyo.Constraint(m.CH, rule=_chain)

    # I_p and the dose increment: rays superpose, and one pixel can be crossed twice.
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

    # --- 2. dose accumulation, eq:xd_dose_state ----------------------------------------
    def _dose(mm, q, k):
        terms = dQ_terms[(q, k)]
        if not terms:
            return mm.Q[q, k + 1] == mm.Q[q, k]
        inc = sum(p.c_q * p.I0 * pyo.exp(-mm.S[idx]) * chord for idx, chord in terms)
        return mm.Q[q, k + 1] == mm.Q[q, k] + inc
    m.c_dose = pyo.Constraint(m.PIX, m.TM, rule=_dose)

    # --- 4, 5. response and the RELATIVE converted fraction ----------------------------
    def _om(expr):
        return p.omega_inf + (1.0 - p.omega_inf) * pyo.exp(-expr / p.Q_c)

    m.dw = pyo.Var(m.PIX, m.TM, initialize=0.0)

    def _dwc(mm, q, k):
        # dw = 1 - omega(Q_{k+1})/omega(Q_k), written multiplicatively.  RELATIVE, not the
        # absolute difference -- that is the classic bug this model is written against.
        return mm.dw[q, k] * _om(mm.Q[q, k]) == _om(mm.Q[q, k]) - _om(mm.Q[q, k + 1])
    m.c_dw = pyo.Constraint(m.PIX, m.TM, rule=_dwc)

    # --- 6, 7. eigenstrain and equilibrium, eq:xd_elastic_discrete ---------------------
    nn = solver.nn
    if transport and not freeze_mechanics:
        m.DOF = pyo.RangeSet(0, solver.ndof_total - 1)
        m.u = pyo.Var(m.DOF, m.TM, initialize=0.0)
        pinned = np.setdiff1d(np.arange(solver.ndof_total), solver._free)
        for d in pinned:
            for k in range(K):
                m.u[int(d), k].fix(0.0)

        Kmat = solver.K.tocsr()
        edof, Ee, Le = solver._edof, solver._Ee, solver._Le
        load = {}                                  # dof -> [(element, coefficient), ...]
        for e in range(npix):
            for loc in range(8):
                load.setdefault(int(edof[e, loc]), []).append((e, float(Ee[e] * Le[loc])))

        free_set = set(int(d) for d in solver._free)
        m.FREE = pyo.Set(initialize=sorted(free_set), ordered=True)

        def _eq(mm, d, k):
            lo, hi = Kmat.indptr[d], Kmat.indptr[d + 1]
            lhs = sum(float(Kmat.data[z]) * mm.u[int(Kmat.indices[z]), k]
                      for z in range(lo, hi) if int(Kmat.indices[z]) in free_set)
            rhs = sum(-0.5 * p.c_cp * c * mm.dw[e, k] for e, c in load.get(d, ()))
            return lhs == rhs
        m.c_elastic = pyo.Constraint(m.FREE, m.TM, rule=_eq)

        def _cx(mm, i, j, k):              # nodal -> pixel centre, as ElasticSolver.solve does
            return 0.25 * sum(mm.u[2 * ((i + a) * nn + (j + b)), k]
                              for a in (0, 1) for b in (0, 1))

        def _cy(mm, i, j, k):
            return 0.25 * sum(mm.u[2 * ((i + a) * nn + (j + b)) + 1, k]
                              for a in (0, 1) for b in (0, 1))
    elif transport:
        vel = [(np.asarray(cx, float), np.asarray(cy, float)) for cx, cy in frozen_velocity]

        def _cx(mm, i, j, k):
            return float(vel[k][0][i, j])

        def _cy(mm, i, j, k):
            return float(vel[k][1][i, j])

    # --- 8. eq:xd_upwind's smoothing, carried squared ----------------------------------
    if transport:
        if freeze_mechanics:
            eps_sq_fixed = [
                p.eps_rel ** 2 * float(np.mean(vel[k][0] ** 2 + vel[k][1] ** 2))
                if p.eps_rel > 0.0 else p.eps_up ** 2 for k in range(K)]

            def _eps(mm, k):
                return float(eps_sq_fixed[k])
            m.eps_sq = pyo.Param(m.TM, initialize=_eps, mutable=False)
        else:
            m.eps_sq = pyo.Var(m.TM, initialize=max(p.eps_up ** 2, 1e-18), domain=pyo.NonNegativeReals)
            if p.eps_rel > 0.0:
                def _epsc(mm, k):
                    # One variable, one row.  Inlining this sum over the whole grid into all
                    # ~8k face splits would make it a shared subexpression of every one.
                    return mm.eps_sq[k] * npix == p.eps_rel ** 2 * sum(
                        _cx(mm, q // res, q % res, k) ** 2 + _cy(mm, q // res, q % res, k) ** 2
                        for q in range(npix))
                m.c_eps = pyo.Constraint(m.TM, rule=_epsc)
            else:
                for k in range(K):
                    m.eps_sq[k].fix(p.eps_up ** 2)

    # --- 9, 10. decay then transport, eq:xd_decay then eq:xd_mass_transport ------------
    def _ftilde(mm, q, k):
        return mm.f[q, k] * pyo.exp(-p.a * mm.Ipix[q, k] - p.b * mm.Ipix[q, k] ** 2)

    def _split(v, e2):
        s = pyo.sqrt(v ** 2 + e2)
        return 0.5 * (s + v), 0.5 * (s - v)

    def _mass(mm, q, k):
        i, j = q // res, q % res
        rhs = _ftilde(mm, q, k)
        if transport:
            e2 = mm.eps_sq[k]
            # Every face this pixel owns.  Signs mirror upwind_flux_divergence exactly: the
            # face between (i, j) and (i, j+1) adds +F/dx here and -F/dx there.
            if j < res - 1:
                vp, vm = _split(0.5 * (_cx(mm, i, j, k) + _cx(mm, i, j + 1, k)), e2)
                rhs -= (vp * _ftilde(mm, q, k) - vm * _ftilde(mm, q + 1, k)) / p.dx
            if j > 0:
                vp, vm = _split(0.5 * (_cx(mm, i, j - 1, k) + _cx(mm, i, j, k)), e2)
                rhs += (vp * _ftilde(mm, q - 1, k) - vm * _ftilde(mm, q, k)) / p.dx
            if i < res - 1:
                vp, vm = _split(0.5 * (_cy(mm, i, j, k) + _cy(mm, i + 1, j, k)), e2)
                rhs -= (vp * _ftilde(mm, q, k) - vm * _ftilde(mm, q + res, k)) / p.dx
            if i > 0:
                vp, vm = _split(0.5 * (_cy(mm, i - 1, j, k) + _cy(mm, i, j, k)), e2)
                rhs += (vp * _ftilde(mm, q - res, k) - vm * _ftilde(mm, q, k)) / p.dx
        return mm.f[q, k + 1] == rhs
    m.c_mass = pyo.Constraint(m.PIX, m.TM, rule=_mass)

    # --- 11. observation: the last link of the chain, no new variable ------------------
    m.obs_index = [(k, j, n) for (k, j, n) in ray_id]
    return m


# --- a check against the SPEC, not against a sibling implementation --------------------------

def reference_local_intensity(f, r, angle_rad, I0, c_q):
    """eq:xd_local_intensity and eq:xd_dose_state, written from the spec alone.

    The spec says the ray "crosses pixels p_1, p_2, ... IN TRAVERSAL ORDER with chord lengths
    delta_{p_i}" and sums over ``m < i``: upstream material shields downstream material, and a
    pixel never shields itself.  So: walk the SEGMENTS in travel order; the pixel owning chord
    ``s`` is ``rows[s]``; deposit there; then let it shield everything after it.

    **How this resolves delta_p for a bundle, which the spec leaves implicit.**  The note writes
    ``delta_p`` as though a pixel has one chord, and under a single ray it does.  Under a bundle
    it does not: eq:xd_local_intensity "sums the contributions" of simultaneous rays and a pixel
    is crossed ~1.17 times per projection on a 64 grid, so the chord is really per
    ``(ray, pixel)`` pair while the notation carries no ray index.  This resolves it as:
    ``I_p`` is the sum over rays of the per-ray intensity and carries NO chord factor, while the
    dose increment is the sum over rays of ``c_q * I_ray * delta_(ray,p)``.  That is the same
    reading :func:`degrade_v2.accumulate_dose` takes, so the two differ only in the deposit index.

    This exists because agreeing with :mod:`degrade_v2` to 1e-16 proves only that two
    implementations share a convention -- including a wrong one.  The manuscript records exactly
    that failure mode for its own ordering test ("easy to write so that it passes vacuously").
    None of the three structural invariants in ``degrade_v2.check_invariants`` can catch a
    misplaced deposit, because all three are self-consistency properties of the composition and
    none of them asks which pixel the dose landed in.  This one does.
    """
    f = np.asarray(f, dtype=float)
    res = f.shape[0]
    dQ = np.zeros_like(f)
    I_sum = np.zeros_like(f)
    g = ray_geometry(float(r), float(angle_rad), res, res)
    if g is None:
        return dQ, I_sum
    rows, cols, seg, forward = g
    n_seg = len(seg)
    shield = 0.0
    for s in (range(n_seg) if forward else range(n_seg - 1, -1, -1)):
        local = I0 * np.exp(-shield)
        pix = (int(rows[s]), int(cols[s]))
        dQ[pix] += c_q * local * seg[s]
        I_sum[pix] += local
        shield += seg[s] * f[pix]
    return dQ, I_sum


def check_photon_balance(image_res: int = 12, verbose: bool = True, strict: bool = True):
    """Compare ``degrade_v2.accumulate_dose`` against :func:`reference_local_intensity`.

    Reports forward-ordered and antiparallel rays separately, because that is where they differ.
    Both families are asserted.  They did not always agree: antiparallel rays were out by a
    full ``I0`` until the deposit index was fixed, because each pixel was shielded by its own
    chord.  This is the check that found it.

    **What it does NOT certify, and this is the honest limit of it.**  The reference calls
    :func:`dose_response.ray_geometry` for its crossing list, so it is independent of
    ``accumulate_dose`` only in the *walk* -- the travel order and the deposit index.  It shares
    the underlying chord/pixel attribution and is therefore blind to any error in it.  There is
    one: see :func:`check_chord_attribution`.  The lesson in the module docstring applies to this
    function too, one level down, which is worth saying plainly rather than leaving for someone
    to discover: a reference is only independent along the axes on which it does not reuse the
    thing it checks.
    """
    from degrade_v2 import accumulate_dose

    rng = np.random.default_rng(7)
    res = int(image_res)
    f = rng.random((res, res)) * 0.4          # non-uniform: a symmetric object hides the defect
    worst = {"forward": 0.0, "antiparallel": 0.0}
    count = {"forward": 0, "antiparallel": 0}
    for ang_deg in np.arange(0.0, 360.0, 15.0):
        ang = float(np.deg2rad(ang_deg))
        for r in bundle_r_values(0.0, 0, res):
            g = ray_geometry(float(r), ang, res, res)
            if g is None:
                continue
            key = "forward" if g[3] else "antiparallel"
            _dq_r, I_ref = reference_local_intensity(f, r, ang, 1.0, 1.0)
            _dq_c, I_code = accumulate_dose(f, [r], ang, 1.0, 1.0)
            worst[key] = max(worst[key], float(np.abs(I_ref - I_code).max()))
            count[key] += 1
    if verbose:
        print("eq:xd_local_intensity -- accumulate_dose against a reference from the spec alone")
        for k in ("forward", "antiparallel"):
            print("    %-13s rays: max |I_p(code) - I_p(spec)| = %.3e   over %d rays"
                  % (k, worst[k], count[k]))
        if worst["antiparallel"] > 1e-12:
            print("    ^ antiparallel rays disagree. On those the deposit lands on rows[i] while")
            print("      the chord just added to the shielding belongs to rows[i-1], so each pixel")
            print("      is shielded by its own chord. dose_response.degradation_dose_response has")
            print("      the same split, so the 2D live picture and the 3D simulator share it.")
    assert worst["forward"] < 1e-12, "forward rays disagree with the spec -- that is a new bug"
    if strict:
        assert worst["antiparallel"] < 1e-12, (
            "antiparallel rays disagree with eq:xd_local_intensity by %.3e -- the deposit index "
            "regressed: eq:xd_dose_state puts the deposit pixel and the chord owner at the same "
            "symbol p, so deposit into rows[s], not rows[i]" % worst["antiparallel"])
    return worst


def check_chord_attribution(image_res: int = 32, verbose: bool = True, strict: bool = False):
    """Is each chord attributed to the pixel that actually CONTAINS it?

    Independent of the crossing-point convention, because the ground truth is the pixel holding
    the segment's own MIDPOINT -- a segment lies in exactly one pixel, and its midpoint is
    interior, so there is no boundary ambiguity to resolve.

    The vendored ``line_grid_intersections`` instead labels each segment with the pixel at its
    *starting crossing*, via ``col = int(x + w/2)``, ``row = int(h/2 - y)``.  Sorted ascending in
    ``(x, y)``, that names the correct pixel only when the line has negative slope.  For positive
    slope -- ``theta`` roughly in ``(90, 180)`` degrees, plus the axis-aligned cases -- the row is
    off by one, so the chord is charged to a neighbour.

    Measured: 0% mis-attributed on negative-slope rays, 20-100% on positive-slope ones, rising as
    the ray approaches axis-aligned.  The *line integral* barely notices (0.31% against 0.46%
    mean error on an analytic disc, both discretisation-level), so sinograms are essentially
    unaffected; what moves is WHERE the dose lands.  At 135 degrees the per-pixel intensity field
    differs from the corrected one by 52% of its own peak, while the full 12-projection Rg moves
    by 0.01 percentage points -- the same signature as the deposit-index defect, a large per-ray
    error that averages out over a scan.

    ``strict`` is off: the fix belongs in :func:`dose_response.ray_geometry` (the vendored file
    must not be edited), and it would move the 2D picture, the 3D simulator and
    ``check_reference_numbers``, while NOT moving v1's Pyomo path, which calls
    ``line_grid_intersections`` directly.  That is a decision, not a cleanup.
    """
    from senDOE.helpers.geometry import get_line_abc_from_r_theta, line_grid_intersections

    n = int(image_res)
    by_slope = {"negative": [0, 0], "positive": [0, 0]}
    for ang_deg in np.arange(0.0, 360.0, 7.5):
        th = float(np.deg2rad(ang_deg))
        a, b, c = get_line_abc_from_r_theta(0.5, th)
        try:
            cross, pix, _r, seg = line_grid_intersections(
                a, b, c, np.zeros((n, n)), x_range=[-n / 2, n / 2], y_range=[-n / 2, n / 2])
        except IndexError:
            continue
        st, ct = np.sin(th), np.cos(th)
        key = "positive" if (abs(st) < 1e-9 or abs(ct) < 1e-9 or (-ct / st) > 0) else "negative"
        for sgi in range(len(seg)):
            (x0, y0), (x1, y1) = cross[sgi], cross[sgi + 1]
            mx, my = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            ct_ = min(max(int(mx + n / 2), 0), n - 1)
            rt_ = min(max(int(n / 2 - my), 0), n - 1)
            by_slope[key][1] += 1
            if (int(pix[sgi, 0]), int(pix[sgi, 1])) != (rt_, ct_):
                by_slope[key][0] += 1
    out = {k: (v[0] / v[1] if v[1] else 0.0) for k, v in by_slope.items()}
    if verbose:
        print("chord attribution -- is each chord charged to the pixel containing it?")
        for k in ("negative", "positive"):
            print("    %-9s-slope rays: %d of %d mis-attributed (%.1f%%)"
                  % (k, by_slope[k][0], by_slope[k][1], 100 * out[k]))
    assert out["negative"] < 1e-12, "negative-slope rays regressed -- that is a new bug"
    if strict:
        assert out["positive"] < 1e-12, (
            "positive-slope rays mis-attribute %.1f%% of chords" % (100 * out["positive"]))
    return out


# --- pinning the model to a numpy trajectory -------------------------------------------------

def numpy_trajectory(theta, seq, p: V2Params, image_res: int, solver=None):
    """Everything the Pyomo model has a variable for, computed by :mod:`degrade_v2`.

    Returns a dict of arrays keyed like the model's variables.  Used to pin the model for the
    residual check, to initialise it for a solve, and as the answer the forward solve is
    compared against.

    With ``solver`` left ``None`` this runs :func:`degrade_v2.simulate` unmodified -- which is
    the point, since the whole value of the cross-check is that the two implementations are
    independent.  Passing an :class:`ElasticSolver` instead steps the model by hand with *that*
    stiffness: needed only to initialise an inverse solve, where ``K`` is assembled from the
    reference density and not from the current guess, so a trajectory built the other way would
    start infeasible in the elasticity rows.
    """
    from degrade_v2 import accumulate_dose, omega as _omega, step as _step

    theta = np.asarray(theta, dtype=float)
    res = int(image_res)
    meas = measurement_rays(seq, res)
    K = len(meas)
    if solver is None:
        solver = ElasticSolver(theta, p.nu, p.E0, p.e_min_ratio, p.dx, p.clamp_bottom)
        _f, _Q, infos, obs, (f_hist, Q_hist) = simulate(
            theta, seq, p, res, record_observations=True, record_trajectory=True)
    else:
        fk, Qk = theta.copy(), np.zeros_like(theta)
        f_hist, Q_hist, obs, infos = [fk.copy()], [Qk.copy()], [], []
        for (angle_deg, offset, n_beams) in seq:
            ang = float(np.deg2rad(float(angle_deg)))
            rs = bundle_r_values(float(offset), int(n_beams), res)
            from dose_response import ray_line_integral as _rli
            obs.append(np.array([_rli(fk, r, ang) for r in rs]))
            fk, Qk, info = _step(fk, Qk, rs, ang, p, solver)
            f_hist.append(fk.copy())
            Q_hist.append(Qk.copy())
            infos.append(info)

    S, Ipix, dW, U, EPS = {}, np.zeros((res * res, K)), np.zeros((res * res, K)), [], []
    for k, (ang, rays) in enumerate(meas):
        fk = f_hist[k].ravel()
        for j, (_r, walk) in enumerate(rays):
            acc = 0.0
            S[(k, j, 0)] = 0.0
            for t, (_pix, chord, shield) in enumerate(walk):
                acc += chord * fk[shield]
                S[(k, j, t + 1)] = acc
        rs = [r for r, _w in rays]
        _dQ, I_sum = accumulate_dose(f_hist[k], rs, ang, p.I0, p.c_q)
        Ipix[:, k] = I_sum.ravel()
        dw = 1.0 - (_omega(Q_hist[k + 1], p.omega_inf, p.Q_c)
                    / _omega(Q_hist[k], p.omega_inf, p.Q_c))
        dW[:, k] = dw.ravel()
        u = solver.solve_nodal(dw, p.c_cp)
        U.append(u)
        cx, cy = solver.solve(dw, p.c_cp)
        EPS.append(p.eps_rel ** 2 * float(np.mean(cx ** 2 + cy ** 2))
                   if p.eps_rel > 0.0 else p.eps_up ** 2)

    return dict(f=np.stack([h.ravel() for h in f_hist], axis=1),
                Q=np.stack([h.ravel() for h in Q_hist], axis=1),
                S=S, Ipix=Ipix, dw=dW, u=np.stack(U, axis=1) if U else None,
                eps_sq=np.array(EPS), obs=obs, infos=infos, meas=meas)


def pin_model(m, traj, *, fix=True):
    """Set (and optionally fix) every model variable to a :func:`numpy_trajectory`."""
    for q in m.PIX:
        for k in m.T:
            m.f[q, k].set_value(float(traj["f"][q, k]))
            m.Q[q, k].set_value(float(traj["Q"][q, k]))
        for k in m.TM:
            m.Ipix[q, k].set_value(float(traj["Ipix"][q, k]))
            m.dw[q, k].set_value(float(traj["dw"][q, k]))
    for idx in m.CH:
        m.S[idx].set_value(float(traj["S"][idx]))
    if m.transport and not m.frozen:
        for d in m.DOF:
            for k in m.TM:
                m.u[d, k].set_value(float(traj["u"][d, k]))
    if m.transport and not m.frozen:
        for k in m.TM:
            m.eps_sq[k].set_value(float(traj["eps_sq"][k]))
    if not fix:
        return m
    for v in m.component_data_objects(pyo.Var):
        v.fix()
    return m


def max_residual(m, per_constraint=False):
    """Largest violation of any active equality, and which one it was."""
    worst, where, by_block = 0.0, None, {}
    for c in m.component_data_objects(pyo.Constraint, active=True):
        try:
            r = abs(pyo.value(c.body) - pyo.value(c.lower))
        except (ValueError, ZeroDivisionError):
            continue
        name = c.parent_component().name
        by_block[name] = max(by_block.get(name, 0.0), r)
        if r > worst:
            worst, where = r, c.name
    return (worst, where, by_block) if per_constraint else (worst, where)


# --- solving ---------------------------------------------------------------------------------

_FALLBACK_LINEAR_SOLVERS = ["ma27", "ma57", "mumps"]


def _make_solver(linear_solver: str, max_iter: int, tol: float = 1e-8, options=None):
    from tomography_uq import _resolve_ipopt          # one definition of where IPOPT lives
    s = pyo.SolverFactory("ipopt", executable=_resolve_ipopt())
    s.options["max_iter"] = int(max_iter)
    s.options["linear_solver"] = linear_solver
    s.options["tol"] = float(tol)
    for k, v in (options or {}).items():
        s.options[k] = v
    return s


# IPOPT stops on the *scaled* dual error, but a square forward run has no objective at all, so
# what decides its accuracy is constr_viol_tol -- which defaults to 1e-4.  Left alone it returns
# "optimal" with the fields still 1e-5 out, which reads as a model disagreement and is not one.
_FEASIBILITY_OPTIONS = {
    "constr_viol_tol": 1e-14,
    "acceptable_constr_viol_tol": 1e-14,
    "acceptable_tol": 1e-14,
    "bound_relax_factor": 0.0,
}


def solve_with_fallback(model, *, linear_solver="ma27", max_iter=3000, tol=1e-8, tee=False,
                        log_callback=None, options=None):
    """Solve, walking ``ma27 -> ma57 -> mumps`` only when a solver fails to *run*.

    Same rule as :mod:`tomography_uq`: a non-optimal termination is a result, not a reason to
    switch linear solvers.  ``ma86`` is deliberately absent -- it is not in the IDAES build.
    """
    order = [linear_solver] + [s for s in _FALLBACK_LINEAR_SOLVERS if s != linear_solver]
    last = None
    for name in order:
        try:
            res = _solve_streaming(_make_solver(name, max_iter, tol, options), model, tee,
                                   log_callback)
            return res, name
        except Exception as exc:
            text = str(exc)
            if _is_model_error(text):
                # IPOPT launched and rejected the *problem*.  Retrying it on ma57 and mumps
                # would fail identically and report "no usable linear solver", which sends the
                # reader hunting for a missing binary that is sitting right there.
                raise RuntimeError(_curate(text)) from exc
            last = exc
            if log_callback:
                log_callback("\n[linear solver %r unavailable: %s]\n" % (name, text[:200]))
    raise RuntimeError("no usable IPOPT linear solver (tried %s): %s" % (order, last))


def _is_model_error(text: str) -> bool:
    """Did IPOPT start and object to the model, rather than fail to start?

    Any of these means the binary ran: retrying on another linear solver fails identically and
    reports a missing binary that is sitting right there.
    """
    return ("can't evaluate" in text or "Error evaluating" in text
            or "Invalid number" in text or "Ipopt " in text
            or "too few degrees of freedom" in text)


def _curate(text: str) -> str:
    if "sqrt'(0)" in text:
        return ("IPOPT could not differentiate eq:xd_upwind: \"can't evaluate sqrt'(0)\". "
                "Some face is at rest and the smoothing there is zero, so the upwind split is "
                "|v|. The relative smoothing eps_up^2 = eps_rel^2*mean|dx|^2 vanishes with the "
                "flow by design, so raising eps_rel does NOT fix a field that is identically "
                "motionless -- set eps_up > 0 (a constant, which does not vanish) or switch "
                "transport off with c_cp = 0.")
    first = [ln for ln in text.splitlines() if "valuat" in ln]
    return "IPOPT rejected the model: %s" % (first[0].strip() if first else text[:300])


def _solve_streaming(solver, model, tee, log_callback):
    """``solver.solve(tee=True)`` with the subprocess log forwarded to ``log_callback``.

    The log is ALWAYS captured, even with no ``log_callback``.  Pyomo's ``tee=False`` buries the
    solver's stdout in a temp file it then deletes, and raises only "Solver (ipopt) did not exit
    normally" -- so a model IPOPT explicitly rejected ("can't evaluate sqrt'(0)") became
    indistinguishable from a missing binary, and :func:`solve_with_fallback` walked the whole
    linear-solver chain and blamed the linear solver.  Capturing it means the reason survives
    into the exception.
    """
    try:
        from pyomo.common.tee import capture_output
    except Exception:
        return solver.solve(model, tee=tee)

    buf = []

    class _W:
        def write(self, chunk):
            if chunk:
                buf.append(chunk)
                if log_callback is not None:
                    log_callback(chunk)
            return len(chunk)

        def flush(self):
            pass

    try:
        with capture_output(_W()):
            return solver.solve(model, tee=True)
    except Exception as exc:
        tail = "".join(buf)[-4000:]
        raise RuntimeError("%s\n--- solver log ---\n%s" % (exc, tail)) from exc


# --- the gate: does the Pyomo model reproduce the numpy forward model? -----------------------

def check_forward(image_res: int = 24, n_steps: int = 3, verbose: bool = True,
                  c_cp: float = 0.3, eps_rel: float = 1e-3, clamp_bottom: bool = False,
                  solve: bool = True, linear_solver: str = "ma27"):
    """Compare the Pyomo transcription against :func:`degrade_v2.simulate`, two ways.

    **Residual.**  Pin every variable to the numpy trajectory and evaluate every constraint.
    A correct transcription leaves nothing: this is the check that says the NLP encodes v2 and
    not something adjacent to it.  It needs no solver, so it runs in the Docker build.

    **Forward solve.**  Fix ``f[:, 0] = theta``, start IPOPT at the undamaged field (which is
    *not* the answer for any step past the first), and let it find the trajectory.  Compare
    field by field.  This is strictly more than the residual check: it also says the model is
    square, solvable, and scaled well enough to converge -- none of which a residual at the
    true solution can tell you.

    Returns a dict of measured errors.  Raises ``AssertionError`` if either fails.
    """
    from skimage.data import shepp_logan_phantom
    from skimage.transform import resize

    res = int(image_res)
    theta = scale_to_optical_depth(
        resize(shepp_logan_phantom(), (res, res)).astype(float), 1.1, res)
    seq = tuple((180.0 * i / n_steps, 0.0, 0) for i in range(n_steps))
    p = V2Params(c_cp=c_cp, eps_rel=eps_rel, eps_up=0.0, clamp_bottom=clamp_bottom)
    out = {}

    def say(msg):
        if verbose:
            print(msg)

    say("v2 Pyomo model vs degrade_v2.simulate   [%dx%d, %d steps, c_cp=%.2f, eps_rel=%g%s]"
        % (res, res, n_steps, c_cp, eps_rel, ", clamped" if clamp_bottom else ""))

    pb = check_photon_balance(image_res=min(res, 16), verbose=False)
    out["photon_balance"] = max(pb.values())
    say("    eq:xd_local_intensity vs a reference from the spec: %.3e (both ray families)"
        % out["photon_balance"])

    traj = numpy_trajectory(theta, seq, p, res)
    m = build_v2_model(theta, seq, p, res, allow_nondifferentiable=True)
    n_v = sum(1 for _ in m.component_data_objects(pyo.Var))
    n_c = sum(1 for _ in m.component_data_objects(pyo.Constraint, active=True))
    say("    %d variables, %d constraints" % (n_v, n_c))

    # --- residual -------------------------------------------------------------------
    pin_model(m, traj)
    worst, where, blocks = max_residual(m, per_constraint=True)
    for name in sorted(blocks, key=lambda n: -blocks[n]):
        say("        %-12s %.3e" % (name, blocks[name]))
    out["residual"] = worst
    say("    residual   max |constraint| = %.3e   (%s)" % (worst, where))
    assert worst < 1e-10, "the Pyomo model does not reproduce the numpy trajectory: %s" % where

    # ... and the same for the frozen-mechanics build, which swaps the elasticity block for
    # precomputed velocities and so is a genuinely different set of constraints.
    solver_f = ElasticSolver(theta, p.nu, p.E0, p.e_min_ratio, p.dx, p.clamp_bottom)
    vel = [solver_f.solve(traj["dw"][:, k].reshape(res, res), p.c_cp) for k in range(n_steps)]
    m_f = build_v2_model(theta, seq, p, res, freeze_mechanics=True, frozen_velocity=vel,
                         allow_nondifferentiable=True)
    pin_model(m_f, traj)
    wf, wheref = max_residual(m_f)
    out["residual_frozen"] = wf
    say("    residual   max |constraint| = %.3e   (%s)   [frozen mechanics]" % (wf, wheref))
    assert wf < 1e-10, "the frozen-mechanics model does not reproduce the trajectory: %s" % wheref
    del m_f

    if not solve:
        return out
    if not m.differentiable:
        # Not a failure of the transcription: the residual above is exact.  This setting simply
        # cannot be handed to a solver, which is the whole reason eps_rel exists.
        out["forward_status"] = "skipped (not differentiable)"
        say("    forward solve  SKIPPED: eps_up = eps_rel = 0 gives |v|, no derivative at rest")
        return out

    # --- forward solve --------------------------------------------------------------
    m2 = build_v2_model(theta, seq, p, res)
    flat = theta.ravel()
    for q in m2.PIX:
        m2.f[q, 0].fix(float(flat[q]))          # theta is known in a forward run
    m2.obj = pyo.Objective(expr=0.0)            # square system: pure feasibility
    res_obj, used = solve_with_fallback(m2, linear_solver=linear_solver, max_iter=500,
                                        tol=1e-12, options=_FEASIBILITY_OPTIONS)
    tc = str(res_obj.solver.termination_condition)
    out["forward_status"], out["forward_linear_solver"] = tc, used
    say("    forward solve  termination=%s  (%s)" % (tc, used))
    assert tc in ("optimal", "locallyOptimal", "feasible"), "forward solve did not converge: %s" % tc

    f_py = np.array([[pyo.value(m2.f[q, k]) for k in m2.T] for q in m2.PIX])
    Q_py = np.array([[pyo.value(m2.Q[q, k]) for k in m2.T] for q in m2.PIX])
    scale = float(np.abs(traj["f"]).max())
    out["f_err"] = float(np.abs(f_py - traj["f"]).max())
    out["Q_err"] = float(np.abs(Q_py - traj["Q"]).max())
    out["f_err_rel"] = out["f_err"] / scale
    say("    forward fields  max|df| = %.3e  (%.2e relative)   max|dQ| = %.3e"
        % (out["f_err"], out["f_err_rel"], out["Q_err"]))

    # the observation is the last link of the shielding chain -- check it against step 11
    o_err = 0.0
    for k, j, n in m2.obs_index:
        o_err = max(o_err, abs(pyo.value(m2.S[k, j, n]) - float(traj["obs"][k][j])))
    out["obs_err"] = o_err
    say("    observation     max|dy| = %.3e" % o_err)

    assert out["f_err_rel"] < 1e-7, "forward solve disagrees with the numpy model"
    assert o_err < 1e-9, "the observation chain does not reproduce ray_line_integral"
    say("    AGREES")
    return out




# --- reconstruction ---------------------------------------------------------------------------

@dataclass
class V2UQParams:
    """Inputs to :func:`run_v2_reconstruction`.  Physics defaults match the v2 tab's seeds."""

    image_res: int = 64
    optical_depth: float = 1.1
    beam_steps: tuple = ()             # (angle_deg, offset, n_beams) triples, _table_to_seq form
    phantom: Optional[np.ndarray] = None

    # --- v2 physics (V2Params, minus eps_up: the NLP needs the relative form) ---
    I0: float = 1.0
    c_q: float = 0.032
    Q_c: float = 1.0
    omega_inf: float = 0.2
    c_cp: float = 0.3
    a: float = 0.05
    b: float = 0.0
    E0: float = 1.0
    nu: float = 0.3
    e_min_ratio: float = 1e-6
    clamp_bottom: bool = False
    dx: float = 1.0
    eps_rel: float = 1e-3

    # --- estimation ---
    tv_weight: float = 0.001
    noise_sigma: float = 0.0           # 0 = noiseless data, as v1 does
    noise_cov_scale: float = 10.0      # sigma^2 in Sigma = sigma^2 J J^T
    freeze_mechanics: bool = False
    continuation: bool = True          # seed from the I0 = 0 (undamaged) solve
    run_uq: bool = True
    ipopt_max_iter: int = 3000
    linear_solver: str = "ma27"

    def physics(self, **over) -> V2Params:
        kw = dict(I0=self.I0, c_q=self.c_q, Q_c=self.Q_c, omega_inf=self.omega_inf,
                  c_cp=self.c_cp, a=self.a, b=self.b, eps_up=0.0, eps_rel=self.eps_rel,
                  E0=self.E0, nu=self.nu, e_min_ratio=self.e_min_ratio,
                  clamp_bottom=self.clamp_bottom, dx=self.dx)
        kw.update(over)
        return V2Params(**kw)


@dataclass
class V2UQResults:
    """Arrays and scalars, not matplotlib figures -- the caller draws.

    Same choice ``_recon_slice_3d`` makes in ``app.py``: figures in session state are expensive
    and this way the 2D and 3D renderers stay the app's business.
    """

    theta_true: np.ndarray
    theta_hat: np.ndarray
    f_final_true: np.ndarray
    f_final_hat: np.ndarray
    Q_final_hat: np.ndarray
    log_cov_diag_2D: Optional[np.ndarray] = None
    d_optimality: float = float("nan")
    inverse_status: str = ""
    inverse_linear_solver: str = ""
    continuation_status: str = ""
    obs_rms: float = float("nan")          # fit residual, RMS over all rays
    theta_rms: float = float("nan")        # ||theta_hat - theta_true|| RMS, synthetic-data only
    forward_residual: float = float("nan")  # the gate, re-measured on this exact geometry
    n_measurements: int = 0
    n_rays: int = 0
    n_vars: int = 0
    n_cons: int = 0
    uq_error: Optional[str] = None
    uq_conditioning: float = float("nan")   # cond(J J^T); large is expected, see CLAUDE.md
    courant: float = float("nan")           # max|dx|/dx over the run -- how much moved at all
    # eq:xd_box's active set on theta.  The spec appends the box to "g <= 0"; this renders it as
    # variable bounds, which is the same feasible set but reaches k_aug as bound multipliers
    # rather than constraint rows, and the manuscript's IFT sensitivity is built on the active
    # set.  So the counts are reported rather than assumed.  SCOPE: this is the active set of
    # THIS model -- v2 dynamics, this grid, this phantom.  The manuscript's active-constraint
    # claim is evidenced by a different codebase (sDOE_senNLP, 10x10, v1 dose-response, no
    # transport), so nothing here confirms or falsifies that.
    n_theta_at_lower: int = 0
    n_theta_at_upper: int = 0
    n_theta_interior: int = 0
    rg_pct: float = float("nan")            # contraction of the true field, % change in Rg
    mass_true: float = float("nan")
    mass_hat: float = float("nan")


def _tv_expression(m, theta_scale: float):
    """Smoothed isotropic total variation of ``f[:, 0]``.

    Not the vendored ``update_image_TV_expression``: its smoothing is hardcoded at ``eps = 1e-4``,
    which suits v1's 0..1.1 image but swamps this one.  ``theta`` here is scaled to peak optical
    depth ~1.1 over the whole ray, so a *pixel* is ~0.03 and neighbour differences are ~1e-3 --
    ``1e-4`` would dominate the radicand and flatten TV into a constant.  Same functional form,
    with the smoothing tied to the field instead.
    """
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
    """Fit the observations of eq:xd_obs_damage, regularised by TV on ``theta``.

    ``y_data`` enters as *fixed variables*, not Params, because those are precisely the
    parameters k_aug differentiates with respect to.  Unlike v1 they are declared only over the
    rays actually fired, so there are no structurally-dead columns to prune off afterwards.
    """
    m.YD = pyo.Set(initialize=[(k, j) for (k, j, _n) in m.obs_index], dimen=2, ordered=True)
    m.y_data = pyo.Var(m.YD, initialize=0.0)
    for (k, j, n) in m.obs_index:
        m.y_data[k, j].set_value(float(y_data[k][j]))
        m.y_data[k, j].fix()

    # Both terms are normalised to O(1) before they are weighed against each other.  v1 gets
    # away without this because its image runs 0..1.1 and its ray integrals are O(10), so the
    # two land within an order of magnitude by luck.  Here theta peaks near 0.03 while the ray
    # integrals are still O(1) -- optical depth is the product of the two -- so raw sums put the
    # fit ~1e3 above TV and tv_weight would be decoration.  Normalising also makes tv_weight
    # mean roughly the same thing as it does on the other two tabs.
    y_scale = max(float(np.max([np.max(np.abs(y)) for y in y_data])), 1e-30)
    n_obs = len(m.obs_index)
    n_pix = m.res * m.res
    m.fit_expression = sum(
        (m.S[k, j, n] - m.y_data[k, j]) ** 2 for (k, j, n) in m.obs_index) / (n_obs * y_scale ** 2)
    m.tv_expression = _tv_expression(m, theta_scale) / (n_pix * max(theta_scale, 1e-30))
    m.obj = pyo.Objective(expr=m.fit_expression + tv_weight * m.tv_expression)
    return m


def _rg_pct(theta, f_final) -> float:
    """Percent change in radius of gyration -- the contraction the transport actually produced.

    Reported because it qualifies everything else: if the mechanics barely moved the field, a
    comparison of exact against frozen transport is a comparison in a regime where there was
    nothing much to freeze.  Read against the c_cp = 0 baseline, not against zero (the decay
    fades the field non-uniformly, so c_cp = 0 already registers a contraction).
    """
    from degrade_v2 import radius_of_gyration
    r0 = radius_of_gyration(theta)
    return float(100.0 * (radius_of_gyration(f_final) - r0) / r0) if r0 else float("nan")


def _phantom(image_res: int, override=None):
    from skimage.data import shepp_logan_phantom
    from skimage.transform import resize
    if override is not None:
        src = np.asarray(override, dtype=float)
        if src.shape != (image_res, image_res):
            src = resize(src, (image_res, image_res), anti_aliasing=True)
        return src.astype(float)
    return resize(shepp_logan_phantom(), (image_res, image_res)).astype(float)


def run_v2_reconstruction(params: V2UQParams, log_callback=None) -> V2UQResults:
    """Estimate ``theta = f_0`` from the v2 dynamics, and differentiate the estimate.

    Data comes from :func:`degrade_v2.simulate`, so the measurements and the model that fits
    them are two independent implementations of section 3.2 -- which is what makes
    :func:`check_forward` worth running.  That check is re-run here, cheaply and without a
    solver, on the caller's *actual* geometry, and its residual is reported: a reconstruction
    against a model that has drifted from the simulator would otherwise look like a physics
    result.
    """
    def say(msg):
        if log_callback:
            log_callback(msg)

    res = int(params.image_res)
    seq = tuple(params.beam_steps)
    if not seq:
        raise ValueError("no measurements: take at least one before reconstructing")
    p = params.physics()

    theta = scale_to_optical_depth(_phantom(res, params.phantom), params.optical_depth, res)
    scale = float(np.abs(theta).max())

    # --- data, from the numpy simulator ------------------------------------------------
    say("Simulating measurements (numpy forward model)...\n")
    f_true, Q_true, infos, y_true = simulate(theta, seq, p, res, record_observations=True)
    n_rays = int(sum(len(y) for y in y_true))
    if params.noise_sigma > 0.0:
        rng = np.random.default_rng(0)
        y_true = [y + rng.normal(0.0, params.noise_sigma, size=y.shape) for y in y_true]
    say("    %d measurements, %d rays, Courant %.3f, mass left %.4f\n"
        % (len(seq), n_rays, max(i.courant for i in infos), f_true.sum() / theta.sum()))

    # --- the gate: does the Pyomo model still reproduce the simulator here? -------------
    say("Checking the Pyomo model against the simulator on this geometry...\n")
    traj = numpy_trajectory(theta, seq, p, res)
    m_chk = build_v2_model(theta, seq, p, res)
    pin_model(m_chk, traj)
    fwd_resid, where = max_residual(m_chk)
    say("    max constraint residual %.3e  (%s)\n" % (fwd_resid, where))
    del m_chk
    if fwd_resid > 1e-8:
        raise RuntimeError(
            "The Pyomo model no longer reproduces degrade_v2.simulate on this geometry "
            "(residual %.3e at %s). Reconstructing against it would not mean anything; "
            "run degrade_v2_uq.check_forward() to localise the disagreement." % (fwd_resid, where))

    # --- continuation: the undamaged problem first --------------------------------------
    theta0 = np.full_like(theta, float(theta.mean()))
    cont_status = "skipped"
    if params.continuation:
        say("Continuation solve at I0 = 0 (linear tomography + TV)...\n")
        p0 = params.physics(I0=0.0, c_cp=0.0)   # identity dynamics; c_cp=0 drops the flux block
        m0 = build_v2_model(theta, seq, p0, res, f_bounds=(0.0, 1.5 * scale))
        for q in m0.PIX:
            m0.f[q, 0].set_value(float(theta0.ravel()[q]))
        add_estimation_objective(m0, y_true, params.tv_weight, scale)
        r0, ls0 = solve_with_fallback(m0, linear_solver=params.linear_solver,
                                      max_iter=params.ipopt_max_iter, log_callback=log_callback)
        cont_status = str(r0.solver.termination_condition)
        theta0 = np.array([pyo.value(m0.f[q, 0]) for q in m0.PIX]).reshape(res, res)
        say("    %s\n" % cont_status)
        del m0

    # --- the full inverse solve ----------------------------------------------------------
    say("Building the v2 estimation NLP...\n")
    # K comes from the reference density (see the module docstring), so both the initialisation
    # trajectory and any frozen velocities must be stepped with that same stiffness.
    solver_ref = ElasticSolver(theta, p.nu, p.E0, p.e_min_ratio, p.dx, p.clamp_bottom)
    t0 = numpy_trajectory(theta0, seq, p, res, solver=solver_ref)
    frozen = None
    if params.freeze_mechanics:
        frozen = [solver_ref.solve(t0["dw"][:, k].reshape(res, res), p.c_cp)
                  for k in range(len(seq))]

    m = build_v2_model(theta, seq, p, res, f_bounds=(0.0, 1.5 * scale),
                       freeze_mechanics=params.freeze_mechanics, frozen_velocity=frozen)
    # Start on a trajectory that actually satisfies the dynamics, so IPOPT begins feasible in
    # every constraint and only the fit is wrong.  v1 starts every pixel at 0.01, which violates
    # its own dynamic constraints from iteration zero.
    pin_model(m, t0, fix=False)
    for q in m.PIX:
        m.Q[q, 0].fix(0.0)
    add_estimation_objective(m, y_true, params.tv_weight, scale)
    n_v = sum(1 for _ in m.component_data_objects(pyo.Var))
    n_c = sum(1 for _ in m.component_data_objects(pyo.Constraint, active=True))
    say("    %d variables, %d constraints%s\n"
        % (n_v, n_c, " (frozen mechanics)" if params.freeze_mechanics else ""))

    say("Solving...\n")
    r1, ls1 = solve_with_fallback(m, linear_solver=params.linear_solver,
                                  max_iter=params.ipopt_max_iter, log_callback=log_callback)
    status = str(r1.solver.termination_condition)
    say("    %s (%s)\n" % (status, ls1))

    # eq:xd_box active-set census, before anything else reads the solution.
    lo_b, hi_b = 0.0, 1.5 * scale
    _vals = [pyo.value(m.f[q, 0]) for q in m.PIX]
    n_lo = sum(1 for v in _vals if abs(v - lo_b) < 1e-8)
    n_hi = sum(1 for v in _vals if abs(v - hi_b) < 1e-8)
    say("    eq:xd_box active set on theta: %d at lower, %d at upper, %d interior\n"
        % (n_lo, n_hi, len(_vals) - n_lo - n_hi))

    theta_hat = np.array([pyo.value(m.f[q, 0]) for q in m.PIX]).reshape(res, res)
    f_hat = np.array([pyo.value(m.f[q, len(seq)]) for q in m.PIX]).reshape(res, res)
    Q_hat = np.array([pyo.value(m.Q[q, len(seq)]) for q in m.PIX]).reshape(res, res)
    resid = [pyo.value(m.S[k, j, n]) - float(y_true[k][j]) for (k, j, n) in m.obs_index]

    out = V2UQResults(
        theta_true=theta, theta_hat=theta_hat, f_final_true=f_true, f_final_hat=f_hat,
        Q_final_hat=Q_hat, inverse_status=status, inverse_linear_solver=ls1,
        continuation_status=cont_status,
        courant=float(max(i.courant for i in infos)),
        rg_pct=_rg_pct(theta, f_true),
        obs_rms=float(np.sqrt(np.mean(np.square(resid)))),
        theta_rms=float(np.sqrt(np.mean((theta_hat - theta) ** 2))),
        n_theta_at_lower=n_lo, n_theta_at_upper=n_hi,
        n_theta_interior=len(_vals) - n_lo - n_hi,
        forward_residual=fwd_resid, n_measurements=len(seq), n_rays=n_rays,
        n_vars=n_v, n_cons=n_c,
        mass_true=float(f_true.sum()), mass_hat=float(f_hat.sum()))

    # --- k_aug: d(theta)/d(y), eq:xd_composed_jacobian -----------------------------------
    if params.run_uq:
        try:
            from senDOE.helpers.statistics import d_optimality
            from senDOE.sensitivity.pyomo_sensitivity import extract_sensitivity_matrix
            say("Extracting d(theta)/d(y) with k_aug...\n")
            J = extract_sensitivity_matrix(
                model=m,
                var_list=[m.f[q, 0] for q in m.PIX],
                param_list=[m.y_data[k, j] for (k, j, _n) in m.obs_index],
                mode="k_aug", return_type="dense")
            J = np.asarray(J, dtype=float)
            if not np.all(np.isfinite(J)):
                raise ValueError("k_aug returned a non-finite sensitivity matrix")
            # k_aug prints "Could not fix the accuracy of the problem ... results might be
            # incorrect" when it cannot drive the KKT residual ratio below 1e-10, which this
            # model routinely trips.  That is a caveat on the covariance, not a failure -- the
            # covariance here is rank deficient on purpose -- but it must not scroll past
            # unremarked, so it is carried on the result.
            out.uq_conditioning = float(np.linalg.cond(J @ J.T)) if J.shape[0] <= 2048 else float("nan")
            cov = params.noise_cov_scale * (J @ J.T)
            with np.errstate(divide="ignore", invalid="ignore"):
                out.log_cov_diag_2D = np.log10(np.diag(cov)).reshape(res, res)
            out.d_optimality = float(d_optimality(cov))
            say("    D-optimality %.6g\n" % out.d_optimality)
        except Exception as exc:
            # Non-fatal by design: the covariance here is intentionally rank deficient (a
            # starved geometry leaves pixels no ray constrains), and losing it must not lose
            # the reconstruction.  Same call the 3D slice loop makes.
            out.uq_error = "%s: %s" % (type(exc).__name__, str(exc).splitlines()[0][:200])
            say("    UQ failed (reconstruction kept): %s\n" % out.uq_error)
    return out


def _cli(argv=None):
    """``python3 degrade_v2_uq.py`` checks the model; ``--reconstruct`` runs one.

    The headless route exists because the exact coupling at the v2 tab's own grid is a long
    solve -- see CLAUDE.md for measured numbers -- and a browser session will not sit through
    it. Results land in an .npz the app does not need to be running to produce.
    """
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reconstruct", action="store_true",
                    help="run a reconstruction instead of the model check")
    ap.add_argument("--image-res", type=int, default=24)
    ap.add_argument("--n-steps", type=int, default=3)
    ap.add_argument("--I0", type=float, default=1.0)
    ap.add_argument("--c-cp", type=float, default=0.3)
    ap.add_argument("--tv-weight", type=float, default=0.05)
    ap.add_argument("--eps-rel", type=float, default=1e-3)
    ap.add_argument("--freeze-mechanics", action="store_true")
    ap.add_argument("--no-uq", action="store_true")
    ap.add_argument("--max-iter", type=int, default=3000)
    ap.add_argument("-o", "--out", default=None, help="write results to this .npz")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)

    if not a.reconstruct:
        check_forward(image_res=a.image_res, n_steps=a.n_steps, c_cp=a.c_cp,
                      eps_rel=a.eps_rel, verbose=not a.quiet)
        return 0

    import time
    params = V2UQParams(
        image_res=a.image_res,
        beam_steps=tuple((180.0 * i / a.n_steps, 0.0, 0) for i in range(a.n_steps)),
        I0=a.I0, c_cp=a.c_cp, tv_weight=a.tv_weight, eps_rel=a.eps_rel,
        freeze_mechanics=a.freeze_mechanics, run_uq=not a.no_uq, ipopt_max_iter=a.max_iter)
    t0 = time.time()
    cb = None if a.quiet else (lambda chunk: (sys.stdout.write(chunk), sys.stdout.flush()))
    r = run_v2_reconstruction(params, log_callback=cb)
    dt = time.time() - t0
    print("\n%dx%d, %d measurements, %s coupling, %.1f s"
          % (a.image_res, a.image_res, a.n_steps,
             "frozen" if a.freeze_mechanics else "exact", dt))
    print("  model vs simulator  %.2e" % r.forward_residual)
    print("  inverse             %s (%s), continuation %s"
          % (r.inverse_status, r.inverse_linear_solver, r.continuation_status))
    print("  size                %d vars / %d cons" % (r.n_vars, r.n_cons))
    print("  fit RMS             %.4e" % r.obs_rms)
    print("  theta RMS error     %.4e  (%.2f%% of peak)"
          % (r.theta_rms, 100.0 * r.theta_rms / float(r.theta_true.max())))
    print("  transport           Courant %.3f, Rg %+.2f%%, mass left %.4f"
          % (r.courant, r.rg_pct, r.mass_true / float(r.theta_true.sum())))
    print("  D-optimality        %s%s"
          % (r.d_optimality, "" if not r.uq_error else "   UQ failed: " + r.uq_error))
    if a.out:
        np.savez_compressed(
            a.out, theta_true=r.theta_true, theta_hat=r.theta_hat,
            f_final_true=r.f_final_true, f_final_hat=r.f_final_hat, Q_final_hat=r.Q_final_hat,
            log_cov_diag_2D=(r.log_cov_diag_2D if r.log_cov_diag_2D is not None
                             else np.zeros(0)),
            d_optimality=r.d_optimality, obs_rms=r.obs_rms, theta_rms=r.theta_rms,
            seconds=dt)
        print("  wrote               %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
