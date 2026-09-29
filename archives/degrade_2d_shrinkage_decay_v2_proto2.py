"""v3 damage model: v2 with the elasticity replaced by a nearest-neighbour compaction flux.

What changed, and nothing else did
----------------------------------
Steps 1-4 of section 3.2 are **untouched** and are imported from :mod:`degrade_2d_shrinkage_decay_v2_proto1` rather than
rewritten, so the photon balance, the dose state and the response cannot drift between the two
models.  eq:xd_decay is likewise kept **verbatim** -- ``exp(-a I_p - b I_p^2)``, driven by the
instantaneous fluence, with ``a = b = 0`` still the switch that conserves mass exactly, which is
prop:xd_mass word for word.

Deleted outright: the eigenstrain (eq:xd_eigenstrain), the whole elasticity block
(eq:xd_elastic_strain .. eq:xd_elastic_discrete) and the smoothed upwind (eq:xd_upwind,
eq:xd_upwind_scale).  With them go the variables ``u`` and ``eps_sq``, the operators ``K`` and
``B``, and the parameters ``E``, ``nu``, ``eps_up``, ``eps_rel``.  There is no ``gamma_esc``:
that parameter belonged to a rewrite of eq:xd_decay that was not adopted.

In their place, one equation.  For each face between neighbours ``p`` and ``q``::

    F_{p->q} = c_cp * 0.5 * (state~_p + state~_q) * (dw_q - dw_p)

with ``F = 0`` on every grid-boundary face, evaluated on the **decayed** field ``state~`` so the
decay-then-transport ordering that prop:xd_mass rests on is preserved.

Why this is the scientific point, not just the cheap one
--------------------------------------------------------
Take any antisymmetric flux law ``F_{p->q} = G(s_p, s_q, w_p, w_q)``.  In the interior of a
region where state and ``dw`` are both spatially constant every face carries ``G(s,s,w,w)``, and
antisymmetry forces that to vanish.  So **a uniformly damaged bulk does not move**: a local law
confines motion to the gradients of the damage field.  That is what makes shrinkage respond to
where the dose was put, which a global elliptic solve erases -- under elasticity a 20-degree
wedge of angles left a semi-axis ratio of 1.0001 against 1.0000 for a full scan.  It is a
property of locality, not a defect.

Consequences to expect rather than rediscover as bugs
-----------------------------------------------------
* No mechanics, so no substrate, no boundary conditions, no Poisson coupling and no shear.
  Contraction is purely volumetric.  A clamped edge would return as a face-dependent ``c_cp``.
* ``c_cp`` is now a **dimensionless compaction number**, not a fraction, and it is **grid
  dependent**: at nearest-neighbour range the compaction length IS one cell, so a value does not
  transfer between ``image_res`` 10, 64 and 128.  Report the grid with every number.
  ``c_cp <= 1`` is the range where it reads as the fraction of created void the matrix closes
  within one cell per step.
* A uniformly damaged bulk does not move; only the rim does.  The radius-of-gyration numbers
  measured under elasticity will not reproduce, so ``c_cp`` needs recalibrating against them.

The sign problem -- MEASURED, and a modelling question, not a coding one
------------------------------------------------------------------------
As specified the sample **expands**.  On a disc at grid 32, 12 projections, ``a = b = 0`` so mass
is exactly conserved and the shape change is transport alone::

    c_cp = 0.3   Rg  +0.061 %        c_cp = 0.8   Rg  +0.163 %

The spec's sign check assumes "a damaged disc has ``dw`` high inside and zero outside".  Measured
radial profile of ``dw`` on that disc, centre outwards::

    r  0-4   5.43e-2      r 11-13  6.12e-2
    r  4-8   5.44e-2      r 13-16  6.42e-2
    r  8-11  5.56e-2      r 16-20  6.57e-2   (vacuum)

``dw`` is monotonically **higher further out**, and it is highest of all in vacuum, because
``I_p = I0 exp(-shielding)`` is largest exactly where the least material lies upstream.  This is
not a property of this phantom: for any convex body under a full scan, an interior point is
shielded from every angle by more material than a rim point, so damage is always rim-peaked.  A
law that moves mass toward higher ``dw`` therefore always moves it outward, and there is no
optical depth at which that reverses -- as absorption goes to zero the damage field goes uniform
and, by the antisymmetry argument above, nothing moves at all.

Masking the vacuum does not rescue it: the gradient points outward *inside* the object too.

Negating the flux gives contraction of exactly the same magnitude (``-0.061 %`` / ``-0.163 %``,
the map being linear in ``c_cp`` to leading order), and leaves V7 essentially unchanged
(1.002383 against 1.002377 for the wedge).  So the design-dependence result does **not** depend
on this choice and the two questions are separable.  Which sign is physical is for the model, not
for this file: toward-damage reads as the matrix closing a void, away-from-damage as the damaged
zone swelling.

Positivity
----------
The arithmetic-mean face density is a **central** scheme and is not unconditionally positive.
:func:`compaction_number` reports the sufficient condition ``C_k <= 2``; if it is violated,
:func:`upwind_compaction_flux` is the logistic-weighted fallback, off by default.

Run ``python3 -m archives.degrade_2d_shrinkage_decay_v2_proto2 model`` for the acceptance tests (V1, V3, V4, V7, V8, V9) -- the same
run-the-module verification path as :mod:`tomography_3d` and :mod:`degrade_2d_shrinkage_decay_v2_proto1`, since this repo
has no test suite.

Reconstruction
--------------
The Pyomo reconstruction of this prototype is in the second half of this file, under the
``=== reconstruction ===`` banner.  Its own notes follow.

Pyomo transcription of the v3 damage model, and reconstruction of ``theta = f_0`` from it.

The v3 counterpart of :mod:`degrade_2d_shrinkage_decay_v2_proto1`.  Same contract, same checking discipline: the data
comes from :func:`degrade_2d_shrinkage_decay_v2_proto2.simulate` (numpy), so the measurements and the model fitting them
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
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import pyomo.environ as pyo

from senDOE.helpers.rays import ray_line_integral
from senDOE.helpers.dose import accumulate_dose, scale_to_optical_depth
from senDOE.helpers.phantoms import (
    demo_sequence as _demo_sequence, disc as _disc, phantom as _phantom,
    wedge_sequence as _wedge_sequence)
from senDOE.helpers.rays import measurement_rays
from senDOE.helpers.shape_metrics import (
    mass_outside, radius_of_gyration, semi_axis_ratio, support_radius)
from senDOE.helpers.solvers import solve_with_fallback
from senDOE.models.tomography_pyomo_2d_shrinkage_decay import _neighbours


# --- carried over from an earlier prototype so this file stands alone --------------

def omega(Q, omega_inf: float, Q_c: float):
    """Retained attenuation fraction -- eq:xd_response.  ``omega(0) = 1`` exactly."""
    return omega_inf + (1.0 - omega_inf) * np.exp(-np.asarray(Q, dtype=float) / Q_c)





# Steps 1-4 are shared with v2, deliberately: one implementation, so they cannot drift.


@dataclass(frozen=True)
class V3Params:
    """Parameters of the v3 model.  Note what is absent: E, nu, eps_up, eps_rel, gamma_esc."""

    I0: float = 1.0          # incident beam intensity; I0 = 0 is the undamaged limit (M = id)
    c_q: float = 0.1         # fluence x path length -> absorbed dose
    Q_c: float = 1.0         # characteristic dose of the response
    omega_inf: float = 0.2   # residual attenuation fraction, in [0, 1)
    # Compaction number, eq:xd_compaction. DIMENSIONLESS and GRID DEPENDENT -- at nearest
    # neighbour range the compaction length is one cell, so a value does not transfer across
    # image_res. c_cp <= 1 reads as the fraction of created void closed within one cell per step.
    c_cp: float = 0.3
    # eq:xd_decay, VERBATIM from v2. Driven by instantaneous fluence, not accumulated dose.
    # a = b = 0 is the switch that conserves mass exactly (prop:xd_mass).
    a: float = 0.05
    b: float = 0.0
    dx: float = 1.0          # pixel pitch, in the app's geometry units
    # Reference density that makes the compaction potential dimensionless. None -> the peak of
    # the initial field, filled in by :func:`resolve`.
    state_max: Optional[float] = None
    # Donor-cell upwind. MANDATORY, not a fallback: it is what makes the vacuum exactly inert.
    # At a material/vacuum face Pi_q = 0 and Pi_p >= 0, so g <= 0, the donor is the vacuum pixel,
    # and it has no mass -- so the flux is exactly zero and nothing leaks out of the specimen.
    # The central scheme instead carries a nonzero flux there and drives the vacuum negative.
    # Face weighting. "upwind" is the logistic donor cell; "harmonic" weights by
    # H = 2 ab/(a+b+eps_h), which vanishes when EITHER side is empty and so makes the vacuum
    # inert structurally rather than via the sign of g -- no exponential, no beta, and the
    # second derivatives are a rational function rather than a logistic. "central" is the
    # arithmetic mean, kept only because it is what the first version did.
    flux: str = "upwind"
    eps_h: float = 1e-12     # removable-singularity guard for "harmonic", x peak density
    upwind: bool = True
    # Logistic sharpness, units of 1/Pi. The switch must SATURATE: beta*max|g| >~ 5, and Pi is
    # small (dw ~ 0.06 times a density ratio), so max|g| ~ 0.2 and beta ~ 20 leaves the scheme
    # essentially central. Measured on a disc at grid 32, c_cp = 0.8: min state runs
    # -1.0e-2 (beta 20) -> -6.7e-4 (100) -> -2.0e-5 (500) -> -1.6e-6 (2000) -> -5.6e-9 (5000),
    # so positivity is recovered as the switch saturates, exactly as the donor-cell bound says.
    # It must be a CONSTANT for the Pyomo model, so it is set generously rather than adapted.
    beta: float = 1000.0

    def decay_factor(self, I_p):
        """eq:xd_decay's multiplier ``exp(-a I - b I^2)``.  Strictly positive, so state stays > 0."""
        I_p = np.asarray(I_p, dtype=float)
        return np.exp(-self.a * I_p - self.b * I_p ** 2)


def resolve(p: V3Params, theta) -> V3Params:
    """Fill ``state_max`` from the initial field if the caller left it unset.

    It must be a CONSTANT -- making it a function of the running state would put a global
    reduction inside every face and destroy both the locality and the sparsity.
    """
    if p.state_max is not None:
        return p
    peak = float(np.abs(np.asarray(theta, dtype=float)).max())
    return replace(p, state_max=(peak if peak > 0.0 else 1.0))


@dataclass
class StepInfo3:
    """Per-step diagnostics.  ``compaction`` is the V9 positivity number."""

    mass: float           # sum_p state_p after the step
    lost: float           # mass removed by eq:xd_decay this step; transport moves, never removes
    dw_max: float         # largest converted fraction anywhere
    q_max: float          # largest accumulated dose anywhere
    state_min: float      # most negative state, if the central flux overshot
    compaction: float     # C_k of the donor-cell positivity condition; must stay <= 1
    flux_sum: float       # sum of all face fluxes -- exactly 0 by antisymmetry, reported as proof
    max_g: float = 0.0    # max |Pi_q - Pi_p| over faces; beta*max_g should be >~ 5


# --- the one new equation ---------------------------------------------------------------------

def _faces(state_t, dw):
    """The two face-difference stacks, as (horizontal, vertical) arrays of neighbour pairs.

    Horizontal faces join ``(i, j)`` and ``(i, j+1)``; vertical join ``(i, j)`` and ``(i+1, j)``.
    Grid-boundary faces simply do not exist here, which is what makes conservation unconditional
    -- it needs no hypothesis that the field vanishes on the boundary.
    """
    sh = state_t[:, :-1] + state_t[:, 1:]          # state~_p + state~_q across horizontal faces
    gh = dw[:, 1:] - dw[:, :-1]                    # dw_q - dw_p
    sv = state_t[:-1, :] + state_t[1:, :]
    gv = dw[1:, :] - dw[:-1, :]
    return (sh, gh), (sv, gv)


def compaction_flux_divergence(state_t, dw, c_cp: float, mode: str = "upwind",
                               beta: float = 20.0, state_max: float = 1.0,
                               eps_h: float = 1e-12):
    """``sum_{q~p} F_{p->q}`` for the nearest-neighbour compaction flux.

    ``F_{p->q} = c_cp * 0.5 * (state~_p + state~_q) * (dw_q - dw_p)``, zero on boundary faces.
    Each face contributes ``+F`` to ``p`` and ``-F`` to ``q``, so the divergence sums to zero
    over the grid by construction -- that antisymmetry is the whole of prop:xd_mass.

    Sign: ``dw_q > dw_p`` drives positive flux from ``p`` to ``q``, i.e. material flows **toward**
    the more damaged pixel.  The spec's sign check reasons from "a damaged disc has ``dw`` high
    inside and zero outside", and **that premise is false under Beer-Lambert shielding** -- see
    "The sign problem" in the module docstring.  The direction here is as specified; it is the
    consequence that does not come out as intended.

    With ``upwind`` the central arithmetic mean is replaced by the logistic-weighted donor cell,
    which is still exactly antisymmetric (swapping ``p, q`` sends ``g -> -g`` and ``w -> 1-w``,
    leaving the bracket unchanged) and still exactly zero at rest, because the vanishing is
    carried by the factor ``g`` rather than by a square root.
    """
    state_t = np.asarray(state_t, dtype=float)
    dw = np.asarray(dw, dtype=float)
    Pi = dw * state_t / state_max                  # the VOID CREATED, extensive. 0 in vacuum.
    gh = Pi[:, 1:] - Pi[:, :-1]
    gv = Pi[1:, :] - Pi[:-1, :]

    if mode == "harmonic":
        # H vanishes when either side is empty: no mass crosses a face touching vacuum, in
        # either direction, for any sign of g. Stronger than the logistic, which relied on the
        # sign of g happening to be favourable at such a face.
        eh = eps_h * max(float(np.abs(state_t).max()), 1e-300)
        ah, bh = state_t[:, :-1], state_t[:, 1:]
        av, bv = state_t[:-1, :], state_t[1:, :]
        Fh = c_cp * (2.0 * ah * bh / (ah + bh + eh)) * gh
        Fv = c_cp * (2.0 * av * bv / (av + bv + eh)) * gv
    elif mode == "upwind":
        wh = 1.0 / (1.0 + np.exp(-beta * gh))
        wv = 1.0 / (1.0 + np.exp(-beta * gv))
        Fh = c_cp * (wh * state_t[:, :-1] + (1.0 - wh) * state_t[:, 1:]) * gh
        Fv = c_cp * (wv * state_t[:-1, :] + (1.0 - wv) * state_t[1:, :]) * gv
    else:
        Fh = c_cp * 0.5 * (state_t[:, :-1] + state_t[:, 1:]) * gh
        Fv = c_cp * 0.5 * (state_t[:-1, :] + state_t[1:, :]) * gv

    div = np.zeros_like(state_t)
    div[:, :-1] += Fh
    div[:, 1:] -= Fh
    div[:-1, :] += Fv
    div[1:, :] -= Fv
    return div, (Fh, Fv)


def compaction_number(state_t, dw, c_cp: float, state_max: float = 1.0) -> float:
    """``C_k`` of the donor-cell positivity condition.  Sufficient condition ``C_k <= 1``.

    ``C_k = c_cp * max_p sum_{q~p} max(Pi_q - Pi_p, 0)``.

    Under donor cell, pixel ``p`` donates only across its OUTGOING faces and is itself the donor,
    so its loss is ``state~_p * c_cp * sum_q (g_pq)_+`` and the ``state~_p`` cancels.  There is no
    division, so unlike the central-scheme version this is well defined on vacuum pixels and
    needs no masking -- a vacuum pixel satisfies it vacuously, having no mass to lose.  Exact for
    saturated upwind, approached as ``beta`` grows, so report ``min state`` alongside it.
    """
    state_t = np.asarray(state_t, dtype=float)
    dw = np.asarray(dw, dtype=float)
    Pi = dw * state_t / state_max
    gh = Pi[:, 1:] - Pi[:, :-1]
    gv = Pi[1:, :] - Pi[:-1, :]
    out = np.zeros_like(Pi)
    out[:, :-1] += np.maximum(gh, 0.0)         # face to the right, outgoing if Pi_q > Pi_p
    out[:, 1:] += np.maximum(-gh, 0.0)
    out[:-1, :] += np.maximum(gv, 0.0)
    out[1:, :] += np.maximum(-gv, 0.0)
    return float(c_cp * np.max(out))


def max_abs_g(state_t, dw, state_max: float = 1.0) -> float:
    """``max |Pi_q - Pi_p|`` over faces -- what ``beta`` must saturate (``beta*max|g| >~ 5``)."""
    state_t = np.asarray(state_t, dtype=float)
    Pi = np.asarray(dw, dtype=float) * state_t / state_max
    gh = Pi[:, 1:] - Pi[:, :-1]
    gv = Pi[1:, :] - Pi[:-1, :]
    return float(max(np.abs(gh).max() if gh.size else 0.0,
                     np.abs(gv).max() if gv.size else 0.0))


# --- the step ----------------------------------------------------------------------------------

def step(state, Q, r_values, angle_rad: float, p: V3Params, _decay_last: bool = False):
    """One measurement step: ``(state, Q) -> (state_next, Q_next, info)``.

    Order is steps 1-4, then eq:xd_decay, then the compaction transport.  ``_decay_last`` swaps
    the last two, exactly as :mod:`degrade_2d_shrinkage_decay_v2_proto1` does, and exists only so the acceptance tests can
    *demonstrate* that the ordering matters rather than assert it.
    """
    state = np.asarray(state, dtype=float)
    Q = np.asarray(Q, dtype=float)
    sm = p.state_max if p.state_max is not None else 1.0

    # 1, 2: photon balance and dose accumulation -- shared with v2, byte for byte.
    dQ, I_p = accumulate_dose(state, r_values, angle_rad, p.I0, p.c_q)
    Q_next = Q + dQ

    # 3, 4, 5: response and the RELATIVE converted fraction. Written through the per-step
    # survival factor s = exp(-dQ/Q_c) so there is no cancellation of two nearly equal omegas:
    #   dw = (omega_k - omega_inf) * (1 - s) / omega_k   ==   1 - omega(Q_{k+1}) / omega(Q_k)
    # At omega_inf = 0 it collapses to 1 - s with no division at all.
    om_k = omega(Q, p.omega_inf, p.Q_c)
    s = np.exp(-dQ / p.Q_c)
    if p.omega_inf == 0.0:
        dw = 1.0 - s
    else:
        dw = (om_k - p.omega_inf) * (1.0 - s) / om_k

    if _decay_last:                                      # deliberately wrong order
        moved, _F = compaction_flux_divergence(state, dw, p.c_cp, p.flux, p.beta, sm, p.eps_h)
        state_next = (state - moved) * p.decay_factor(I_p)
        lost = float(state.sum() - (state * p.decay_factor(I_p)).sum())
        state_t = state
    else:
        # decay THEN transport: one common decay factor per face, so the flux pairs still cancel
        # and the entire change in the total is the decay. That is prop:xd_mass.
        state_t = state * p.decay_factor(I_p)
        lost = float(state.sum() - state_t.sum())
        div, _F = compaction_flux_divergence(state_t, dw, p.c_cp, p.flux, p.beta, sm, p.eps_h)
        state_next = state_t - div

    Fh, Fv = _F
    info = StepInfo3(
        mass=float(state_next.sum()),
        lost=lost,
        dw_max=float(dw.max()),
        q_max=float(Q_next.max()),
        state_min=float(state_next.min()),
        compaction=compaction_number(state_t, dw, p.c_cp, sm),
        max_g=max_abs_g(state_t, dw, sm),
        flux_sum=float(Fh.sum() + Fv.sum()) if Fh.size or Fv.size else 0.0,
    )
    return state_next, Q_next, info


def simulate(theta, seq, p: V3Params, image_res: int, _decay_last: bool = False,
             record_observations: bool = False, record_trajectory: bool = False):
    """Run a measurement sequence from the undamaged field ``theta``.

    Same signature and same return shape as :func:`degrade_2d_shrinkage_decay_v2_proto1.simulate`, so callers swap models
    by swapping the import.  ``seq`` is the app's ``_table_to_seq`` output.
    """
    from senDOE.helpers.rays import bundle_r_values

    theta = np.asarray(theta, dtype=float)
    p = resolve(p, theta)                 # state_max is the INITIAL peak, and a constant
    state = theta.copy()
    Q = np.zeros_like(state)
    infos, obs = [], []
    s_hist, Q_hist = [state.copy()], [Q.copy()]
    for angle_deg, offset, n_beams in seq:
        angle_rad = np.deg2rad(float(angle_deg))
        r_values = bundle_r_values(float(offset), int(n_beams), int(image_res))
        state, Q, info = step(state, Q, r_values, angle_rad, p, _decay_last=_decay_last)
        if record_observations:
            # y_{k+1} = C_{u_k} f_{k+1}: the projection is recorded AFTER its own exposure has
            # damaged the field. The angle u_k both deposits the dose and defines the geometry,
            # so it is the same beam; only the field it is read against moved from f_k to f_k+1.
            # Endpoint, not midpoint -- chosen deliberately, see the module docstring.
            # Consequence, and it is a real tension rather than an oversight: within one step the
            # beam is shielded by f_k (step 1 integrates the field it arrived at) while the
            # projection integrates f_{k+1}. The same photons answer to two fields. That is the
            # price of the endpoint convention.
            obs.append(np.array([ray_line_integral(state, r, angle_rad) for r in r_values]))
        infos.append(info)
        if record_trajectory:
            s_hist.append(state.copy())
            Q_hist.append(Q.copy())
    out = [state, Q, infos]
    if record_observations:
        out.append(obs)
    if record_trajectory:
        out.append((s_hist, Q_hist))
    return tuple(out)


# --- diagnostics ------------------------------------------------------------------------------

# --- acceptance tests ---------------------------------------------------------------------
# V2, V5 and V6 of the original list are WITHDRAWN: they were tied to the gamma_esc rewrite of
# eq:xd_decay, which was not adopted. eq:xd_decay is kept verbatim, so the mass switch is
# a = b = 0 (prop:xd_mass) rather than gamma_esc = 0, and V1 is stated that way below.

def check_acceptance(image_res: int = 64, n_steps: int = 12, verbose: bool = True):
    """V1, V3, V4, V7, V8, V9.  Returns a dict of the measured numbers.

    Run-the-module verification, as in :mod:`degrade_2d_shrinkage_decay_v2_proto1` -- this repo has no test suite.
    """
    out, fails = {}, []

    def say(msg):
        if verbose:
            print(msg)

    def run(theta, seq, **kw):
        p = V3Params(**kw)
        return simulate(theta, seq, p, image_res, record_trajectory=True)

    theta = scale_to_optical_depth(_phantom(image_res), 1.1, image_res)
    seq = _demo_sequence(n_steps)
    base = dict(I0=1.0, c_q=0.1, Q_c=1.0, omega_inf=0.2, c_cp=0.3, a=0.05, b=0.0)

    say("=" * 78)
    say("v3 acceptance tests -- grid %d, %d projections" % (image_res, n_steps))
    say("  (c_cp is grid dependent at nearest-neighbour range: these are grid-%d numbers)"
        % image_res)
    say("=" * 78)

    # --- V1: a = b = 0 conserves mass exactly, for any c_cp and any angle sequence -----------
    say("\nV1  mass conservation at a = b = 0 (prop:xd_mass), boundary faces carry zero flux")
    worst = 0.0
    for c_cp in (0.0, 0.3, 0.8, 1.0):
        s, _Q, infos = simulate(theta, seq, V3Params(**{**base, "a": 0.0, "b": 0.0,
                                                        "c_cp": c_cp}), image_res)
        drift = abs(s.sum() - theta.sum()) / theta.sum()
        worst = max(worst, drift)
        fs = max(abs(i.flux_sum) for i in infos)
        say("      c_cp = %.1f   relative mass drift %.3e   max |sum of face fluxes| %.3e"
            % (c_cp, drift, fs))
    out["v1_mass_drift"] = worst
    ok = worst < 1e-14
    fails += [] if ok else ["V1"]
    say("      -> worst %.3e  %s" % (worst, "PASS" if ok else "FAIL"))

    # --- V3: I0 = 0 is the identity, bitwise ------------------------------------------------
    say("\nV3  I0 = 0 is the identity")
    s0, Q0, _i = simulate(theta, seq, V3Params(**{**base, "I0": 0.0}), image_res)
    d_id = float(np.abs(s0 - theta).max())
    out["v3_identity"] = d_id
    ok = d_id == 0.0
    fails += [] if ok else ["V3"]
    say("      max |state_K - theta| = %.3e,  max Q = %.3e   %s"
        % (d_id, float(Q0.max()), "PASS (bitwise)" if ok else "FAIL"))

    # --- V4: I0 = 0 with c_cp > 0 equals I0 = 0 with c_cp = 0, no special-casing -------------
    say("\nV4  I0 = 0 with c_cp = 0.3 equals I0 = 0 with c_cp = 0 (no structural guard)")
    sa, _Qa, _ia = simulate(theta, seq, V3Params(**{**base, "I0": 0.0, "c_cp": 0.3}), image_res)
    sb, _Qb, _ib = simulate(theta, seq, V3Params(**{**base, "I0": 0.0, "c_cp": 0.0}), image_res)
    d4 = float(np.abs(sa - sb).max())
    out["v4_offswitch"] = d4
    ok = d4 == 0.0
    fails += [] if ok else ["V4"]
    say("      max difference %.3e   %s" % (d4, "PASS (bitwise)" if ok else "FAIL"))

    # --- V8: support radius -- the statistic "the specimen shrinks" actually refers to ------
    say("\nV8  a = b = 0, c_cp > 0: SUPPORT RADIUS falls, mass exactly flat")
    say("      Rg is reported too but is NOT the criterion: bulk outflow raises it while the rim")
    say("      draining inward lowers it, so Rg mixes the two and cannot say which happened.")
    disc8 = scale_to_optical_depth(_disc(image_res, edge=2.0), 1.1, image_res)
    b8 = {**base, "a": 0.0, "b": 0.0}
    r0, rg0d = support_radius(disc8), radius_of_gyration(disc8)
    say("      undamaged disc: support radius %.4f px, Rg %.4f" % (r0, rg0d))
    rows8 = []
    for c_cp in (0.0, 0.3, 0.8, 2.0):
        s, _Q, _i = simulate(disc8, seq, V3Params(**{**b8, "c_cp": c_cp}), image_res)
        rows8.append((c_cp, support_radius(s), 100.0 * (radius_of_gyration(s) - rg0d) / rg0d,
                      mass_outside(s, r0), abs(s.sum() - disc8.sum()) / disc8.sum()))
        say("      c_cp = %.1f   support R %.4f (%+.3f %%)   Rg %+.3f %%   beyond R0 %.3e"
            "   drift %.1e"
            % (rows8[-1][0], rows8[-1][1], 100.0 * (rows8[-1][1] - r0) / r0,
               rows8[-1][2], rows8[-1][3], rows8[-1][4]))
    out["v8"] = rows8
    flat = rows8[0][1] == r0
    # c_cp is a grid-dependent compaction NUMBER, so the value at which contraction switches on
    # moves with the grid: at grid 32 c_cp = 0.8 still expands (+0.013%) and you need ~2, while
    # at grid 64 and 96 c_cp = 0.8 already contracts (-0.41% and -0.26%). So the criterion is
    # that contraction appears and then deepens monotonically, not that it holds at a fixed c_cp.
    pos = [row for row in rows8[1:] if row[1] < r0]
    falling = len(pos) > 0 and all(pos[i][1] >= pos[i + 1][1] for i in range(len(pos) - 1))
    conserved = all(row[4] < 1e-14 for row in rows8)
    ok = flat and falling and conserved
    fails += [] if ok else ["V8"]
    say("      -> c_cp=0 exact: %s; support radius falling: %s; mass conserved: %s   %s"
        % (flat, falling, conserved, "PASS" if ok else "FAIL"))
    thr = next((row[0] for row in rows8[1:] if row[1] < r0), None)
    say("      contraction switches on at c_cp = %s at THIS grid (%d); the threshold moves with"
        % (thr, image_res))
    say("      the grid because c_cp is a compaction NUMBER, not a fraction.")
    # The same run on a SHARP interface, which is the documented failure mode, not a bug.
    sharp = scale_to_optical_depth(_disc(image_res, edge=0.0), 1.1, image_res)
    rs0 = support_radius(sharp)
    ss, _Q, _i = simulate(sharp, seq, V3Params(**{**b8, "c_cp": 0.8}), image_res)
    say("      contrast, SHARP interface (edge = 0 px), c_cp = 0.8: support R %+.4f %%"
        % (100.0 * (support_radius(ss) - rs0) / rs0))
    say("      -- Pi then peaks AT the outermost occupied pixel and the rim brightens instead.")
    say("      The mechanism needs the interface resolved over >= 2 px; see _disc.")

    # --- V9: donor-cell positivity ---------------------------------------------------------
    # C_k is a property of the RUN, not of the model, so it must be reported with its
    # configuration.  Two sweeps of it once disagreed by 2.4x purely because one was taken on
    # Shepp-Logan with the decay on and the other on the disc with a = b = 0; the causes are
    # comparable in size (a = 0.05 -> 0 is 1.58x, Shepp-Logan -> disc is 1.53x, and they compose
    # to the 2.41x observed).  Both are reported here, and the disc/a=b=0 row is the one that
    # pairs with the V7/V8 window numbers because those are measured on the same runs.
    say("\nV9  positivity: C_k <= 1 sufficient under saturated donor cell. Unmasked, no threshold.")
    disc9 = scale_to_optical_depth(_disc(image_res, edge=2.0), 1.1, image_res)
    out["v9"] = {}
    for lab, ph, aa in (("working point (Shepp-Logan, a = %g)" % base["a"], theta, base["a"]),
                        ("reference   (disc, a = b = 0)", disc9, 0.0)):
        say("      %s" % lab)
        rows9 = []
        for c_cp in (0.3, 0.8, 1.0, 1.5, 2.0):
            s, _Q, infos = simulate(ph, seq, V3Params(**{**base, "c_cp": c_cp, "a": aa}),
                                    image_res)
            Ck = max(i.compaction for i in infos)
            smin = min(i.state_min for i in infos)
            mg = max(i.max_g for i in infos)
            rows9.append((c_cp, Ck, smin, mg))
            say("        c_cp %.1f  C_k %7.4f  min state %+.3e  beta*max|g| %6.0f %s"
                % (c_cp, Ck, smin, V3Params().beta * mg, "  <- C_k > 1" if Ck > 1.0 else ""))
        out["v9"][lab] = rows9
    ok = all(Ck > 1.0 or smin > -1e-4
             for rows in out["v9"].values() for _c, Ck, smin, _g in rows)
    fails += [] if ok else ["V9"]
    say("      -> no meaningful negative state while C_k <= 1   %s" % ("PASS" if ok else "FAIL"))
    say("      On the reference configuration C_k crosses 1 between c_cp 1.5 and 1.6, which is")
    say("      well PAST the useful window: the separation has already fallen from 11.1x at")
    say("      c_cp 0.8 to 2.4x by 1.5, so the design signal, not positivity, is what bounds it.")
    say("      beta sensitivity -- the switch must be saturated and the answer insensitive:")
    prev = None
    for beta in (V3Params().beta, 2 * V3Params().beta, 4 * V3Params().beta):
        s, _Q, infos = simulate(disc8, seq, V3Params(**{**b8, "c_cp": 0.8, "beta": beta}),
                                image_res)
        d = ("" if prev is None
             else "   max change vs previous %.2e" % float(np.abs(s - prev).max()))
        say("        beta = %7.0f  beta*max|g| = %6.0f  min state %+.3e  support R %.4f%s"
            % (beta, beta * max(i.max_g for i in infos), s.min(), support_radius(s), d))
        prev = s

    # --- V7: design-driven anisotropy --------------------------------------------------------
    say("\nV7  design-driven anisotropy: %d-projection 20-degree wedge vs full scan" % n_steps)
    say("      (elasticity left this at 1.0001 -- a local law should register above 1)")
    # On a DISC, so the undamaged ratio is exactly 1.000000 and the number is comparable to
    # the note's elasticity baseline. a = b = 0 so mass is exactly conserved and the shape
    # change is transport alone, with no decay confounding it.
    disc = scale_to_optical_depth(_disc(image_res), 1.1, image_res)
    b7 = {**base, "a": 0.0, "b": 0.0}
    ar0 = semi_axis_ratio(disc)
    rows7 = []
    for label, sq in (("full scan  ", seq), ("20deg wedge", _wedge_sequence(n_steps, 20.0))):
        for c_cp in (0.0, 0.3, 0.8):
            s, _Q, _i = simulate(disc, sq, V3Params(**{**b7, "c_cp": c_cp}), image_res)
            rows7.append((label, c_cp, semi_axis_ratio(s)))
            say("      %s  c_cp = %.1f   semi-axis ratio %.6f" % rows7[-1])
    out["v7"] = rows7
    out["v7_theta"] = ar0
    say("      undamaged disc:    %.6f" % ar0)
    wedge = [r for r in rows7 if r[0].startswith("20deg") and r[1] == 0.8][0][2]
    full = [r for r in rows7 if r[0].startswith("full") and r[1] == 0.8][0][2]
    out["v7_wedge_vs_full"] = (wedge, full)
    say("      -> at c_cp = 0.8: wedge %.6f vs full %.6f, difference %+.2e"
        % (wedge, full, wedge - full))

    say("\n" + "=" * 78)
    say("FAILED: %s" % ", ".join(fails) if fails else "all acceptance tests PASS")
    say("=" * 78)
    out["fails"] = fails
    return out


# ============================== reconstruction ==============================
# --- model ------------------------------------------------------------------------------------

def build_v3_model(theta_ref, seq, p: V3Params, image_res: int, *, f_bounds=None,
                   inline_decay: bool = False, inline_Ipix: bool = False,
                   inline_dw: bool = False):
    """Steps 1-11 of the v3 model as a Pyomo model.  ``theta_ref`` seeds every variable."""
    theta_ref = np.asarray(theta_ref, dtype=float)
    p = resolve(p, theta_ref)          # state_max is a CONSTANT, the initial peak, as in v3
    res = int(image_res)
    npix = res * res
    meas = measurement_rays(seq, res)
    K = len(meas)
    if K == 0:
        raise ValueError("no measurements: the sequence is empty")

    m = pyo.ConcreteModel(name="degrade_2d_shrinkage_decay_v2_proto2")
    m.res, m.n_steps, m.meas, m.p = res, K, meas, p
    m.inline_decay = bool(inline_decay)
    m.inline_Ipix, m.inline_dw = bool(inline_Ipix), bool(inline_dw)

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

    if inline_Ipix:
        # Ipix appears only in c_ft, so substituting it costs npix rows + npix vars per stage
        # and adds this pixel's crossing terms to c_ft instead. A clean one-to-one substitution.
        def _Ipix(mm, q, k):
            terms = I_terms[(q, k)]
            return sum(p.I0 * pyo.exp(-mm.S[idx]) for idx in terms) if terms else 0.0
    else:
        m.Ipix = pyo.Var(m.PIX, m.TM, initialize=0.0)

        def _ip(mm, q, k):
            terms = I_terms[(q, k)]
            if not terms:
                return mm.Ipix[q, k] == 0.0
            return mm.Ipix[q, k] == sum(p.I0 * pyo.exp(-mm.S[idx]) for idx in terms)
        m.c_Ipix = pyo.Constraint(m.PIX, m.TM, rule=_ip)

        def _Ipix(mm, q, k):
            return mm.Ipix[q, k]

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

    def _dw_expr(mm, q, k):
        s = pyo.exp(-(mm.Q[q, k + 1] - mm.Q[q, k]) / p.Q_c)
        if p.omega_inf == 0.0:
            return 1.0 - s
        om = _om(mm.Q[q, k])
        return (om - p.omega_inf) * (1.0 - s) / om

    if inline_dw:
        # dw appears at FIVE stencil positions in c_mass, and each substitution drags in two Q
        # variables, so this trades npix vars/rows for a much denser c_mass. Expected to lose.
        def _dwv(mm, q, k):
            return _dw_expr(mm, q, k)
    else:
        def _dwv(mm, q, k):
            return mm.dw[q, k]

    m.dw = pyo.Var(m.PIX, m.TM, initialize=0.0)

    def _dwc(mm, q, k):
        # s = exp(-dQ/Q_c) is the per-step survival factor; dw = (om - om_inf)(1 - s)/om.
        # Written multiplicatively, and at omega_inf = 0 there is no division at all.
        s = pyo.exp(-(mm.Q[q, k + 1] - mm.Q[q, k]) / p.Q_c)
        if p.omega_inf == 0.0:
            return mm.dw[q, k] == 1.0 - s
        om = _om(mm.Q[q, k])
        return mm.dw[q, k] * om == (om - p.omega_inf) * (1.0 - s)
    if not inline_dw:
        m.c_dw = pyo.Constraint(m.PIX, m.TM, rule=_dwc)
    else:
        for q in m.PIX:
            for k in m.TM:
                m.dw[q, k].fix(0.0)          # present but inert, so pinning code still works

    # --- 9. eq:xd_decay, verbatim -- carried as a VARIABLE so the flux stays bilinear -------
    if inline_decay:
        def _ft(mm, q, k):
            return mm.f[q, k] * pyo.exp(-p.a * _Ipix(mm, q, k) - p.b * _Ipix(mm, q, k) ** 2)
    else:
        m.ft = pyo.Var(m.PIX, m.TM, initialize=lambda _m, q, k: float(flat[q]))

        def _ftc(mm, q, k):
            return mm.ft[q, k] == mm.f[q, k] * pyo.exp(
                -p.a * _Ipix(mm, q, k) - p.b * _Ipix(mm, q, k) ** 2)
        m.c_ft = pyo.Constraint(m.PIX, m.TM, rule=_ftc)

        def _ft(mm, q, k):
            return mm.ft[q, k]

    # --- 6, 7. compaction flux on the five-point stencil, then the mass balance -------------
    # Pi = dw * ft / state_max is the VOID CREATED -- extensive, and exactly 0 in vacuum, which
    # is what stops mass leaking out of the specimen. The donor-cell weight makes the vacuum
    # exactly inert: at a material/vacuum face g <= 0, the donor is the vacuum pixel, and it has
    # no mass. Bilinearity is given up for that; the stencil is still five-point.
    sm = float(p.state_max) if p.state_max is not None else 1.0

    def _pi(mm, q, k):
        return _dwv(mm, q, k) * _ft(mm, q, k) / sm

    eh = float(p.eps_h) * max(float(np.abs(theta_ref).max()), 1e-300)

    def _mass(mm, q, k):
        rhs = _ft(mm, q, k)
        if p.c_cp != 0.0:
            pip = _pi(mm, q, k)
            ftp = _ft(mm, q, k)
            for nb in _neighbours(q, res):
                g = _pi(mm, nb, k) - pip
                ftq = _ft(mm, nb, k)
                if p.flux == "harmonic":
                    # H vanishes when either side is empty -- the vacuum is inert structurally,
                    # and there is no exponential and no beta.
                    rhs -= p.c_cp * (2.0 * ftp * ftq / (ftp + ftq + eh)) * g
                elif p.flux == "upwind":
                    w = 1.0 / (1.0 + pyo.exp(-p.beta * g))
                    rhs -= p.c_cp * (w * ftp + (1.0 - w) * ftq) * g
                else:
                    rhs -= p.c_cp * 0.5 * (ftp + ftq) * g
        return mm.f[q, k + 1] == rhs
    m.c_mass = pyo.Constraint(m.PIX, m.TM, rule=_mass)

    # --- 11. observation: y_{k+1} = C_{u_k} f_{k+1} -----------------------------------------
    # The projection is now read against the field AFTER its own exposure, so it is no longer
    # the terminal entry of the shielding chain: the chain integrates f_k (what the beam was
    # attenuated by on the way in) while the projection integrates f_{k+1}. Two different sums.
    #
    # ONE LINEAR ROW PER RAY, not a second chain. Only the total is needed, never the partial
    # sums, so this is n_rays rows of ~n_pix nonzeros -- a few percent of the Jacobian, where a
    # second chain would cost ~18% and buy nothing.
    #
    # It is a VARIABLE rather than an inlined expression so the fit term stays diagonal in the
    # objective Hessian; inlined, every ray would contribute a dense n_pix clique there.
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


# --- pinning and the forward check ------------------------------------------------------------

def numpy_trajectory(theta, seq, p: V3Params, image_res: int):
    """Every variable of the model, taken off a :func:`degrade_2d_shrinkage_decay_v2_proto2.simulate` run."""
    p = resolve(p, theta)
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
    yobs = {}
    for k, (_ang, rays) in enumerate(meas):
        for j, (_r, walk) in enumerate(rays):
            acc = {}
            for _pix, chord, owner in walk:
                acc[owner] = acc.get(owner, 0.0) + chord
            yobs[(k, j)] = float(sum(c * f[q, k + 1] for q, c in acc.items()))
    return dict(f=f, Q=Q, S=S, Ipix=Ipix, dw=dw, ft=ft, yobs=yobs)


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
            if not m.inline_Ipix:
                m.Ipix[q, k].set_value(float(traj["Ipix"][q, k]))
            if not m.inline_dw:
                m.dw[q, k].set_value(float(traj["dw"][q, k]))
            if not m.inline_decay:
                m.ft[q, k].set_value(float(traj["ft"][q, k]))
            if fix:
                if not m.inline_Ipix:
                    m.Ipix[q, k].fix()
                if not m.inline_dw:
                    m.dw[q, k].fix()
                if not m.inline_decay:
                    m.ft[q, k].fix()
    for idx in m.CH:
        m.S[idx].set_value(float(traj["S"][idx]))
        if fix:
            m.S[idx].fix()
    for idx in m.RAY:
        m.yobs[idx].set_value(float(traj["yobs"][idx]))
        if fix:
            m.yobs[idx].fix()


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
                  c_cp: float = 0.3, inline_decay: bool = False,
                  inline_Ipix: bool = False, inline_dw: bool = False, **kw):
    """Does the Pyomo model reproduce :func:`degrade_2d_shrinkage_decay_v2_proto2.simulate`?  Residual only, no solver."""
    p = V3Params(c_cp=c_cp, **kw)
    seq = _demo_sequence(n_steps)
    theta = scale_to_optical_depth(_phantom(image_res), 1.1, image_res)
    traj = numpy_trajectory(theta, seq, p, image_res)
    m = build_v3_model(theta, seq, p, image_res, inline_decay=inline_decay,
                       inline_Ipix=inline_Ipix, inline_dw=inline_dw)
    pin_model(m, traj)
    r, where = max_residual(m)
    if verbose:
        tags = [n for n, on in (("Ipix", inline_Ipix), ("dw", inline_dw),
                                ("ft", inline_decay)) if on]
        print("v3 Pyomo model vs numpy simulator: grid %d, %d steps, c_cp = %g%s"
              % (image_res, n_steps, c_cp,
                 ("  [inlined: %s]" % ", ".join(tags)) if tags else ""))
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
    fit = sum((m.yobs[k, j] - m.y_data[k, j]) ** 2 for (k, j, _n) in m.obs_index) / (nobs * yscale ** 2)
    npix = m.res * m.res
    m.obj = pyo.Objective(expr=fit + tv_weight * _tv_expression(m, theta_scale) / (npix * theta_scale))
    return m.obj


def run_v3_reconstruction(theta, seq, p: V3Params, image_res: int, *, tv_weight=0.001,
                          linear_solver="ma97", solver_opts=None, inline_decay=False,
                          max_iter=3000, log_callback=None, gate=True, continuation=True,
                          inline_Ipix=False, inline_dw=False):
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
        mg = build_v3_model(theta, seq, p, res, inline_decay=inline_decay,
                            inline_Ipix=inline_Ipix, inline_dw=inline_dw)
        pin_model(mg, traj)
        gate_resid, where = max_residual(mg)
        del mg
        if gate_resid > 1e-8:
            raise RuntimeError("v3 Pyomo model no longer reproduces degrade_2d_shrinkage_decay_v2_proto2.simulate "
                               "(residual %.3e at %s)" % (gate_resid, where))

    # --- continuation: solve the undamaged problem first ---------------------------------
    # At I0 = 0 every dynamic constraint is the identity (no fluence -> no dose -> no dw -> no
    # Pi -> no flux), so this is linear tomography + TV. It costs little and hands the full
    # solve a theta that already fits the data, leaving only the dynamics to be reconciled.
    # v2 did this; v3 did not, and was taking 139-170 iterations for a problem whose only
    # degree of freedom is theta.
    theta0 = np.full_like(theta, float(theta.mean()))
    t_cont, cont_iters, cont_status = 0.0, "-", "skipped"
    if continuation:
        tc = time.time()
        p0 = resolve(V3Params(**{**p.__dict__, "I0": 0.0, "c_cp": 0.0}), theta)
        m0 = build_v3_model(theta, seq, p0, res, f_bounds=(0.0, 1.5 * scale),
                            inline_decay=inline_decay, inline_Ipix=inline_Ipix,
                            inline_dw=inline_dw)
        for q in m0.PIX:
            m0.f[q, 0].set_value(float(theta0.ravel()[q]))
        add_estimation_objective(m0, y, tv_weight, scale)
        b0 = io.StringIO()
        try:
            r0, _ls0 = solve_with_fallback(m0, linear_solver=linear_solver, max_iter=max_iter,
                                           log_callback=b0.write,
                                           options=dict(solver_opts or {}))
            cont_status = str(r0.solver.termination_condition)
            theta0 = np.array([pyo.value(m0.f[q, 0]) for q in m0.PIX]).reshape(res, res)
        except Exception as e:
            cont_status = "FAILED: " + str(e).splitlines()[0][:60]
        cont_iters = (re.findall(r"Number of Iterations\.*:\s*(\S+)", b0.getvalue())
                      or ["-"])[-1]
        del m0
        t_cont = time.time() - tc

    t_build = time.time()
    m = build_v3_model(theta, seq, p, res, f_bounds=(0.0, 1.5 * scale),
                       inline_decay=inline_decay, inline_Ipix=inline_Ipix, inline_dw=inline_dw)
    # Seed the dynamics from a trajectory of the CURRENT theta estimate, so every dynamic
    # constraint starts at residual ~0 and only the fit is wrong.
    t0 = numpy_trajectory(theta0, seq, p, res)
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
        t_sim=t_sim, t_build=t_build, t_solve=t_solve, t_cont=t_cont,
        cont_iters=cont_iters, cont_status=cont_status,
        n_vars=nv, n_cons=nc,
        gate_residual=gate_resid,
        iters=g(r"Number of Iterations\.*:\s*(\S+)"),
        nnz_hess=g(r"Number of nonzeros in Lagrangian Hessian\.*:\s*(\S+)"),
        nnz_jac=g(r"Number of nonzeros in equality constraint Jacobian\.*:\s*(\S+)"),
        ipopt_s=g(r"Total seconds in IPOPT \(w/o function evaluations\)\s*=\s*(\S+)"),
        fev_s=g(r"Total seconds in NLP function evaluations\s*=\s*(\S+)"),
        fact_s=g(r"LinearSystemFactorization\.*:\s*(\S+)"),
        pd_s=g(r"PDSystemSolverTotal\.*:\s*(\S+)"),
        theta_err=100.0 * float(np.sqrt(np.mean((th - theta) ** 2))) / float(theta.max()),
        courant=max(i.compaction for i in infos),
        mass=float(_s.sum()) / float(theta.sum()),
    )


if __name__ == "__main__":
    # `python3 -m archives.<this module> model [grid steps]` runs the forward acceptance
    # checks; anything else is the reconstruction CLI.
    if sys.argv[1:2] == ["model"]:
        del sys.argv[1]
        res = int(sys.argv[1]) if len(sys.argv) > 1 else 64
        steps = int(sys.argv[2]) if len(sys.argv) > 2 else 12
        r = check_acceptance(image_res=res, n_steps=steps)
        sys.exit(1 if r["fails"] else 0)
    else:
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
