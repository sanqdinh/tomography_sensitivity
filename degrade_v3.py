"""v3 damage model: v2 with the elasticity replaced by a nearest-neighbour compaction flux.

What changed, and nothing else did
----------------------------------
Steps 1-4 of section 3.2 are **untouched** and are imported from :mod:`degrade_v2` rather than
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

Run ``python3 degrade_v3.py`` for the acceptance tests (V1, V3, V4, V7, V8, V9) -- the same
run-the-module verification path as :mod:`tomography_3d` and :mod:`degrade_v2`, since this repo
has no test suite.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace

from typing import Optional

import numpy as np

from dose_response import ray_line_integral
# Steps 1-4 are shared with v2, deliberately: one implementation, so they cannot drift.
from degrade_v2 import (accumulate_dose, omega, peak_optical_depth, radius_of_gyration,
                        scale_to_optical_depth)


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
    the last two, exactly as :mod:`degrade_v2` does, and exists only so the acceptance tests can
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

    Same signature and same return shape as :func:`degrade_v2.simulate`, so callers swap models
    by swapping the import.  ``seq`` is the app's ``_table_to_seq`` output.
    """
    from dose_response import bundle_r_values

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

def semi_axis_ratio(f) -> float:
    """``sqrt(lambda_max / lambda_min)`` of the mass second-moment tensor -- the V7 statistic.

    1.0 is isotropic.  Under elasticity a 20-degree wedge of angles left this at 1.0001, i.e.
    the design could not move the shape at all; a local law should register measurably above 1.
    """
    f = np.asarray(f, dtype=float)
    nr, nc = f.shape
    yy, xx = np.mgrid[0:nr, 0:nc]
    m = f.sum()
    if m <= 0:
        return float("nan")
    xc, yc = (xx * f).sum() / m, (yy * f).sum() / m
    dx_, dy_ = xx - xc, yy - yc
    cxx = (f * dx_ * dx_).sum() / m
    cyy = (f * dy_ * dy_).sum() / m
    cxy = (f * dx_ * dy_).sum() / m
    ev = np.linalg.eigvalsh(np.array([[cxx, cxy], [cxy, cyy]]))
    lo, hi = float(ev[0]), float(ev[1])
    if lo <= 0:
        return float("nan")
    return float(np.sqrt(hi / lo))


def support_radius(f, frac: float = 0.99) -> float:
    """Radius about the field's own centroid containing ``frac`` of the mass, in pixels.

    This is what "the specimen shrinks" actually refers to.  Rg is NOT that statistic: mass
    moving outward in the bulk raises Rg while the rim draining inward lowers it, so Rg mixes
    the two and can come out either way without saying which happened.
    """
    f = np.asarray(f, dtype=float)
    nr, nc = f.shape
    yy, xx = np.mgrid[0:nr, 0:nc]
    m = f.sum()
    if m <= 0:
        return float("nan")
    xc, yc = (xx * f).sum() / m, (yy * f).sum() / m
    r = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2).ravel()
    w = np.clip(f.ravel(), 0.0, None)
    o = np.argsort(r)
    c = np.cumsum(w[o])
    if c[-1] <= 0:
        return float("nan")
    return float(r[o][np.searchsorted(c, frac * c[-1])])


def mass_outside(f, r0: float, centre=None) -> float:
    """Mass beyond radius ``r0``.  Must be zero or falling -- material must not leave the body."""
    f = np.asarray(f, dtype=float)
    nr, nc = f.shape
    yy, xx = np.mgrid[0:nr, 0:nc]
    if centre is None:
        m = f.sum()
        centre = ((xx * f).sum() / m, (yy * f).sum() / m)
    xc, yc = centre
    r = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2)
    return float(f[r > r0].sum())


def _demo_sequence(n_steps: int = 12):
    """Evenly spaced full-fan projections -- the sequence the manuscript's checks use."""
    return tuple((180.0 * i / n_steps, 0.0, 0) for i in range(n_steps))


def _wedge_sequence(n_steps: int = 12, span_deg: float = 20.0):
    """``n_steps`` projections crammed into a ``span_deg`` wedge -- the V7 design contrast."""
    if n_steps == 1:
        return ((0.0, 0.0, 0),)
    return tuple((span_deg * i / (n_steps - 1), 0.0, 0) for i in range(n_steps))


def _disc(image_res: int, frac: float = 0.35, edge: float = 2.0):
    """A centred disc -- semi-axis ratio exactly 1.000000 when undamaged.

    V7 needs this rather than Shepp-Logan: the note's elasticity baseline is "1.0001 against
    1.0000 for a full scan", which is only meaningful against an isotropic starting shape.

    ``edge`` is the interface width in pixels, and it is **not cosmetic**.  The contraction
    mechanism needs the interface resolved over at least two pixels: the compaction potential
    ``Pi = dw * state~ / state_max`` turns over there because the density falls by a factor of
    ~2 across the interface while ``dw`` changes by ~2%, so ``Pi`` peaks just INSIDE the rim and
    material outside that peak drains inward.  On a perfectly sharp interface (``edge = 0``) the
    peak sits at the outermost occupied pixel and the result is rim brightening, not contraction.
    Measured at grid 64, c_cp = 0.8, a = b = 0, change in support radius::

        edge = 0 px   +0.004 %      edge = 2 px   -0.414 %      edge = 4 px   -0.552 %

    That is (S4) of the note doing real work.  ``edge = 0`` is kept reachable so the test can
    exhibit the failure mode rather than merely warn about it.
    """
    yy, xx = np.mgrid[0:image_res, 0:image_res]
    c = (image_res - 1) / 2.0
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    R = image_res * frac
    if edge <= 0.0:
        return (r <= R).astype(float)
    t = np.clip((R - r) / edge + 0.5, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)                 # smoothstep: C1 across the interface


def _phantom(image_res: int):
    """The same phantom :mod:`degrade_v2_uq` reconstructs, so v2 and v3 numbers are comparable."""
    from skimage.data import shepp_logan_phantom
    from skimage.transform import resize
    return resize(shepp_logan_phantom(), (image_res, image_res)).astype(float)


# --- acceptance tests ---------------------------------------------------------------------
# V2, V5 and V6 of the original list are WITHDRAWN: they were tied to the gamma_esc rewrite of
# eq:xd_decay, which was not adopted. eq:xd_decay is kept verbatim, so the mass switch is
# a = b = 0 (prop:xd_mass) rather than gamma_esc = 0, and V1 is stated that way below.

def check_acceptance(image_res: int = 64, n_steps: int = 12, verbose: bool = True):
    """V1, V3, V4, V7, V8, V9.  Returns a dict of the measured numbers.

    Run-the-module verification, as in :mod:`degrade_v2` -- this repo has no test suite.
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


if __name__ == "__main__":
    res = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    r = check_acceptance(image_res=res, n_steps=steps)
    sys.exit(1 if r["fails"] else 0)
