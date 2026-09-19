"""v4: the reduced damage model. v3 with the dose state removed.

What changed
------------
Setting the response floor to zero makes the accumulated dose cancel out of the algebra::

    dw = 1 - omega(Q_{k+1})/omega(Q_k)
       = 1 - exp(-(Q_{k+1} - Q_k)/Q_c)          (omega_inf = 0)
       = 1 - exp(-c * I_p * delta_p)            with c = c_q / Q_c

so ``dw`` depends on THIS exposure's fluence and nothing accumulated, and ``Q`` has nothing left
to do.  **The state is the single field f.**  Nothing else crosses a step boundary.

Deleted: ``Q``, ``omega(Q)`` and the saturating response, ``omega_inf``, ``Q_c``, ``c_q``, the
energy density ``f*Q``, the detector smoothing ``H`` (never implemented beyond the identity, so
nothing measured moves), and the dose budget.

Kept, and unchanged: the photon balance, the fluence-driven decay ``exp(-a I - b I^2)`` including
its quadratic term, the compaction flux with its logistic upwind, the mass balance, and the
index-shifted observation ``y_{k+1} = C f_{k+1}``.

``c`` IS INDEPENDENT OF ``a``
----------------------------
They are separate parameters and must stay separate.  Tying them would make ``a = 0`` kill the
compaction along with the decay, and ``a = b = 0`` with ``c_cp > 0`` is exactly the case that
demonstrates contraction at exactly conserved mass -- prop:xd_mass, and the requirement the whole
closure exists to satisfy.

Relationship to v3
------------------
This model is **bit-identical to v3 run at omega_inf = 0**, because the collapse above is an
identity rather than an approximation.  It is NOT identical to v3 at its default omega_inf = 0.2,
where ``dw`` is smaller by the factor ``(1 - omega_inf)`` at the first step; the v3 numbers quoted
elsewhere were measured at 0.2 and are a different configuration.  :func:`check_gates` verifies
the omega_inf = 0 equivalence directly.

Step 1 and the flux are IMPORTED from the v2/v3 modules rather than retyped, so the walk order,
the chord/deposit convention and the flux law cannot drift between models.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

from dose_response import bundle_r_values, ray_line_integral
# accumulate_dose with c_q = 1 returns (sum_r I_r*delta_r, sum_r I_r) -- the two fluence moments
# this model needs, on exactly the walk v2 and v3 use.
from degrade_v2 import accumulate_dose, scale_to_optical_depth
# The flux law is unchanged, so it is the same function, not a copy of it.
from degrade_v3 import (compaction_flux_divergence, compaction_number, max_abs_g,
                        radius_of_gyration, support_radius, mass_outside, semi_axis_ratio,
                        _disc, _phantom, _demo_sequence, _wedge_sequence)


@dataclass(frozen=True)
class V4Params:
    """Parameters of the reduced model.  Absent: Q_c, c_q, omega_inf."""

    I0: float = 1.0          # incident beam intensity; I0 = 0 is the undamaged limit
    # Converted fraction per unit fluence-path, dw = 1 - exp(-c * I_p * delta_p).
    # INDEPENDENT of a. c = c_q/Q_c from the v3 configuration it replaces.
    c: float = 0.1
    a: float = 0.05          # eq:xd_decay, linear in fluence
    b: float = 0.0           # quadratic term, retained; v3's value is 0.0 (see module notes)
    c_cp: float = 0.3        # compaction number, dimensionless and GRID DEPENDENT
    f_max: Optional[float] = None   # reference density; None -> the initial peak
    beta: float = 1000.0     # logistic sharpness; beta*max|g| must be >~ 5 to saturate
    flux: str = "upwind"     # "upwind" (default) | "harmonic" | "central"
    eps_h: float = 1e-12     # harmonic-mean singularity guard, x peak density
    dx: float = 1.0

    def decay_factor(self, I_p):
        """``exp(-a I - b I^2)``.  Strictly positive, so f stays > 0 under the decay alone."""
        I_p = np.asarray(I_p, dtype=float)
        return np.exp(-self.a * I_p - self.b * I_p ** 2)


@dataclass
class StepInfo4:
    """Per-step diagnostics."""

    mass: float           # sum_p f_p after the step
    lost: float           # mass removed by the decay this step
    dw_max: float         # largest converted fraction
    I_max: float          # largest local fluence
    state_min: float      # most negative f, if the flux overshot
    compaction: float     # C_k positivity number; sufficient condition C_k <= 1
    max_g: float          # max |Pi_q - Pi_p|; beta*max_g should be >~ 5
    flux_sum: float       # sum of all face fluxes -- exactly 0 by antisymmetry


def resolve(p: V4Params, theta) -> V4Params:
    """Fill ``f_max`` from the initial field if unset.  It must be a CONSTANT."""
    from dataclasses import replace
    if p.f_max is not None:
        return p
    peak = float(np.abs(np.asarray(theta, dtype=float)).max())
    return replace(p, f_max=(peak if peak > 0.0 else 1.0))


def step(f, r_values, angle_rad: float, p: V4Params):
    """One measurement step: ``f -> (f_next, info)``.  No dose state.

    Steps 1-5 of the reduced model.  The observation (step 6) is taken by :func:`simulate`
    AFTER this returns, on ``f_{k+1}``.
    """
    f = np.asarray(f, dtype=float)
    fm = p.f_max if p.f_max is not None else 1.0

    # 1. photon balance. Passing c as the accumulator's coefficient returns
    #    sum_r c * I_r * delta_r directly, on the same walk, chord convention and deposit index
    #    v2 and v3 use. The coefficient goes INSIDE the sum deliberately: v3 forms
    #    sum(c_q * I * delta) and float addition is not associative, so accumulating
    #    sum(I*delta) and multiplying afterwards would differ from v3 at ~1e-16 and cost the
    #    exact bit-identity that gate G4 is for. Same arithmetic, same order, same bits.
    cIdelta, I_p = accumulate_dose(f, r_values, angle_rad, p.I0, p.c)

    # 2. converted fraction, a direct function of THIS exposure. In [0, 1) since cIdelta >= 0.
    dw = 1.0 - np.exp(-cIdelta)

    # 3. mass loss, unchanged, driven by instantaneous fluence.
    ft = f * p.decay_factor(I_p)
    lost = float(f.sum() - ft.sum())

    # 4, 5. compaction flux on the five-point stencil, then the mass balance. Same function v3
    #       uses, so the flux law cannot drift.
    div, (Fh, Fv) = compaction_flux_divergence(ft, dw, p.c_cp, p.flux, p.beta, fm, p.eps_h)
    f_next = ft - div

    info = StepInfo4(
        mass=float(f_next.sum()), lost=lost, dw_max=float(dw.max()), I_max=float(I_p.max()),
        state_min=float(f_next.min()),
        compaction=compaction_number(ft, dw, p.c_cp, fm),
        max_g=max_abs_g(ft, dw, fm),
        flux_sum=float(Fh.sum() + Fv.sum()) if Fh.size or Fv.size else 0.0,
    )
    return f_next, info


def simulate(theta, seq, p: V4Params, image_res: int,
             record_observations: bool = False, record_trajectory: bool = False):
    """Run a measurement sequence from the undamaged field ``theta``.

    Observations are the index-shifted form, ``y_{k+1} = C_{u_k} f_{k+1}``: recorded AFTER each
    exposure has damaged the field, so ``f_0 = theta`` is never observed.
    """
    theta = np.asarray(theta, dtype=float)
    p = resolve(p, theta)
    f = theta.copy()
    infos, obs = [], []
    hist = [f.copy()]
    for angle_deg, offset, n_beams in seq:
        ang = np.deg2rad(float(angle_deg))
        r_values = bundle_r_values(float(offset), int(n_beams), int(image_res))
        f, info = step(f, r_values, ang, p)
        if record_observations:
            obs.append(np.array([ray_line_integral(f, r, ang) for r in r_values]))
        infos.append(info)
        if record_trajectory:
            hist.append(f.copy())
    out = [f, infos]
    if record_observations:
        out.append(obs)
    if record_trajectory:
        out.append(hist)
    return tuple(out)


def mass_audit(theta, seq, p: V4Params, image_res: int):
    """Audit the total against prop:xd_mass, in both the per-step and cumulative-product forms.

    The transport moves mass and never removes it, so the ONLY exact statement is per step::

        sum_p f_{k+1,p} == sum_p f_{k,p} * exp(-a I_{k,p} - b I_{k,p}^2)

    The cumulative form ``sum_p theta_p * prod_k exp(-a I - b I^2)`` is a DIFFERENT claim and is
    exact only when nothing moves.  With transport a parcel decays at pixel p on one step and at
    pixel q on the next, seeing two different fluences, so the per-pixel product does not follow
    the mass.  Both are returned so the gap can be reported rather than assumed away.
    """
    theta = np.asarray(theta, dtype=float)
    p = resolve(p, theta)
    f = theta.copy()
    acc = np.ones_like(f)
    worst_step = 0.0
    for angle_deg, offset, n_beams in seq:
        ang = np.deg2rad(float(angle_deg))
        rv = bundle_r_values(float(offset), int(n_beams), int(image_res))
        _Id, I_p = accumulate_dose(f, rv, ang, p.I0, 1.0)
        D = p.decay_factor(I_p)
        predicted = float((f * D).sum())          # the exact per-step statement
        acc = acc * D                             # the cumulative-product claim
        f, _info = step(f, rv, ang, p)
        worst_step = max(worst_step, abs(f.sum() - predicted) / max(abs(predicted), 1e-300))
    return dict(final=float(f.sum()),
                per_step_worst_rel=worst_step,
                cumulative_product=float((theta * acc).sum()),
                initial=float(theta.sum()))
