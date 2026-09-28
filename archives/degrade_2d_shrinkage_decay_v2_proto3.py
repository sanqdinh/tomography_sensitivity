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

Reconstruction
--------------
The Pyomo reconstruction of this prototype is in the second half of this file, under the
``=== reconstruction ===`` banner.  Its own notes follow.

Pyomo transcription of the reduced (v4) damage model, and reconstruction of ``theta = f_0``.

The v4 counterpart of :mod:`degrade_2d_shrinkage_decay_v2_proto2`, with the dose block deleted.  Same checking
discipline: data comes from :func:`degrade_2d_shrinkage_decay_v2_proto3.simulate` (numpy), so the measurements and the
model fitting them stay two independent implementations, and :func:`check_forward` compares them
at the true solution before any reconstruction is allowed to mean anything.

What is gone relative to v3
---------------------------
``Q`` and its two constraint blocks.  In v3 the converted fraction needed a dose state carried
across steps (``c_dose`` defining ``Q[k+1]``, then ``c_dw`` relating two ``omega(Q)`` values).
Here ``dw`` is a direct function of this step's fluence, so it is ONE row off the shielding chain
and nothing crosses the step boundary but ``f``.

Blocks, per stage: ``S`` (chain), ``Ipix``, ``dw``, ``ft``, ``f``, plus ``yobs`` per ray.
"""

from __future__ import annotations

import io
import re
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyomo.environ as pyo

from senDOE.helpers.rays import bundle_r_values, ray_line_integral
from senDOE.helpers.dose import accumulate_dose, scale_to_optical_depth
from senDOE.helpers.phantoms import demo_sequence as _demo_sequence, phantom as _phantom
from senDOE.helpers.rays import measurement_rays
from senDOE.helpers.solvers import solve_with_fallback
from senDOE.models.tomography_pyomo_2d_shrinkage_decay import _neighbours


# --- carried over from an earlier prototype so this file stands alone --------------

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




# accumulate_dose with c_q = 1 returns (sum_r I_r*delta_r, sum_r I_r) -- the two fluence moments
# this model needs, on exactly the walk v2 and v3 use.
# The flux law is unchanged, so it is the same function, not a copy of it.


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


# ============================== reconstruction ==============================
def build_v4_model(theta_ref, seq, p: V4Params, image_res: int, *, f_bounds=None,
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

    m = pyo.ConcreteModel(name="degrade_2d_shrinkage_decay_v2_proto3")
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

    # --- 4, 5. compaction flux and the mass balance, unchanged from v3 ---------------------
    fm = float(p.f_max)
    eh = float(p.eps_h) * max(float(np.abs(theta_ref).max()), 1e-300)

    def _pi(mm, q, k):
        return _dwv(mm, q, k) * _ft(mm, q, k) / fm

    def _mass(mm, q, k):
        rhs = _ft(mm, q, k)
        if p.c_cp != 0.0:
            pip = _pi(mm, q, k)
            ftp = _ft(mm, q, k)
            for nb in _neighbours(q, res):
                g = _pi(mm, nb, k) - pip
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

def numpy_trajectory(theta, seq, p: V4Params, image_res: int):
    """Every variable of the model, taken off a :func:`degrade_2d_shrinkage_decay_v2_proto3.simulate` run."""
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
    yobs = {}
    for k, (_ang, rays) in enumerate(meas):
        for j, (_r, walk) in enumerate(rays):
            acc = {}
            for _pix, chord, owner in walk:
                acc[owner] = acc.get(owner, 0.0) + chord
            yobs[(k, j)] = float(sum(c * f[q, k + 1] for q, c in acc.items()))
    return dict(f=f, S=S, Ipix=Ipix, dw=dw, ft=ft, yobs=yobs)


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
            if fix:
                if not m.inline_Ipix:
                    m.Ipix[q, k].fix()
                if not m.inline_dw:
                    m.dw[q, k].fix()
                if not m.inline_ft:
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
    """G1: does the Pyomo model reproduce degrade_2d_shrinkage_decay_v2_proto3.simulate? Residual only, no solver."""
    inline = {k: kw.pop(k) for k in ("inline_Ipix", "inline_dw", "inline_ft") if k in kw}
    p = V4Params(**kw)
    seq = _demo_sequence(n_steps)
    theta = scale_to_optical_depth(_phantom(image_res), 1.1, image_res)
    traj = numpy_trajectory(theta, seq, p, image_res)
    m = build_v4_model(theta, seq, p, image_res, **inline)
    pin_model(m, traj)
    r, where = max_residual(m)
    if verbose:
        tags = ",".join(k.replace("inline_", "") for k, v in inline.items() if v) or "none"
        print("  v4 Pyomo vs numpy: grid %d, %d steps, c_cp=%g, inlined %s -> %.3e (%s)  %s"
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


def run_v4_reconstruction(theta, seq, p: V4Params, image_res: int, *, tv_weight=0.001,
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
        mg = build_v4_model(theta, seq, p, res, **inline)
        pin_model(mg, traj)
        gate_resid, where = max_residual(mg)
        del mg
        if gate_resid > 1e-8:
            raise RuntimeError("v4 Pyomo model no longer reproduces degrade_2d_shrinkage_decay_v2_proto3.simulate "
                               "(residual %.3e at %s)" % (gate_resid, where))

    theta0 = np.full_like(theta, float(theta.mean()))
    t_cont, cont_iters, cont_status = 0.0, "-", "skipped"
    if continuation:
        tc = time.time()
        from dataclasses import replace
        p0 = replace(p, I0=0.0, c_cp=0.0)
        m0 = build_v4_model(theta, seq, p0, res, f_bounds=(0.0, 1.5 * scale), **inline)
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
    m = build_v4_model(theta, seq, p, res, f_bounds=(0.0, 1.5 * scale), **inline)
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
