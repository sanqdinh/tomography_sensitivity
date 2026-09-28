"""v5, the shrinkage dose-response model: the compaction potential made nonlocal.

v4 with step 4 changed and nothing else moved.

The problem v5 solves
---------------------
v4's flux is driven by ``Pi``, a pointwise function of the local state.  prop:xd_locality then
bites: for an antisymmetric ``F_{p->q} = G(f_p, f_q, dw_p, dw_q)``, the interior of a region
where the state is constant carries ``G(s,s,w,w) = 0`` on every face, so **a uniformly damaged
bulk does not move**.  Measured in v4: interior flux divergence exactly ``0.000e+00`` under
uniform ``dw``.  The sample therefore shuffles mass between neighbours at the rim instead of
condensing as a body.

What changes, and it is one line of physics
-------------------------------------------
The flux law is untouched.  Only its DRIVER changes, from ``Pi`` to a potential ``phi`` that is a
global functional of the field::

    [ varsigma + gamma*(1 - sigma_p) ] phi_p  -  sum_{q~p} sigma_pq (phi_q - phi_p)  =  Pi_p

with ``sigma`` a material indicator and ``sigma_pq`` its face average.  Antisymmetry of the flux
is untouched, so prop:xd_mass holds with its proof unchanged; what is broken is the POINTWISE
hypothesis of prop:xd_locality, which is what forbade bulk contraction.  The proposition needs no
correction -- antisymmetry is the whole of conservation, but it is pointwise locality of the
driver that forbids whole-body motion.

This is not a new closure.  subsec:rationale already names the family: reduce the elasticity to
potential flow and take one Jacobi sweep from rest, and you get v4; iterate the sweep and you
walk back to the elliptic answer.  ``varsigma`` is that dial, unpinned.

Two things that look right and are not
--------------------------------------
* **The source is Pi, not dw.**  ``dw`` is a fraction of what is present and, by shielding, is
  largest where least material lies upstream -- so it peaks at the rim and is largest of all in
  vacuum, where nothing shields the beam and nothing is there to damage.  Driving the potential
  with ``dw`` pushes the specimen OUTWARD.  ``Pi = dw * ftilde / f_max`` is the created void and
  vanishes in vacuum by construction.
* **The free surface needs an absorption coefficient that does NOT scale with the reach.**  A
  plain screened Poisson ``(I + l^2 L) phi = Pi``, leaving the vacuum to pin itself, fails
  quietly: ``phi`` comes out as a smoothed ``Pi``, still rim-peaked at every reach, because the
  conduction in the ersatz background beats the identity term once ``l`` is large.  ``gamma`` is
  a separate coefficient held fixed as ``varsigma`` varies.

Why this is not simply v2 again
-------------------------------
Elasticity erased the design anisotropy because it has SHEAR: a dilatational eigenstrain is
relieved largely by deviatoric rearrangement, so the body accommodates a concentrated contraction
internally and its outline barely moves.  A scalar potential has no deviatoric degrees of
freedom, so the whole relief must appear as boundary motion and inherits the angular structure of
the source.  Nonlocality was never what cost the signal.

Limits
------
``varsigma -> infinity`` recovers v4 exactly, with ``c_cp_v4 = c_cp_v5 * (l/Delta)^2``.
``varsigma = 0`` is the pure Poisson closure and is the default.

Reconstruction
--------------
The Pyomo reconstruction of this prototype is in the second half of this file, under the
``=== reconstruction ===`` banner.  Its own notes follow.

Pyomo transcription of v5, the shrinkage dose-response model.

The reduced model with a NONLOCAL compaction potential.

Same checking discipline as v4: data comes from :func:`degrade_2d_shrinkage_decay_v2_proto4.simulate` (numpy), so the
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

Dose fractionation study
------------------------
(Formerly ``scripts/experiment_fractionation.py``; now the ``fractionation`` CLI verb.)

Dose fractionation in the v5 forward model: 10 angles one at a time vs all 10 at once.

The two runs deliver the SAME total exposure -- ten full-fan bundles at ten evenly spaced
angles, the same ``I0`` per ray -- and differ only in how it is split in time:

  A. SEQUENTIAL   ten measurement steps, one angle each.  This is what ``simulate`` does.
  B. SIMULTANEOUS one measurement step carrying all ten angles.

Any difference between them is fractionation, not dose.  It is expected to be non-zero because
every channel in the step map is nonlinear in the per-step fluence: the converted fraction
``dw = 1 - exp(-c I delta)`` saturates, the decay ``exp(-a I - b I^2)`` compounds, and the
compaction flux is driven by a potential solved once per step.

WHAT "SIMULTANEOUS" MEANS HERE, precisely.  Within one step every ray integrates the field as it
stood at the START of that step, so rays do not shield one another's damage.  That is not an
approximation invented for this experiment -- it is the convention the model already uses for a
bundle, and the one CLAUDE.md records for the 3D sinogram ("rays within one measurement do not
see each other's damage").  Firing ten angles at once therefore just sums the ten dose fields
before the single decay and the single compaction, which is what simultaneous irradiation is.

:func:`check_equivalence` is the gate: with ONE bundle, the simultaneous step must reproduce
:func:`degrade_2d_shrinkage_decay_v2_proto4.step` bit-for-bit.  It does (0.0e+00), so the generalisation adds nothing and
removes nothing at K=1, and the whole difference measured below is the fractionation.

Run it::

    python3 -m archives.degrade_2d_shrinkage_decay_v2_proto4 fractionation
    python3 -m archives.degrade_2d_shrinkage_decay_v2_proto4 fractionation --a 0 --c 0.8 --tag a0_c0.8

``--a 0 --b 0`` is the clean shrinkage case: the decay channel is off, :func:`prop:xd_mass`
makes the total EXACTLY conserved, and every difference between the two runs is then transport
alone rather than transport plus a difference in how much mass each schedule destroyed.
"""

from __future__ import annotations

import io
import os
import re
import sys
import time
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import pyomo.environ as pyo

from senDOE.helpers.rays import bundle_r_values, ray_line_integral
from senDOE.helpers.dose import accumulate_dose, scale_to_optical_depth
from senDOE.helpers.phantoms import demo_sequence as _demo_sequence, phantom as _phantom
from senDOE.helpers.rays import measurement_rays
from senDOE.helpers.shape_metrics import shape_diagnostics, shape_diagnostics as _shape_diagnostics
from senDOE.helpers.solvers import reg_fraction as _reg_fraction, solve_with_fallback
from senDOE.models.tomography_pyomo_2d_shrinkage_decay import _neighbours, _tv_expression
from senDOE.models.tomography_2d_shrinkage_decay import compaction_potential, resolve






@dataclass(frozen=True)
class V5Params:
    """Parameters of v5.  Three new symbols over v4: one physical (``reach``), two numerical."""

    I0: float = 1.0
    c: float = 0.1           # conversion coefficient, dw = 1 - exp(-c I_p delta_p)
    a: float = 0.05          # decay, linear in fluence
    b: float = 0.0           # decay, quadratic
    c_cp: float = 0.3        # compaction number
    # COMPACTION REACH, a physical length in the same units as dx. MANDATORY and strictly
    # positive: varsigma = (dx/reach)^2 is the only unconditional regulariser of the potential
    # operator, and l = infinity is NOT a legal setting -- see resolve(). None means "resolve to
    # the specimen radius R", the default, not "infinite".
    #   lower bound, physical : l >~ R/2, or the potential stays rim-peaked and you are back in
    #                           v4's regime (centre/rim 0.42 at R/5, 2.24 at R/2, 4.06 as l->inf)
    #   upper bound, numerical: cond(A) ~ (gamma+8)*(l/dx)^2, so l ~ 90 px is still unremarkable
    reach: Optional[float] = None
    gamma: float = 100.0     # vacuum absorption. Numerical. Needs gamma >> max(varsigma, 1).
    # Density at which material starts conducting. Numerical. None -> the linear indicator
    # sigma = ft/f_max, which needs no new parameter but makes a low-contrast interior behave
    # partly like vacuum.
    f_ref_frac: Optional[float] = 0.05
    f_max: Optional[float] = None
    beta: float = 1000.0     # logistic upwind sharpness
    flux: str = "upwind"     # "upwind" | "harmonic" | "central"
    eps_h: float = 1e-12
    dx: float = 1.0

    def varsigma(self) -> float:
        """``(dx/reach)^2``.  Strictly positive; ``reach`` must have been resolved first."""
        if self.reach is None:
            raise ValueError("reach is unresolved: call degrade_2d_shrinkage_decay_v2_proto4.resolve(p, theta) first")
        if not np.isfinite(self.reach) or self.reach <= 0:
            raise ValueError(
                "reach = %r is not a legal setting. The pure Poisson closure (l = infinity, "
                "varsigma = 0) is a LIMIT to be approached and reported, never selected: with "
                "no varsigma the only regulariser is the absorption gamma*(1-sigma), which "
                "vanishes on any field without vacuum, leaving a singular pure Neumann "
                "Laplacian. Use a finite l; l ~ R is the default." % (self.reach,))
        return float((self.dx / self.reach) ** 2)

    def decay_factor(self, I_p):
        I_p = np.asarray(I_p, dtype=float)
        return np.exp(-self.a * I_p - self.b * I_p ** 2)


@dataclass
class StepInfo5:
    mass: float
    lost: float
    dw_max: float
    I_max: float
    state_min: float
    compaction: float          # C_k, now built from phi rather than Pi
    max_g: float
    flux_sum: float
    phi_max: float
    phi_core_rim: float        # centre/rim ratio of phi; > 1 means the potential has inverted


# --- step 4a: the compaction potential ---------------------------------------------------

# --- step 4b: the flux, driven by phi instead of Pi ---------------------------------------

def flux_divergence(ft, phi, c_cp: float, mode: str = "upwind", beta: float = 1000.0,
                    eps_h: float = 1e-12):
    """``sum_{q~p} F_{p->q}`` with the potential supplied directly.

    Identical in form to v4's, and identical in value when ``phi`` is handed ``Pi`` -- which is
    how the v4 limit is checked.  Antisymmetric by construction: each face contributes ``+F`` to
    one cell and ``-F`` to its neighbour, so the divergence sums to zero over the grid and
    prop:xd_mass is untouched.
    """
    ft = np.asarray(ft, dtype=float)
    phi = np.asarray(phi, dtype=float)
    gh = phi[:, 1:] - phi[:, :-1]
    gv = phi[1:, :] - phi[:-1, :]
    if mode == "harmonic":
        eh = eps_h * max(float(np.abs(ft).max()), 1e-300)
        ah, bh = ft[:, :-1], ft[:, 1:]
        av, bv = ft[:-1, :], ft[1:, :]
        Fh = c_cp * (2.0 * ah * bh / (ah + bh + eh)) * gh
        Fv = c_cp * (2.0 * av * bv / (av + bv + eh)) * gv
    elif mode == "upwind":
        wh = 1.0 / (1.0 + np.exp(-beta * gh))
        wv = 1.0 / (1.0 + np.exp(-beta * gv))
        Fh = c_cp * (wh * ft[:, :-1] + (1.0 - wh) * ft[:, 1:]) * gh
        Fv = c_cp * (wv * ft[:-1, :] + (1.0 - wv) * ft[1:, :]) * gv
    else:
        Fh = c_cp * 0.5 * (ft[:, :-1] + ft[:, 1:]) * gh
        Fv = c_cp * 0.5 * (ft[:-1, :] + ft[1:, :]) * gv
    div = np.zeros_like(ft)
    div[:, :-1] += Fh
    div[:, 1:] -= Fh
    div[:-1, :] += Fv
    div[1:, :] -= Fv
    return div, (Fh, Fv)


def compaction_number(ft, phi, c_cp: float) -> float:
    """``C_k = c_cp * max_p sum_{q~p} max(phi_q - phi_p, 0)``.  Donor-cell positivity bound."""
    phi = np.asarray(phi, dtype=float)
    gh = phi[:, 1:] - phi[:, :-1]
    gv = phi[1:, :] - phi[:-1, :]
    out = np.zeros_like(phi)
    out[:, :-1] += np.maximum(gh, 0.0)
    out[:, 1:] += np.maximum(-gh, 0.0)
    out[:-1, :] += np.maximum(gv, 0.0)
    out[1:, :] += np.maximum(-gv, 0.0)
    return float(c_cp * np.max(out))


# --- the step ------------------------------------------------------------------------------

def step(f, r_values, angle_rad: float, p: V5Params):
    """One measurement step.  Steps 1, 2, 3 and 5 are v4's verbatim; only 4 changed."""
    f = np.asarray(f, dtype=float)
    fm = float(p.f_max)

    cIdelta, I_p = accumulate_dose(f, r_values, angle_rad, p.I0, p.c)   # 1
    dw = 1.0 - np.exp(-cIdelta)                                          # 2
    ft = f * p.decay_factor(I_p)                                         # 3
    lost = float(f.sum() - ft.sum())

    phi, Pi, sigma = compaction_potential(ft, dw, p)                     # 4a
    div, (Fh, Fv) = flux_divergence(ft, phi, p.c_cp, p.flux, p.beta, p.eps_h)   # 4b
    f_next = ft - div                                                    # 5

    nr, nc = f.shape
    yy, xx = np.mgrid[0:nr, 0:nc]
    c = (nr - 1) / 2.0
    rad = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    core, rim = rad < 0.25 * nr, (rad >= 0.30 * nr) & (rad < 0.40 * nr)
    pr = (float(phi[core].mean()) / float(phi[rim].mean())
          if rim.any() and abs(float(phi[rim].mean())) > 1e-300 else float("nan"))
    gmax = max(float(np.abs(phi[:, 1:] - phi[:, :-1]).max()) if nc > 1 else 0.0,
               float(np.abs(phi[1:, :] - phi[:-1, :]).max()) if nr > 1 else 0.0)
    info = StepInfo5(mass=float(f_next.sum()), lost=lost, dw_max=float(dw.max()),
                     I_max=float(I_p.max()), state_min=float(f_next.min()),
                     compaction=compaction_number(ft, phi, p.c_cp), max_g=gmax,
                     flux_sum=float(Fh.sum() + Fv.sum()),
                     phi_max=float(phi.max()), phi_core_rim=pr)
    return f_next, info


def simulate(theta, seq, p: V5Params, image_res: int,
             record_observations: bool = False, record_trajectory: bool = False):
    """Run a measurement sequence.  Observations are index-shifted: ``y_{k+1} = C f_{k+1}``."""
    theta = np.asarray(theta, dtype=float)
    p = resolve(p, theta)
    f = theta.copy()
    infos, obs = [], []
    hist = [f.copy()]
    for angle_deg, offset, n_beams in seq:
        ang = np.deg2rad(float(angle_deg))
        rv = bundle_r_values(float(offset), int(n_beams), int(image_res))
        f, info = step(f, rv, ang, p)
        if record_observations:
            obs.append(np.array([ray_line_integral(f, r, ang) for r in rv]))
        infos.append(info)
        if record_trajectory:
            hist.append(f.copy())
    out = [f, infos]
    if record_observations:
        out.append(obs)
    if record_trajectory:
        out.append(hist)
    return tuple(out)


def match_compaction_number(theta, seq, p: V5Params, image_res: int, target: float = 0.30,
                            tol: float = 1e-3, max_iter: int = 40) -> float:
    """Return the ``c_cp`` whose peak ``C_k`` over the run equals ``target``.

    Comparisons between closures must be at matched CONTRACTION, not matched ``c_cp``: the
    potential rescales the driver, so equal ``c_cp`` means very different amounts of motion and
    any design-separation number taken that way is confounded.  ``C_k`` is linear in ``c_cp`` at
    fixed trajectory, so a secant iteration converges in a handful of steps.
    """
    p = resolve(p, theta)
    lo, hi = 1e-6, 1e6
    for _ in range(max_iter):
        mid = np.sqrt(lo * hi)
        _f, infos = simulate(theta, seq, replace(p, c_cp=mid), image_res)
        ck = max(i.compaction for i in infos)
        if not np.isfinite(ck) or ck > target:
            hi = mid
        else:
            lo = mid
        if np.isfinite(ck) and abs(ck - target) < tol * target:
            return float(mid)
    return float(np.sqrt(lo * hi))


# ============================== reconstruction ==============================
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

    m = pyo.ConcreteModel(name="degrade_2d_shrinkage_decay_v2_proto4")
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
    """Every variable of the model, taken off a :func:`degrade_2d_shrinkage_decay_v2_proto4.simulate` run."""
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
    # Through compaction_potential, NOT a local spsolve: that function carries the exact
    # M-matrix bound ||phi||_inf <= ||Pi||_inf/varsigma, and a sparse direct solver handed a
    # near-singular operator returns a large finite answer rather than an error. Re-solving it
    # here would have skipped the one guard that catches that, on the very path -- estimation
    # initialisation -- whose iterates can carry no vacuum and so trigger it.
    Pi = np.zeros_like(ft)
    sig = np.zeros_like(ft)
    phi = np.zeros_like(ft)
    for k in range(K):
        ph_k, pi_k, sg_k = compaction_potential(ft[:, k].reshape(res, res),
                                                dw[:, k].reshape(res, res), p)
        phi[:, k], Pi[:, k], sig[:, k] = ph_k.ravel(), pi_k.ravel(), sg_k.ravel()
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
    """G1: does the Pyomo model reproduce degrade_2d_shrinkage_decay_v2_proto4.simulate? Residual only, no solver."""
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

    # LIFT THE OBJECTIVE. Measured with PyNumero at the start point, grid 32: the objective
    # gradient inf-norm is 3.17e-05 against constraint row inf-norms of 1 to 5.09e+03. IPOPT's
    # gradient-based scaling is min(1, 100/||g||) -- it caps large gradients and NEVER lifts a
    # small one, so obj_scaling_factor comes out 1, the objective stays eight orders below the
    # constraints, and scaled == unscaled bit-for-bit in every exit block. The consequence is not
    # cosmetic: without this the solve dies in restoration (grid 32 at iteration 748, grid 12 at
    # 828), and with it the SAME model reaches optimal.
    #
    # 1e4, not 1e5: the magnitude does NOT transfer. Measured at K=4/K=3, full budget --
    #   grid 32:  none -> Restoration Failed 748 | 1e4 -> optimal 479 it, theta 10.31% (277 s)
    #                                            | 1e5 -> Restoration Failed at 117
    #   grid 12:  none -> Restoration Failed 828 | 1e3 -> optimal 390 | 1e4 -> optimal 530,
    #                                              theta 11.99% | 1e5 -> optimal 123
    # 1e4 is the only value measured to converge at BOTH grids. Treat it as calibrated to this
    # objective's scale, not as a universal constant: change the normalisation in
    # add_estimation_objective and this needs re-measuring.
    #
    # The principled fix is to normalise the objective GRADIENT rather than its value --
    # add_estimation_objective normalises both terms to O(1) in value, which is what makes
    # tv_weight a dimensionless ratio, and nothing there touches the gradient. Until that is
    # done this option is the stand-in.
    obj_scaling_factor: float = 1e4

    # ma97. ma57 is faster at grid 12 (optimal in 308 iterations / 14 s bare, where bare ma97
    # fails), but that did NOT transfer: at grid 32 ma57 ran 87 minutes without returning and was
    # abandoned, as was ma27 at 64 minutes. With obj_scaling_factor in place ma97 converges at
    # both grids, so it is the only pairing measured at the target resolution.
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
                "The Pyomo model no longer reproduces degrade_2d_shrinkage_decay_v2_proto4.simulate on this geometry "
                "(residual %.3e at %s). Reconstructing against it would not mean anything; "
                "run degrade_2d_shrinkage_decay_v2_proto4.check_forward() to localise the disagreement."
                % (gate_resid, where))

    opts = dict(params.solver_opts or {})
    opts.setdefault("ma97_order", "metis")
    # Applied to the continuation too: it has the same tiny-gradient objective, and keeping the
    # two solves on one scale means the warm start is not handed across a scale change.
    if params.obj_scaling_factor and "obj_scaling_factor" not in opts:
        opts["obj_scaling_factor"] = float(params.obj_scaling_factor)

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
    :func:`degrade_2d_shrinkage_decay_v2_proto4.simulate` trajectory and evaluates residuals.  That check can only say the
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
    """Initialise every variable of a v5 Pyomo model from a :mod:`degrade_2d_shrinkage_decay_v2_proto4` run.

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

    def _put(v, x):
        """Set a value, clipped into the variable's own bounds.

        The numpy trajectory carries slightly negative f where the flux overshoots (down to
        ~-1e-8 at grid 32), and f_bounds starts at 0, so writing it raw puts the START POINT
        outside the feasible box -- Pyomo logs W1002 for every one and IPOPT has to relocate
        them before iteration 0, which is exactly the carefully-built feasible start being
        thrown away. Clipping costs nothing: the excursion is at round-off next to a field
        whose peak is ~1e-1.
        """
        lo, hi = v.lb, v.ub
        if lo is not None and x < lo:
            x = lo
        if hi is not None and x > hi:
            x = hi
        v.set_value(x)

    # f and the per-stage scalar fields. Blocks absent through inlining, or frozen to Params,
    # are skipped rather than special-cased at the call site.
    for q in m.PIX:
        for k in m.T:
            _put(m.f[q, k], float(traj["f"][q, k]))
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
            _put(m.f[q, 0], float(theta_seed.ravel()[q]))
            m.f[q, 0].fix()

    resid, where = max_residual(m)
    out = dict(seed_is_theta_ref=default_seed, residual=resid, worst_row=where,
               blocks=written, n_steps=m.n_steps, res=m.res,
               sigma_frozen=m.sigma_frozen, theta_fixed=bool(fix_theta))
    if verbose:
        print("  initialised %d blocks from the numpy model over %d steps; worst residual "
              "%.3e (%s)" % (len(written), m.n_steps, resid, where))
    return out


# --- CLI ------------------------------------------------------------------------------------
#
# A grid-32 reconstruction takes minutes and must not need a browser session open for the whole
# of it. Same shape as degrade_2d_shrinkage_decay_v2_proto1's: no subcommands, one --reconstruct flag switching between
# the model check (the default, and the cheap one) and a run.

def _cli(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reconstruct", action="store_true",
                    help="run a reconstruction instead of the model check")
    ap.add_argument("--image-res", type=int, default=24)
    ap.add_argument("--n-steps", type=int, default=3)
    ap.add_argument("--optical-depth", type=float, default=1.1)
    ap.add_argument("--I0", type=float, default=1.0,
                    help="0 is the CONTROL: identity dynamics, so the error is tomography+TV "
                         "alone and the gap to I0>0 is the cost of inverting the damage model")
    ap.add_argument("--c", type=float, default=0.1)
    ap.add_argument("--a", type=float, default=0.05)
    ap.add_argument("--c-cp", type=float, default=0.3)
    ap.add_argument("--reach", type=float, default=7.0)
    ap.add_argument("--gamma", type=float, default=100.0)
    ap.add_argument("--f-ref-frac", type=float, default=0.002)
    ap.add_argument("--tv-weight", type=float, default=0.001)
    ap.add_argument("--noise-sigma", type=float, default=0.0)
    ap.add_argument("--max-iter", type=int, default=3000)
    ap.add_argument("--linear-solver", default="ma97")
    ap.add_argument("--obj-scaling", type=float, default=1e4,
                    help="IPOPT obj_scaling_factor. 0 disables. The objective gradient is ~3e-05 "
                         "against constraint rows up to 5e+03 and IPOPT's own scaling cannot "
                         "lift it, so without this the solve dies in restoration. 1e4 is the "
                         "only value measured to converge at both grid 12 and grid 32.")
    ap.add_argument("--no-continuation", action="store_true")
    ap.add_argument("-o", "--out", default=None, help="write results to this .npz")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)

    if not a.reconstruct:
        r = check_forward(image_res=a.image_res, n_steps=a.n_steps, I0=a.I0, c=a.c, a=a.a,
                          c_cp=a.c_cp, reach=a.reach, gamma=a.gamma, f_ref_frac=a.f_ref_frac,
                          verbose=not a.quiet)
        return 0 if r < 1e-10 else 1

    params = V5UQParams(
        image_res=a.image_res, optical_depth=a.optical_depth,
        beam_steps=tuple((180.0 * i / a.n_steps, 0.0, 0) for i in range(a.n_steps)),
        I0=a.I0, c=a.c, a=a.a, c_cp=a.c_cp, reach=a.reach, gamma=a.gamma,
        f_ref_frac=a.f_ref_frac, tv_weight=a.tv_weight, noise_sigma=a.noise_sigma,
        continuation=not a.no_continuation, ipopt_max_iter=a.max_iter,
        linear_solver=a.linear_solver, obj_scaling_factor=a.obj_scaling)

    cb = None if a.quiet else (lambda chunk: (sys.stdout.write(chunk), sys.stdout.flush()))
    t0 = time.time()
    r = run_v5_reconstruction(params, log_callback=cb)
    secs = time.time() - t0

    print("\n=== v5 reconstruction, grid %d, %d measurements, I0 = %g ==="
          % (a.image_res, a.n_steps, a.I0))
    print("  model vs simulator : %.3e   (the gate; a reconstruction against a drifted model "
          "means nothing)" % r.forward_residual)
    print("  start residual     : %.3e   (dynamically feasible start, so only the fit is wrong)"
          % r.init_residual)
    print("  continuation       : %-24s %4s it  %6.1fs  err %6.2f%%"
          % (r.continuation_status, r.continuation_iters, r.t_cont, r.theta_rms_cont))
    print("  monolithic solve   : %-24s %4s it  %6.1fs  (%s)"
          % (r.status, r.iters, r.t_solve, r.linear_solver))
    print("  regularised        : %d of %d iterations (%.0f%%)"
          % (r.regularised, r.n_iter_lines, r.regularised_pct))
    print("  size               : %s vars, %s cons" % (format(r.n_vars, ","),
                                                       format(r.n_cons, ",")))
    print("  fit RMS            : %.4e   (what the NLP minimised)" % r.obs_rms)
    print("  theta error        : %.3f%% of peak   (synthetic data only)" % r.theta_pct_peak)
    print("  eq:xd_box          : %d at lower, %d at upper, %d interior"
          % (r.n_theta_at_lower, r.n_theta_at_upper, r.n_theta_interior))
    print("  damage context     : mass %.4g -> %.4g, C_k %.3f, max phi %.3g"
          % (r.theta_true.sum(), r.mass_true, r.ck_max, r.phi_max))
    print("  shape              : support %+.3f%%, half-mass %+.3f%%, %d radial sign change(s)"
          % (r.support_pct, r.half_pct, r.flips))
    print("  total              : %.1f s" % secs)
    if "optimal" not in r.status:
        print("\n  NOT CONVERGED. Read it this way: a LOW start residual with a HIGH")
        print("  regularisation fraction points at the bilinear (sigma, phi) block in c_phi,")
        print("  not at the initialisation; a high start residual points at the start.")
        print("  Compare against --I0 0, which keeps the same block but removes the dynamics.")

    if a.out:
        np.savez_compressed(a.out, theta_true=r.theta_true, theta_hat=r.theta_hat,
                            f_final_true=r.f_final_true, f_final_hat=r.f_final_hat,
                            theta_rms=r.theta_rms, theta_pct_peak=r.theta_pct_peak,
                            obs_rms=r.obs_rms, seconds=secs)
        print("  wrote %s" % a.out)
    return 0


# ========================== dose fractionation study ==========================
_FRACTIONATION_OUT = os.path.dirname(os.path.abspath(__file__))


def step_simultaneous(f, bundles, p: V5Params):
    """One step carrying several ``(r_values, angle_rad)`` bundles at once.

    Mirrors :func:`degrade_2d_shrinkage_decay_v2_proto4.step` line for line; the only change is that steps 1-2 accumulate
    over every bundle against the SAME ``f`` before step 3 decays it and step 4 solves one
    potential.  Returns ``(f_next, info)`` with ``info`` a plain dict.
    """
    f = np.asarray(f, dtype=float)
    cIdelta = np.zeros_like(f)
    I_p = np.zeros_like(f)
    for r_values, angle_rad in bundles:                                  # 1
        a, b = accumulate_dose(f, r_values, angle_rad, p.I0, p.c)
        cIdelta = cIdelta + a
        I_p = I_p + b
    dw = 1.0 - np.exp(-cIdelta)                                          # 2
    ft = f * p.decay_factor(I_p)                                         # 3
    lost = float(f.sum() - ft.sum())

    phi, _Pi, _sigma = compaction_potential(ft, dw, p)                   # 4a
    div, (Fh, Fv) = flux_divergence(ft, phi, p.c_cp, p.flux, p.beta, p.eps_h)   # 4b
    f_next = ft - div                                                    # 5

    info = dict(mass=float(f_next.sum()), lost=lost, dw_max=float(dw.max()),
                I_max=float(I_p.max()), state_min=float(f_next.min()),
                compaction=compaction_number(ft, phi, p.c_cp),
                flux_sum=float(Fh.sum() + Fv.sum()), phi_max=float(phi.max()))
    return f_next, info


def check_equivalence(image_res: int = 24, verbose: bool = True) -> float:
    """Gate: at ONE bundle the simultaneous step must BE :func:`degrade_2d_shrinkage_decay_v2_proto4.step`."""
    theta = scale_to_optical_depth(_phantom(image_res), 1.1, image_res)
    p = resolve(V5Params(reach=7.0, f_ref_frac=0.002), theta)
    rv = bundle_r_values(0.0, 0, image_res)
    ang = np.deg2rad(37.0)
    a, _ = step(theta, rv, ang, p)
    b, _ = step_simultaneous(theta, [(rv, ang)], p)
    err = float(np.abs(a - b).max())
    if verbose:
        print("  gate: one-bundle simultaneous == degrade_2d_shrinkage_decay_v2_proto4.step -> %.3e  %s"
              % (err, "PASS" if err == 0.0 else "FAIL (must be exactly 0)"))
    return err


def radial_profile(img, centroid, n_bins: int, r_max: float):
    """Mass per radial band about a FIXED centroid.  Mass, not mean: it must sum to the total."""
    nr, nc = img.shape
    yy, xx = np.mgrid[0:nr, 0:nc]
    rad = np.sqrt((xx - centroid[0]) ** 2 + (yy - centroid[1]) ** 2).ravel()
    w = np.asarray(img, dtype=float).ravel()
    edges = np.linspace(0.0, r_max, n_bins + 1)
    out = np.array([w[(rad >= edges[i]) & (rad < edges[i + 1])].sum() for i in range(n_bins)])
    return 0.5 * (edges[:-1] + edges[1:]), out


def run_fractionation(image_res=32, n_angles=10, optical_depth=1.1, **over):
    """Both runs on one phantom.  Returns ``(theta, resultA, resultB, params)``."""
    theta = scale_to_optical_depth(_phantom(image_res), optical_depth, image_res)
    kw = dict(I0=1.0, c=0.1, a=0.05, b=0.0, c_cp=0.3, reach=7.0, gamma=100.0,
              f_ref_frac=0.002, beta=1000.0)
    kw.update(over)
    p = resolve(V5Params(**kw), theta)

    angles = [180.0 * k / n_angles for k in range(n_angles)]
    rv = bundle_r_values(0.0, 0, image_res)          # full fan, zero offset: identical per angle

    # A -- ten steps, one angle each. The stock path, untouched.
    seq = tuple((a, 0.0, 0) for a in angles)
    fA, infosA = simulate(theta, seq, p, image_res)
    A = dict(f=fA, steps=len(infosA),
             ck=max(i.compaction for i in infosA), dw=max(i.dw_max for i in infosA),
             fmin=min(i.state_min for i in infosA), phi=max(i.phi_max for i in infosA),
             lost=sum(i.lost for i in infosA))

    # B -- one step, all ten angles.
    fB, infoB = step_simultaneous(theta, [(rv, np.deg2rad(a)) for a in angles], p)
    B = dict(f=fB, steps=1, ck=infoB["compaction"], dw=infoB["dw_max"],
             fmin=infoB["state_min"], phi=infoB["phi_max"], lost=infoB["lost"])

    for d in (A, B):
        d["mass"] = float(d["f"].sum())
        d["support_pct"], d["half_pct"], d["flips"] = shape_diagnostics(theta, d["f"])
    return theta, A, B, p, angles


def match_ck(target, image_res, n_angles, kw, tol=2e-3, max_iter=25, verbose=True):
    """Scale ``c_cp`` until the WORSE of the two schedules hits ``target`` for max C_k.

    One ``c_cp`` for both runs, not one each: it is a material property, and the experiment
    varies the schedule, not the material.  Matching per-run would confound the comparison with
    a different compaction amplitude, which is exactly what
    :func:`degrade_2d_shrinkage_decay_v2_proto4.match_compaction_number` warns against.

    The simultaneous run binds, and there C_k is EXACTLY linear in ``c_cp`` -- within one step
    ``phi`` is solved from ``ft`` and ``dw`` before the flux, so it does not see ``c_cp`` at all.
    Sequentially the feedback through ``f`` makes it only nearly linear, so this iterates the
    fixed point ``c <- c * target / C_k(c)`` rather than assuming one shot is enough.
    """
    c = float(kw.get("c_cp", 0.3))
    for it in range(1, max_iter + 1):
        k = dict(kw); k["c_cp"] = c
        _t, A, B, _p, _a = run_fractionation(image_res=image_res, n_angles=n_angles, **k)
        worst = max(A["ck"], B["ck"])
        if verbose:
            print("    match c_cp=%.5f -> C_k seq %.4f / sim %.4f (worst %.4f)"
                  % (c, A["ck"], B["ck"], worst))
        if abs(worst - target) <= tol:
            return c
        c *= target / max(worst, 1e-12)
    return c


def _figure(theta, res, title, path, vmax, span, prof_ref, dlim):
    """Four panels.  Colour scales are passed in so the two figures are directly comparable."""
    f = res["f"]
    nr, nc = theta.shape
    yy, xx = np.mgrid[0:nr, 0:nc]
    m0 = float(theta.sum())
    cen = ((xx * theta).sum() / m0, (yy * theta).sum() / m0)   # FIXED initial centroid
    r, prof = radial_profile(f, cen, 16, 0.55 * nr)
    r0, prof0 = prof_ref

    import matplotlib
    matplotlib.use("Agg")                  # headless
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 4, figsize=(17.5, 4.4))
    im = ax[0].imshow(theta, cmap="gray", vmin=0.0, vmax=vmax, interpolation="nearest")
    ax[0].set_title("theta (undamaged)", fontsize=10); fig.colorbar(im, ax=ax[0], fraction=0.046)
    im = ax[1].imshow(f, cmap="gray", vmin=0.0, vmax=vmax, interpolation="nearest")
    ax[1].set_title("f after exposure", fontsize=10); fig.colorbar(im, ax=ax[1], fraction=0.046)
    im = ax[2].imshow(f - theta, cmap="coolwarm", vmin=-span, vmax=span, interpolation="nearest")
    ax[2].set_title("change  f - theta", fontsize=10); fig.colorbar(im, ax=ax[2], fraction=0.046)
    for a in ax[:3]:
        a.set_xticks([]); a.set_yticks([])

    # NORMALISED by total mass, and shown as a difference. At a = 0.05 the decay removes ~39% of
    # the mass uniformly, which swamps the transport in an absolute profile and paints the whole
    # change panel one colour. Dividing by the total removes the decay and leaves the SHAPE
    # change, which is what the compaction does and what this experiment is about. Positive
    # inner bands with negative outer bands is condensation.
    d = prof / max(prof.sum(), 1e-300) - prof0 / max(prof0.sum(), 1e-300)
    ax[3].axhline(0.0, color="#bbbbbb", lw=1)
    ax[3].bar(r, d, width=(r[1] - r[0]) * 0.85,
              color=["#1f77b4" if v >= 0 else "#d62728" for v in d])
    ax[3].set_title("radial redistribution, mass-normalised\n"
                    "(inner + / outer - = condensation)", fontsize=10)
    ax[3].set_xlabel("radius (px)"); ax[3].set_ylabel("share of total mass, after - theta")
    ax[3].set_ylim(-dlim, dlim)          # SHARED across both figures, so they compare by eye
    ax[3].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax[3].spines[["top", "right"]].set_visible(False)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _fractionation_cli(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Dose fractionation in the v5 forward model: sequential vs simultaneous.")
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--angles", type=int, default=10)
    ap.add_argument("--I0", type=float, default=1.0)
    ap.add_argument("--c", type=float, default=0.1, help="c_omega, the conversion coefficient")
    ap.add_argument("--a", type=float, default=0.05, help="linear decay; 0 with b=0 conserves mass")
    ap.add_argument("--b", type=float, default=0.0)
    ap.add_argument("--c-cp", type=float, default=0.3)
    ap.add_argument("--reach", type=float, default=7.0)
    ap.add_argument("--gamma", type=float, default=100.0)
    ap.add_argument("--f-ref-frac", type=float, default=0.002)
    ap.add_argument("--match-ck", type=float, default=None, metavar="TARGET",
                    help="solve for the c_cp that puts the WORSE schedule's max C_k at TARGET, "
                         "and use that one c_cp for both runs. Use to stay inside the donor-cell "
                         "positivity bound (C_k <= 1), which the simultaneous schedule breaches "
                         "at c_cp = 0.3 with c_omega = 0.8.")
    ap.add_argument("--tag", default="", help="suffix for the PNG names, so runs do not overwrite")
    a = ap.parse_args(argv)

    print("Dose fractionation in the v5 forward model: 10 angles one at a time vs all 10 at once.")
    print()
    check_equivalence()
    print()

    kw = dict(I0=a.I0, c=a.c, a=a.a, b=a.b, c_cp=a.c_cp, reach=a.reach, gamma=a.gamma,
              f_ref_frac=a.f_ref_frac)
    if a.match_ck is not None:
        print("  matching c_cp so the worse schedule's max C_k = %.3f" % a.match_ck)
        kw["c_cp"] = match_ck(a.match_ck, a.grid, a.angles, kw)
        print("    -> c_cp = %.5f (was %.5f), applied to BOTH runs\n" % (kw["c_cp"], a.c_cp))
    theta, A, B, p, angles = run_fractionation(image_res=a.grid, n_angles=a.angles, **kw)
    sfx = ("_" + a.tag) if a.tag else ""
    nr, _ = theta.shape
    yy, xx = np.mgrid[0:nr, 0:nr]
    m0 = float(theta.sum())
    cen = ((xx * theta).sum() / m0, (yy * theta).sum() / m0)
    prof_ref = radial_profile(theta, cen, 16, 0.55 * nr)

    vmax = float(theta.max())
    span = max(float(np.abs(A["f"] - theta).max()), float(np.abs(B["f"] - theta).max()))

    def _nd(f):
        _r, pr = radial_profile(f, cen, 16, 0.55 * nr)
        return pr / max(pr.sum(), 1e-300) - prof_ref[1] / max(prof_ref[1].sum(), 1e-300)
    dlim = 1.05 * max(float(np.abs(_nd(A["f"])).max()), float(np.abs(_nd(B["f"])).max()))

    hdr = ("grid %d, %d evenly spaced angles, full fan | I0=%g c=%g a=%g b=%g c_cp=%g "
           "reach=%g gamma=%g f_ref_frac=%g"
           % (nr, len(angles), p.I0, p.c, p.a, p.b, p.c_cp, p.reach, p.gamma, p.f_ref_frac))
    pA = _figure(theta, A, "A. SEQUENTIAL - %d steps, one angle each\n%s" % (A["steps"], hdr),
                 os.path.join(_FRACTIONATION_OUT, "v5_fractionation_sequential%s.png" % sfx),
                 vmax, span, prof_ref, dlim)
    pB = _figure(theta, B, "B. SIMULTANEOUS - 1 step carrying all %d angles\n%s"
                 % (len(angles), hdr),
                 os.path.join(_FRACTIONATION_OUT, "v5_fractionation_simultaneous%s.png" % sfx),
                 vmax, span, prof_ref, dlim)

    print("  %-26s %14s %14s %14s" % ("", "A sequential", "B simultaneous", "B - A"))
    rows = [("measurement steps", "steps", "%d", 0),
            ("total mass", "mass", "%.5f", 1),
            ("mass lost to decay", "lost", "%.5f", 1),
            ("support radius 99% (%)", "support_pct", "%+.3f", 1),
            ("half-mass radius (%)", "half_pct", "%+.3f", 1),
            ("radial sign changes", "flips", "%d", 0),
            ("max C_k (bound <= 1)", "ck", "%.4f", 1),
            ("max dw", "dw", "%.4f", 1),
            ("max phi", "phi", "%.4f", 1),
            ("min f", "fmin", "%.3e", 1)]
    for lab, key, fmt, diff in rows:
        a, b = A[key], B[key]
        d = (fmt % (b - a)) if diff else ("%+d" % (b - a))
        print("  %-26s %14s %14s %14s" % (lab, fmt % a, fmt % b, d))

    dif = B["f"] - A["f"]
    print()
    if p.a == 0.0 and p.b == 0.0:
        m0 = float(theta.sum())
        print("  a = b = 0, so prop:xd_mass makes the total EXACTLY conserved and every")
        print("  difference below is TRANSPORT alone:")
        print("    mass drift  sequential %+.3e   simultaneous %+.3e   (relative to %.5f)"
              % (A["mass"] - m0, B["mass"] - m0, m0))
    if max(A["ck"], B["ck"]) > 1.0:
        print("  WARNING C_k > 1: the donor-cell positivity bound no longer holds "
              "(sequential %.3f, simultaneous %.3f). Check min f." % (A["ck"], B["ck"]))
    print("  final fields differ by %.4e max abs = %.2f%% of peak theta"
          % (np.abs(dif).max(), 100.0 * np.abs(dif).max() / vmax))
    print("  mass A -> B: %.5f -> %.5f (%+.3f%%)"
          % (A["mass"], B["mass"], 100.0 * (B["mass"] - A["mass"]) / A["mass"]))
    print()
    print("  wrote %s" % pA)
    print("  wrote %s" % pB)
    return 0


if __name__ == "__main__":
    # `python3 -m archives.<this module> fractionation [...]` runs the dose-fractionation
    # study; anything else is the reconstruction CLI.
    if sys.argv[1:2] == ["fractionation"]:
        sys.exit(_fractionation_cli(sys.argv[2:]))
    sys.exit(_cli())
