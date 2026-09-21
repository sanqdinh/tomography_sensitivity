"""v5: v4 with the compaction potential made nonlocal. Step 4 only; nothing else moves.

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
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from dose_response import bundle_r_values, ray_line_integral
from degrade_v2 import accumulate_dose, scale_to_optical_depth
from degrade_v3 import (radius_of_gyration, support_radius, mass_outside, semi_axis_ratio,
                        _disc, _phantom, _demo_sequence, _wedge_sequence)


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
            raise ValueError("reach is unresolved: call degrade_v5.resolve(p, theta) first")
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


def resolve(p: V5Params, theta) -> V5Params:
    """Fill ``f_max`` from the initial peak and ``reach`` from the specimen radius.

    The default reach is ``R``, the radius containing 99% of the mass, which is the same
    statistic the shrinkage is reported against.  It is a physical length, so unlike v4's
    ``c_cp`` it transfers across grids.
    """
    theta = np.asarray(theta, dtype=float)
    out = p
    if out.f_max is None:
        peak = float(np.abs(theta).max())
        out = replace(out, f_max=(peak if peak > 0.0 else 1.0))
    if out.reach is None:
        out = replace(out, reach=float(support_radius(theta)))
    out.varsigma()          # raises here, at entry, rather than downstream on a degenerate solve
    return out


# --- step 4a: the compaction potential ---------------------------------------------------

def material_indicator(ft, f_max: float, f_ref_frac: Optional[float]):
    """``sigma`` in [0,1): 0 in vacuum, ~1 in bulk material.

    ``f_ref_frac`` None gives the parameter-free linear form ``ft/f_max``, which the spec warns
    makes a low-contrast interior behave partly like vacuum.  Both are testable.
    """
    ft = np.asarray(ft, dtype=float)
    # NO CLIPPING. The clip was a guard against the logistic's ~1e-6 negative residual, but it
    # is not expressible in Pyomo, so numpy and the NLP would compute different functions and
    # the gate fails on exactly those pixels (measured: 9.2e-05 on c_sig). The guard is not
    # needed: at ft = -1e-6 the indicator is -2e-4 against a diagonal of gamma ~ 100, so the
    # operator stays strongly diagonally dominant and the M-matrix property survives. In the
    # NLP f carries a lower bound of 0 anyway, so ft >= 0 there by construction.
    if f_ref_frac is None:
        return ft / f_max
    return 1.0 - np.exp(-ft / (f_ref_frac * f_max))


def potential_operator(sigma, varsigma: float, gamma: float):
    """The sparse operator of step 4a.  Symmetric, irreducibly diagonally dominant M-matrix.

    Rows: ``[varsigma + gamma(1-sigma_p)] phi_p - sum_q sigma_pq (phi_q - phi_p) = Pi_p``.
    Faces outside the grid are simply absent from the sum, which is the natural condition.
    """
    sigma = np.asarray(sigma, dtype=float)
    nr, nc = sigma.shape
    n = nr * nc
    idx = np.arange(n).reshape(nr, nc)
    rows, cols, vals = [], [], []
    diag = varsigma + gamma * (1.0 - sigma.ravel())
    for (a_idx, b_idx, s_face) in (
            (idx[:, :-1].ravel(), idx[:, 1:].ravel(),
             (0.5 * (sigma[:, :-1] + sigma[:, 1:])).ravel()),
            (idx[:-1, :].ravel(), idx[1:, :].ravel(),
             (0.5 * (sigma[:-1, :] + sigma[1:, :])).ravel())):
        rows.append(a_idx); cols.append(b_idx); vals.append(-s_face)
        rows.append(b_idx); cols.append(a_idx); vals.append(-s_face)
        np.add.at(diag, a_idx, s_face)
        np.add.at(diag, b_idx, s_face)
    rows.append(np.arange(n)); cols.append(np.arange(n)); vals.append(diag)
    return sp.csc_matrix((np.concatenate(vals),
                          (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))


def compaction_potential(ft, dw, p: V5Params):
    """Solve step 4a for ``phi``.  Returns ``(phi, Pi, sigma)``."""
    fm = float(p.f_max)
    Pi = np.asarray(dw, dtype=float) * np.asarray(ft, dtype=float) / fm
    sigma = material_indicator(ft, fm, p.f_ref_frac)
    vs = p.varsigma()          # raises if reach is illegal, so varsigma > 0 from here on
    A = potential_operator(sigma, vs, p.gamma)
    phi = spla.spsolve(A, Pi.ravel()).reshape(Pi.shape)
    # EXACT bound, not a heuristic. A is irreducibly diagonally dominant with row sums at least
    # varsigma, so the M-matrix structure gives ||phi||_inf <= ||Pi||_inf / varsigma, and the
    # bound is attained on a vacuum-free field with uniform void. Anything above it means the
    # solve did not converge or the operator was assembled wrong.
    bound = float(np.abs(Pi).max()) / vs
    if not np.all(np.isfinite(phi)) or float(np.abs(phi).max()) > bound * (1.0 + 1e-6) + 1e-12:
        raise ValueError("compaction potential violates its M-matrix bound: max|phi| = %.6e "
                         "against ||Pi||_inf/varsigma = %.6e"
                         % (float(np.abs(phi).max()), bound))
    return phi, Pi, sigma


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


def shape_diagnostics(theta, f):
    """``(support_pct, half_pct, flips)`` -- what the damage did to the body's shape.

    Radii are measured about the **fixed initial centroid**, not a moving one, so a body that
    translates does not read as a body that contracted.  ``flips`` counts sign changes in the
    radially banded mass difference: ONE is coherent condensation (mass leaves the outside and
    arrives inside), several means mass is shuffling between neighbours.

    Lives HERE, in the forward module, because it is a forward diagnostic: degrade_v5_uq
    imports it rather than owning it, so a forward-only experiment does not have to import
    pyomo to measure a shape.  Kept identical to the forward tab's readout in ``app.py`` so
    the two cannot disagree.
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
