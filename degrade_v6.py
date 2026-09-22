"""v6, the implicit-transport damage model: the compaction flux made implicit and unconditional.

v5 with steps 5 and 6 changed and nothing else moved.

What changes
------------
v5 discretised eq:xd_mass_transport explicitly, with a logistic upwind weight picking a donor
value off the *decayed* field::

    F_{p->q} = c_cp [ chi_pq ft_p + (1 - chi_pq) ft_q ] dP_pq,   chi_pq = 1/(1 + e^{-beta dP})
    f_{k+1}  = ft - div F

That is positive only under the compaction number ``C_k <= 1``, and only in the donor-cell limit
``beta -> infinity``; at finite ``beta`` the face value mixes both sides and ``C_k`` is a
diagnostic rather than a proof.  v6 replaces it with a nonnegative directed RATE and evaluates
the flux on the unknown post-transport field (eq:xd_directed_rate, eq:xd_flux)::

    r_{p->q} = c_cp phi_eta(dP_pq),   phi_eta(z) = eta log(1 + e^{z/eta})      (softplus)
    F_{p->q} = r_{p->q} f_{k+1,p} - r_{q->p} f_{k+1,q}

so eq:xd_mass_transport becomes one global sparse system, eq:xd_implicit_transport::

    (1 + sum_q r_{p->q}) f_{k+1,p} - sum_q r_{q->p} f_{k+1,q} = ft_p

Its matrix has positive diagonal, nonpositive off-diagonal and **unit column sums**.  It is a
nonsingular M-matrix, so ``ft >= 0`` implies ``f_{k+1} >= 0`` and the column sums give exact
conservation -- both unconditionally.  ``C_k`` is therefore deleted, along with (S4)'s positivity
caveat: there is no step-size restriction left to diagnose.  Outgoing transport is proportional
to what remains in the donor, so an emptying pixel stops donating and vacuum cannot donate at
all.

Steps 1 to 4 are v5's, by import rather than by copy
----------------------------------------------------
``resolve``, ``material_indicator``, ``potential_operator`` and ``compaction_potential`` are
imported from :mod:`degrade_v5` and called with a :class:`V6Params`.  That works because v6 keeps
v5's field *names* for the potential block and v5's ``resolve`` goes through
``dataclasses.replace``, so handed a ``V6Params`` it returns a ``V6Params``.  Reusing a params
object across versions is new here, so :func:`check_invariants` pins it: check (g) asserts v6's
step 4 equals v5's **bit for bit** on the same field.  The benefit is that eq:xd_potential_solve
has exactly one implementation, and v5's exact M-matrix guard comes along with it -- and that
guard *is* eq:xd_potential_bound, since ``(l/Delta)^2 = 1/varsigma``.

The softplus rate is not zero at zero, and that is structural
-------------------------------------------------------------
``phi_eta(0) = eta log 2 > 0``, so at rest -- ``dP = 0`` on every face -- both directed rates are
``c_cp eta log 2`` and the field still exchanges mass across every face.  This matters because
subsec:system claims the three off switches "hold as identities ... nothing needs guarding", and
specifically that ``I0 = 0`` at ``c_cp > 0`` returns *bitwise* what ``I0 = 0`` at ``c_cp = 0``
returns.  **It does not.**  At ``I0 = 0`` the potential is exactly zero, correctly and with no
guard -- ``Pi = 0`` and eq:xd_potential_solve is nonsingular -- but the transport then runs as
pure diffusion.  Measured on a disc over 8 steps at ``c_cp = 0.5``: ``5.5e-3`` at ``eta = 1e-3``,
``0.30`` at ``eta = 0.1``, linear in ``c_cp eta``.

The mechanism is a property of softplus, not an implementation accident.  ``phi_eta`` satisfies

    phi_eta(z) - phi_eta(-z) = z        exactly, for every eta > 0

(verified to 3.6e-15 over z in [-20, 20]), so the flux splits into a central advection and a
diffusion::

    F_{p->q} = c_cp [ dP (f_p + f_q)/2  +  S(dP) (f_p - f_q) ],
    S(dP) := (phi_eta(dP) + phi_eta(-dP))/2  >=  eta log 2  >  0

The diffusive part has a floor and cannot be removed.  Subtracting ``eta log 2`` to make the rate
vanish at rest sends ``phi_eta`` negative for ``z < 0``, which flips an off-diagonal sign and
costs the M-matrix, hence positivity and the conservation proof with it.  This is the same shape
as v2's ``eps_up`` leak, and the same conclusion: it is numerics, not physics, and the honest
response is to measure it rather than to assert an identity the model does not have.

Mass conservation and positivity are untouched by it -- a diffusion moves mass, it does not
create or destroy it, and it keeps the matrix an M-matrix.  Only the ``I0 = 0`` identity is
affected, and only at order ``c_cp eta``.  So :func:`check_invariants` check (c) reports the
residual beside its ``c_cp eta log 2`` prediction and asserts the *scaling law* instead: the leak
must be proportional to ``eta``, which is the testable statement that the leak is only the
smoothing.  Keep ``eta`` well under the ``max |dP|`` the run actually produces; ``step`` records
both so the ratio is visible.

There is no ``eta = 0`` escape hatch.  The model is eq:xd_rate_function as written.

The conduction threshold has a second condition the note does not state
-----------------------------------------------------------------------
eq:xd_potential_solve's diagonal is ``varsigma + gamma (1 - sigma_p)``, and inside the specimen
``1 - sigma_p = exp(-ft_p / f_ref)``.  For the screening term to set the interior scale, rather
than the vacuum penalty leaking into the bulk, one needs

    gamma exp(-f_interior / f_ref)  <<  varsigma = (Delta / l)^2

subsec:system gives only "``f_ref`` be at most a fifth of the smallest interior value", which
leaves ``gamma e^{-5}``; at ``gamma = 1e3`` and ``varsigma = 6.9e-3`` that is ``6.74``, three
decades too large, and the contraction measures **exactly 0.00%** with nothing raised.  The
sufficient condition is ``f_ref <~ f_interior / log(gamma / varsigma)``.  ``StepInfo6`` carries
``absorption_ratio`` -- the worst ``gamma (1 - sigma_p) / varsigma`` over the bulk -- so the
closure cannot be silently off.  ``experiment_v6_limits.py`` locates the plateau.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from dose_response import bundle_r_values, ray_line_integral
from degrade_v2 import accumulate_dose, scale_to_optical_depth
from degrade_v3 import (radius_of_gyration, support_radius, mass_outside, semi_axis_ratio,
                        _disc, _phantom, _demo_sequence, _wedge_sequence)
# Steps 1 to 4, unchanged by the v6 diff.  Imported, never retyped -- see the module docstring.
from degrade_v5 import (resolve, material_indicator, potential_operator, compaction_potential,
                        shape_diagnostics)


@dataclass(frozen=True)
class V6Params:
    """Parameters of v6.  v5's, minus the three upwind knobs, plus the softplus scale ``eta``.

    Gone with the explicit flux: ``beta`` (logistic sharpness), ``flux`` (the upwind / harmonic /
    central selector) and ``eps_h`` (the harmonic guard).  The implicit form has one face law and
    no positivity condition, so there is nothing left for them to select between.
    """

    I0: float = 1.0
    c: float = 0.1           # conversion coefficient, dw = 1 - exp(-c I_p delta_p)
    a: float = 0.05          # decay, linear in fluence
    b: float = 0.0           # decay, quadratic
    c_cp: float = 0.3        # compaction amplitude; 0 annihilates every flux
    # Fields below here are read by degrade_v5's step-4 functions, which this module calls with
    # a V6Params. Their NAMES are load-bearing -- see check (g).
    reach: Optional[float] = None    # compaction reach l. None -> the specimen radius R.
    gamma: float = 100.0             # vacuum absorption. Needs gamma >> max(varsigma, 1) ...
    f_ref_frac: Optional[float] = 0.002   # ... but ALSO gamma*exp(-f_int/f_ref) << varsigma.
    f_max: Optional[float] = None
    # SOFTPLUS SMOOTHING of eq:xd_rate_function. Strictly positive: the model is the softplus,
    # and eta -> 0 is a limit to be reported, not a setting. Must sit well below the run's
    # max |dP| or the rate stops discriminating direction; StepInfo6 records both. It is also
    # the size of the resting diffusion (rate eta*log2 per face), so it is not free either way.
    eta: float = 1e-3
    dx: float = 1.0

    def varsigma(self) -> float:
        """``(dx/reach)^2``.  Strictly positive; ``reach`` must have been resolved first."""
        if self.reach is None:
            raise ValueError("reach is unresolved: call degrade_v6.resolve(p, theta) first")
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
class StepInfo6:
    mass: float
    lost: float                # what eq:xd_decay removed, measured at the START of the step
    mass_residual: float       # |sum f_{k+1} - sum ft|; prop:xd_mass, and it must be ~0
    dw_max: float
    I_max: float
    state_min: float
    max_g: float
    flux_sum: float
    phi_max: float
    phi_core_rim: float        # centre/rim ratio of phi; > 1 means the potential has inverted
    dP_max: float              # max |dP| over faces -- what eta has to be small against
    rest_rate: float           # c_cp*eta*log2, the directed rate a MOTIONLESS face still carries
    colsum_err: float          # max |column sum - 1| of the transport matrix; conservation
    absorption_ratio: float    # max gamma(1-sigma)/varsigma over the bulk; must be << 1


# --- steps 5 and 6: the implicit transport -------------------------------------------------

def softplus(z, eta: float):
    """``eq:xd_rate_function``: ``eta log(1 + exp(z/eta))``, nonnegative and increasing.

    Evaluated through ``logaddexp`` rather than by forming ``exp(z/eta)``, which subsec:system
    asks for explicitly and which matters here: ``dP/eta`` reaches several hundred at the default
    ``eta``, where the naive form overflows to ``inf`` and the rate becomes NaN.

    ``eta <= 0`` is refused.  ``max(z, 0)`` is the ``eta -> 0`` limit of the model, not a setting
    of it, and admitting it here would let the resting diffusion be switched off by accident --
    the one thing that makes v6's behaviour at ``I0 = 0`` worth reporting.
    """
    if not (eta > 0.0) or not np.isfinite(eta):
        raise ValueError(
            "eta = %r is not a legal setting: eq:xd_rate_function needs eta > 0, and eta -> 0 "
            "is a limit to be reported rather than selected." % (eta,))
    return float(eta) * np.logaddexp(0.0, np.asarray(z, dtype=float) / float(eta))


def _face_pairs(nr: int, nc: int):
    """The oriented face list ``(p, q)``, horizontal then vertical, as v5's operator walks it."""
    idx = np.arange(nr * nc).reshape(nr, nc)
    return ((idx[:, :-1].ravel(), idx[:, 1:].ravel()),
            (idx[:-1, :].ravel(), idx[1:, :].ravel()))


def transport_operator(phi, c_cp: float, eta: float):
    """The sparse operator of ``eq:xd_implicit_transport``.

    Rows: ``(1 + sum_q r_{p->q}) f_p - sum_q r_{q->p} f_q = ft_p``, so ``A[p,p]`` accumulates the
    OUTGOING rates of ``p`` and ``A[p,q] = -r_{q->p}`` is what ``q`` sends in.  Each face puts
    ``+r_{p->q}`` on the diagonal of ``p`` and ``-r_{p->q}`` in row ``q``, column ``p``, so every
    column sums to exactly one -- which is conservation, and is asserted rather than assumed.

    Faces outside the grid are simply absent, the natural no-flux condition on the box.
    """
    phi = np.asarray(phi, dtype=float)
    nr, nc = phi.shape
    n = nr * nc
    ph = phi.ravel()
    diag = np.ones(n)
    rows, cols, vals = [], [], []
    for a_idx, b_idx in _face_pairs(nr, nc):
        dP = ph[b_idx] - ph[a_idx]                 # dP_{pq} = phi_q - phi_p
        r_ab = float(c_cp) * softplus(dP, eta)     # p -> q
        r_ba = float(c_cp) * softplus(-dP, eta)    # q -> p
        np.add.at(diag, a_idx, r_ab)
        np.add.at(diag, b_idx, r_ba)
        rows.append(a_idx); cols.append(b_idx); vals.append(-r_ba)
        rows.append(b_idx); cols.append(a_idx); vals.append(-r_ab)
    rows.append(np.arange(n)); cols.append(np.arange(n)); vals.append(diag)
    return sp.csc_matrix((np.concatenate(vals),
                          (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))


def implicit_transport(ft, phi, c_cp: float, eta: float):
    """Solve ``eq:xd_implicit_transport`` for ``f_{k+1}``.  Returns ``(f_next, colsum_err)``.

    ``c_cp = 0`` is NOT special-cased.  It makes every rate exactly zero, so the operator is
    exactly the identity and the solve returns ``ft`` unchanged -- subsec:system's claim that the
    off switch is an identity of the equations rather than a branch in the code.  Check (b)
    measures that rather than taking it on trust.
    """
    ft = np.asarray(ft, dtype=float)
    A = transport_operator(phi, c_cp, eta)
    colsum_err = float(np.abs(np.asarray(A.sum(axis=0)).ravel() - 1.0).max())
    f_next = spla.spsolve(A, ft.ravel()).reshape(ft.shape)
    if not np.all(np.isfinite(f_next)):
        raise ValueError("implicit transport returned a non-finite field: the operator is an "
                         "M-matrix at every nonnegative rate, so this means it was assembled "
                         "wrong or c_cp/eta are not finite")
    return f_next, colsum_err


def face_fluxes(f_next, phi, c_cp: float, eta: float):
    """``(Fh, Fv)`` from ``eq:xd_flux``, on the solved field.  Diagnostic only."""
    f_next = np.asarray(f_next, dtype=float)
    phi = np.asarray(phi, dtype=float)
    gh = phi[:, 1:] - phi[:, :-1]
    gv = phi[1:, :] - phi[:-1, :]
    cc = float(c_cp)
    Fh = cc * (softplus(gh, eta) * f_next[:, :-1] - softplus(-gh, eta) * f_next[:, 1:])
    Fv = cc * (softplus(gv, eta) * f_next[:-1, :] - softplus(-gv, eta) * f_next[1:, :])
    return Fh, Fv


def absorption_ratio(sigma, ft, p: V6Params, bulk_frac: float = 0.5) -> float:
    """``gamma(1 - sigma)/varsigma`` in the bulk.  Must be ``<< 1``; see the module docstring.

    Above 1 the vacuum penalty, which exists only to represent the free surface, is also setting
    the scale *inside* the specimen: the potential is suppressed by that ratio everywhere and the
    body does not move, with nothing raised and eq:xd_potential_bound still comfortably satisfied.

    Two choices here, both learned from the sweep this feeds.  "Bulk" means at least **half** of
    ``f_max``, not a token 5%: the quantity the condition is about is the *interior* value, and a
    low threshold admits rim pixels that are on their way to vacuum.  And the statistic is the
    **median** over that set rather than the max, because a max is set by whichever pixel happens
    to sit just inside the threshold and is therefore an artefact of where the threshold was put.
    With those two, the ratio is monotone in ``f_ref`` and crosses 1 exactly where the measured
    contraction switches on -- which is what makes it a usable warning rather than a number.
    """
    ft = np.asarray(ft, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    bulk = ft >= bulk_frac * float(p.f_max)
    if not bulk.any():
        return 0.0
    return float(p.gamma * np.median(1.0 - sigma[bulk]) / p.varsigma())


# --- the step ------------------------------------------------------------------------------

def step(f, r_values, angle_rad: float, p: V6Params, _decay_last: bool = False):
    """One measurement step.  Steps 1 to 4 are v5's verbatim; only 5 and 6 changed."""
    f = np.asarray(f, dtype=float)

    cIdelta, I_p = accumulate_dose(f, r_values, angle_rad, p.I0, p.c)   # 1
    dw = 1.0 - np.exp(-cIdelta)                                          # 2
    ft = f * p.decay_factor(I_p)                                         # 3
    lost = float(f.sum() - ft.sum())

    phi, Pi, sigma = compaction_potential(ft, dw, p)                     # 4
    f_next, colsum_err = implicit_transport(ft, phi, p.c_cp, p.eta)      # 5 and 6

    if _decay_last:
        # THE WRONG ORDER, run rather than reasoned about, so check (h) can measure the leak.
        # Transport the undecayed field and decay afterwards. Antisymmetry still makes the
        # unweighted flux sum vanish, so this is NOT a loss of telescoping; what it costs is the
        # common factor at the two ends of each face, leaving sum_faces (e_p - e_q) F_{p->q} in
        # the total, which is zero only for a uniform decay. Never reachable from simulate().
        f_next, colsum_err = implicit_transport(f, phi, p.c_cp, p.eta)
        f_next = f_next * p.decay_factor(I_p)

    Fh, Fv = face_fluxes(f_next, phi, p.c_cp, p.eta)
    nr, nc = f.shape
    yy, xx = np.mgrid[0:nr, 0:nc]
    c = (nr - 1) / 2.0
    rad = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    core, rim = rad < 0.25 * nr, (rad >= 0.30 * nr) & (rad < 0.40 * nr)
    pr = (float(phi[core].mean()) / float(phi[rim].mean())
          if rim.any() and abs(float(phi[rim].mean())) > 1e-300 else float("nan"))
    gmax = max(float(np.abs(phi[:, 1:] - phi[:, :-1]).max()) if nc > 1 else 0.0,
               float(np.abs(phi[1:, :] - phi[:-1, :]).max()) if nr > 1 else 0.0)
    info = StepInfo6(mass=float(f_next.sum()), lost=lost,
                     mass_residual=abs(float(f_next.sum()) - float(ft.sum())),
                     dw_max=float(dw.max()), I_max=float(I_p.max()),
                     state_min=float(f_next.min()), max_g=gmax,
                     flux_sum=float(Fh.sum() + Fv.sum()),
                     phi_max=float(phi.max()), phi_core_rim=pr,
                     dP_max=gmax, rest_rate=float(p.c_cp * p.eta * np.log(2.0)),
                     colsum_err=colsum_err,
                     absorption_ratio=absorption_ratio(sigma, ft, p))
    return f_next, info


def simulate(theta, seq, p: V6Params, image_res: int,
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


# --- reporting ------------------------------------------------------------------------------

def half_mass_radius(f, centroid=None) -> float:
    """Half-mass radius in pixels, about ``centroid`` if given and the field's own otherwise.

    subsec:assumptions asks for shrinkage to be reported as a **fixed-centroid half-mass
    radius**, "because attenuation loss contaminates moments and coarse grids quantize support
    radii".  Fixed-centroid so translation does not read as contraction; half-mass rather than
    the 99% support radius because the support radius is the one a coarse grid quantizes worst.
    Pass ``theta``'s centroid to get the statistic the note means.
    """
    f = np.asarray(f, dtype=float)
    nr, nc = f.shape
    yy, xx = np.mgrid[0:nr, 0:nc]
    w = np.clip(f.ravel(), 0.0, None)
    if w.sum() <= 0.0:
        return float("nan")
    if centroid is None:
        m = f.sum()
        if m <= 0.0:
            return float("nan")
        centroid = ((xx * f).sum() / m, (yy * f).sum() / m)
    cx, cy = centroid
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).ravel()
    o = np.argsort(r)
    cs = np.cumsum(w[o])
    return float(r[o][np.searchsorted(cs, 0.5 * cs[-1])])


def centroid_of(f):
    """``(cx, cy)`` in pixel coordinates, for pinning :func:`half_mass_radius`."""
    f = np.asarray(f, dtype=float)
    nr, nc = f.shape
    yy, xx = np.mgrid[0:nr, 0:nc]
    m = float(f.sum())
    if m <= 0.0:
        return (float("nan"), float("nan"))
    return (float((xx * f).sum() / m), float((yy * f).sum() / m))


# --- invariants -------------------------------------------------------------------------
# There is no test suite in this repo (see CLAUDE.md), so this is the verification path, in the
# "run the module" style of degrade_v2.py and tomography_3d.py. v5 broke that chain -- it has no
# check_* of its own and delegates to degrade_v5_uq, which needs pyomo. v6 restores it: every
# check below is pure numpy/scipy and runs in seconds without a solver.
#
# What is asserted and what is only reported is a deliberate distinction. subsec:assumptions'
# "Numerical checks" paragraph lists six; five are reachable without the estimation model, and
# one of those five -- the I0 = 0 identity -- is not true as written (see the module docstring),
# so it is measured against its prediction and the assertion is moved to the scaling law.

_LOCALITY_MSG = (
    "prop:xd_locality: a POINTWISE antisymmetric flux cannot move a uniformly damaged bulk, a "
    "NONLOCAL one can. If this fails the potential is not doing the job it exists to do.")


def _uniform_damage_pair(image_res: int, p: V6Params, dw_level: float = 0.05,
                         c_cp: float = 3.0, n_steps: int = 8):
    """Uniform damage on a hard disc, driven by ``phi`` and by ``Pi``.  prop:xd_locality's own
    configuration, and measured by its own statistic.

    The hypothesis is that the state and the converted fraction are both constant on a set; a
    hard-edged disc with a constant ``dw`` is exactly that.  ``edge=0`` matters -- a smoothed
    interface hands the pointwise driver the turning point it needs, which is the thing being
    denied it.  The pointwise driver is ``(l/Delta)^2 Pi``, its own small-reach limit, so the two
    run at matched amplitude and the contrast is about locality rather than about size.

    **The statistic is the interior residual, not a radius.**  The proposition concludes that a
    pixel of ``S`` whose four neighbours also lie in ``S`` is left at ``ft`` exactly, so
    ``max |f_next - ft|`` over that set is the claim itself: machine zero under a pointwise
    driver, at any grid, and nonzero under the potential.  A radius cannot carry the assertion --
    coarse grids quantize it, and at grid 48 the pointwise run reads ``+0.72%`` on the half-mass
    radius while its interior residual is ``1.1e-15``, i.e. while nothing has moved at all.  That
    quantization is the reason subsec:assumptions asks for the half-mass radius in the first
    place, and it is still only a reporting statistic.

    ``reach`` is resolved to the specimen radius rather than pinned in pixels, so the physics is
    the same at every grid; ``c_cp`` is raised above the tab default because this is a diagnostic
    configuration and the signal should not be marginal.
    """
    theta = _disc(image_res, edge=0.0)
    # a = b = 0: pure transport, mass conserved. reach=None -> R, so the check is grid-invariant.
    p = resolve(replace(p, a=0.0, b=0.0, c_cp=float(c_cp), reach=None), theta)
    ctr = centroid_of(theta)
    r0 = half_mass_radius(theta, ctr)
    scale = 1.0 / p.varsigma()                        # (l/Delta)^2, the pointwise limit of phi
    out = {}
    for name in ("nonlocal", "pointwise"):
        f = theta.copy()
        interior = 0.0
        for k in range(n_steps):
            dwk = np.where(f > 1e-12, float(dw_level), 0.0)
            drv = (compaction_potential(f, dwk, p)[0] if name == "nonlocal"
                   else scale * (dwk * f / float(p.f_max)))
            f_next, _cs = implicit_transport(f, drv, p.c_cp, p.eta)
            if k == 0:
                occ = f > 1e-12
                inner = occ.copy()                    # pixels all of whose neighbours are in S
                inner[1:, :] &= occ[:-1, :]
                inner[:-1, :] &= occ[1:, :]
                inner[:, 1:] &= occ[:, :-1]
                inner[:, :-1] &= occ[:, 1:]
                interior = float(np.abs(f_next - f)[inner].max()) if inner.any() else float("nan")
            f = f_next
        out[name] = dict(dR=100.0 * (half_mass_radius(f, ctr) - r0) / r0,
                         interior=interior,
                         drift=abs(float(f.sum()) - float(theta.sum())) / float(theta.sum()))
    return out


def check_invariants(image_res: int = 64, n_steps: int = 12, verbose: bool = True):
    """Assert what section 3 claims for v6.  Returns a dict of measured numbers.

    (a) **Exact mass conservation** (prop:xd_mass).  ``a = b = 0`` removes eq:xd_decay and then
        the discrete attenuation sum is conserved to machine precision for *any* ``c_cp``: the
        transport matrix has unit column sums, which is the discrete statement of the
        antisymmetry of eq:xd_flux.  With ``a > 0`` the whole change must be eq:xd_decay, and
        the residual is measured against the field as it stood at the START of the step -- the
        caution in the proposition's own proof, since against the post-transport field it is
        identically zero in either ordering and confirms nothing.

    (b) **Exact collapse at c_cp = 0** to eq:xd_implicit_accumulation,
        ``f_K = theta exp(-a sum_k I_k - b sum_k I_k^2)``.  No branch in the code produces this:
        ``c_cp = 0`` makes every rate zero, the operator is exactly the identity, and the solve
        returns its right-hand side.

    (c) **The I0 = 0 identity, which does NOT hold.**  subsec:system says it must hold bitwise.
        softplus is ``eta log 2`` at zero, so a motionless field still diffuses; the leak is
        reported next to that prediction and the ASSERTION is the scaling law, that the leak is
        proportional to ``eta`` and therefore is only the smoothing.  See the module docstring.

    (d) **eq:xd_potential_bound**, ``max phi <= (l/Delta)^2 max Pi``.  Inherited: v5's
        ``compaction_potential`` raises on violation, so reaching the end of a run is the pass.

    (e) **Positivity and conservation of the operator itself**: ``min f > 0`` along the whole
        trajectory and unit column sums.  The M-matrix gives both with no step-size condition,
        which is what let ``C_k`` be deleted.

    (f) **prop:xd_locality**, the reason the potential exists at all.

    (g) **Step 4 is v5's**, bit for bit, which is what licenses calling v5's step-4 functions
        with a ``V6Params``.

    (h) **The step ordering**, run rather than argued: transport-then-decay leaks where
        decay-then-transport does not.
    """
    import degrade_v5 as _v5

    out = {}
    theta = scale_to_optical_depth(_phantom(image_res), 1.1, image_res)
    seq = _demo_sequence(n_steps)
    base = V6Params(reach=7.0, gamma=100.0, f_ref_frac=0.002, eta=1e-3)
    say = (lambda *a: print(*a)) if verbose else (lambda *a: None)

    say("v6 invariants: grid %d, %d steps, c_cp=%.3g, eta=%.3g"
        % (image_res, n_steps, base.c_cp, base.eta))

    # (a) ---------------------------------------------------------------------------------
    f, infos = simulate(theta, seq, replace(base, a=0.0, b=0.0), image_res)
    out["mass_drift"] = abs(float(f.sum()) - float(theta.sum())) / float(theta.sum())
    f_d, infos_d = simulate(theta, seq, base, image_res)
    out["decay_residual"] = max(i.mass_residual for i in infos_d)
    out["colsum_err"] = max(i.colsum_err for i in infos_d)
    say("  (a) mass drift at a=b=0            %.3e   (any c_cp; prop:xd_mass)"
        % out["mass_drift"])
    say("      per-step transport residual    %.3e   (with a>0: all change is eq:xd_decay)"
        % out["decay_residual"])
    assert out["mass_drift"] < 1e-13, out["mass_drift"]
    assert out["decay_residual"] < 1e-11, out["decay_residual"]

    # (b) ---------------------------------------------------------------------------------
    f0, infos0 = simulate(theta, seq, replace(base, c_cp=0.0), image_res)
    Isum = np.zeros_like(theta)
    Isq = np.zeros_like(theta)
    g = theta.copy()
    for angle_deg, offset, n_beams in seq:
        _dq, I_p = accumulate_dose(g, bundle_r_values(float(offset), int(n_beams), image_res),
                                   float(np.deg2rad(float(angle_deg))), base.I0, base.c)
        Isum += I_p
        Isq += I_p ** 2
        g = g * base.decay_factor(I_p)
    closed = theta * np.exp(-base.a * Isum - base.b * Isq)
    out["collapse"] = float(np.abs(f0 - closed).max())
    say("  (b) c_cp=0 collapse to eq:xd_implicit_accumulation   %.3e" % out["collapse"])
    assert out["collapse"] < 1e-12, out["collapse"]

    # (c) ---------------------------------------------------------------------------------
    leaks = {}
    for eta in (base.eta, base.eta / 10.0):
        fz, _iz = simulate(theta, seq, replace(base, I0=0.0, eta=eta), image_res)
        leaks[eta] = float(np.abs(fz - theta).max())
    hi, lo = base.eta, base.eta / 10.0
    out["I0_leak"] = leaks[hi]
    out["I0_leak_ratio"] = leaks[hi] / leaks[lo] if leaks[lo] > 0 else float("inf")
    out["rest_rate"] = float(base.c_cp * base.eta * np.log(2.0))
    fz0, _ = simulate(theta, seq, replace(base, I0=0.0, c_cp=0.0), image_res)
    out["I0_leak_at_ccp0"] = float(np.abs(fz0 - theta).max())
    say("  (c) I0=0 leak at c_cp=%.3g, eta=%.0e   %.3e   REPORTED, not asserted"
        % (base.c_cp, hi, out["I0_leak"]))
    say("      the same at c_cp=0                    %.3e   (subsec:system says these must match)"
        % out["I0_leak_at_ccp0"])
    say("      resting directed rate c_cp*eta*log2   %.3e   phi_eta(0) is not 0 -- see docstring"
        % out["rest_rate"])
    say("      leak(eta)/leak(eta/10)                %.3f     (asserted ~10: the leak IS the "
        "smoothing)" % out["I0_leak_ratio"])
    assert out["I0_leak_at_ccp0"] == 0.0, out["I0_leak_at_ccp0"]
    assert 9.0 < out["I0_leak_ratio"] < 11.0, out["I0_leak_ratio"]

    # (d) and (e) -------------------------------------------------------------------------
    out["phi_bound_ok"] = True       # compaction_potential raises; reaching here is the pass
    out["f_min"] = min(i.state_min for i in infos_d)
    say("  (d) eq:xd_potential_bound satisfied every step       (v5's guard raises otherwise)")
    say("  (e) min f over the trajectory  %.3e  > 0   |colsum-1|  %.3e"
        % (out["f_min"], out["colsum_err"]))
    assert out["f_min"] > 0.0, out["f_min"]
    assert out["colsum_err"] < 1e-12, out["colsum_err"]

    # (f) ---------------------------------------------------------------------------------
    loc = _uniform_damage_pair(image_res, base)
    out["locality_nonlocal_interior"] = loc["nonlocal"]["interior"]
    out["locality_pointwise_interior"] = loc["pointwise"]["interior"]
    out["locality_nonlocal_pct"] = loc["nonlocal"]["dR"]
    out["locality_pointwise_pct"] = loc["pointwise"]["dR"]
    say("  (f) uniform damage, interior residual: pointwise %.3e   nonlocal %.3e"
        % (out["locality_pointwise_interior"], out["locality_nonlocal_interior"]))
    say("      half-mass radius (reported only): pointwise %+7.3f%%   nonlocal %+7.3f%%"
        % (out["locality_pointwise_pct"], out["locality_nonlocal_pct"]))
    say("      mass drift in both               %.1e / %.1e" % (loc["nonlocal"]["drift"],
                                                                loc["pointwise"]["drift"]))
    assert out["locality_pointwise_interior"] < 1e-12, (_LOCALITY_MSG, loc)
    assert out["locality_nonlocal_interior"] > 1e-6, (_LOCALITY_MSG, loc)
    assert out["locality_nonlocal_pct"] < 0.0, (_LOCALITY_MSG, loc)
    assert loc["nonlocal"]["drift"] < 1e-13 and loc["pointwise"]["drift"] < 1e-13, loc

    # (g) ---------------------------------------------------------------------------------
    p6 = resolve(base, theta)
    p5 = _v5.V5Params(I0=p6.I0, c=p6.c, a=p6.a, b=p6.b, c_cp=p6.c_cp, reach=p6.reach,
                      gamma=p6.gamma, f_ref_frac=p6.f_ref_frac, f_max=p6.f_max)
    dq, I_p = accumulate_dose(theta, bundle_r_values(0.0, 0, image_res), 0.0, p6.I0, p6.c)
    ft = theta * p6.decay_factor(I_p)
    a6 = compaction_potential(ft, 1.0 - np.exp(-dq), p6)
    a5 = _v5.compaction_potential(ft, 1.0 - np.exp(-dq), p5)
    out["step4_vs_v5"] = max(float(np.abs(x - y).max()) for x, y in zip(a6, a5))
    say("  (g) step 4 against degrade_v5          %.3e   (phi, Pi and sigma, bit for bit)"
        % out["step4_vs_v5"])
    assert out["step4_vs_v5"] == 0.0, out["step4_vs_v5"]

    # (h) ---------------------------------------------------------------------------------
    pw = replace(base, a=0.2)
    rv = bundle_r_values(0.0, 0, image_res)
    right, _ri = step(theta, rv, 0.0, resolve(pw, theta))
    wrong, _wi = step(theta, rv, 0.0, resolve(pw, theta), _decay_last=True)
    _dq2, I2 = accumulate_dose(theta, rv, 0.0, pw.I0, pw.c)
    expected = float((theta * pw.decay_factor(I2)).sum())
    out["order_right"] = abs(float(right.sum()) - expected)
    out["order_wrong"] = abs(float(wrong.sum()) - expected)
    say("  (h) decay-then-transport unexplained mass  %.3e" % out["order_right"])
    say("      transport-then-decay                   %.3e   (the ordering is load-bearing)"
        % out["order_wrong"])
    assert out["order_right"] < 1e-11, out["order_right"]
    assert out["order_wrong"] > 1e3 * max(out["order_right"], 1e-15), (out["order_right"],
                                                                       out["order_wrong"])

    say("  ALL ASSERTED CHECKS PASS  (c) is reported, not asserted -- the identity is false")
    return out


if __name__ == "__main__":
    import sys

    _res = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    _steps = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    check_invariants(image_res=_res, n_steps=_steps)
