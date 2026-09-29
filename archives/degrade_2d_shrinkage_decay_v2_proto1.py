"""v2 damage model: dose accumulation, saturating response, decay, and elastic transport.

Implements section 3.2 ("The system") of ``xray_degradation.tex`` at commit ``a2c5ee2`` of the
manuscript repo. Where v1 (:func:`senDOE.helpers.dose.degradation_dose_response`) is a pure *local
sink* -- mass vanishes in place, so the sample fades but never changes shape -- v2 separates the
dose *accumulation* from the dose *response*, and adds a mass balance so mass also **moves**.

State is ``(f, Q)``, parameter ``theta = f_0``, ``Q_0 = 0``. Twelve steps; the ones this module
implements are 1-10, the dynamics map ``M``:

1.  photon balance   ``I_p = I0*exp(-sum_{m<i} f_pm*delta_pm)``, Beer-Lambert over the *upstream*
    pixels of each ray; contributions of simultaneous rays add.  eq:xd_local_intensity.
2.  dose             ``Q <- Q + c_q*I_p*delta_p``, in grays.  Deliberately **no local f factor**:
    specific absorbed energy is ``mu_en*Psi/rho`` and ``mu_en/rho`` is density independent, so it
    cancels.  eq:xd_dose_state.
3.  energy density   ``E = f*Q``, algebraic and read-only; no later step uses it.
4.  response         ``omega(Q) = omega_inf + (1-omega_inf)*exp(-Q/Q_c)``, saturating with a
    floor -- what v1's unbounded geometric decay lacked.  eq:xd_response.
5.  converted        ``dw = 1 - omega(Q_new)/omega(Q_old)``.  **Relative, not absolute.**
6.  eigenstrain      ``eps* = -0.5*c_cp*dw*I``, trace ``-c_cp*dw``.  The only point damage enters
    the mechanics.
7.  equilibrium      ``K dx = B dw`` -- plane-stress linear elasticity, Q1 elements.  ``K`` depends
    only on grid, ``E``, ``nu`` and the BCs, so it is assembled and factored **once**.
8.  face fluxes      ``F_{p->q} = (1/Delta)*(v_+ ftilde_p - v_- ftilde_q)`` with the smoothed
    upwind split, antisymmetric by construction.
9.  mass loss        ``ftilde = f*exp(-a*I_p - b*I_p^2)`` -- the v1 multiplicative decay, driven
    by instantaneous local fluence rather than accumulated dose, with no floor.  eq:xd_decay.
10. mass balance     ``f_next = ftilde - sum_q F[ftilde]``.  eq:xd_mass_transport.

11. observation      ``y_k = C_v^loc f_k`` -- the ray integrals of the field as it stood at the
    *start* of step k, with ``H = I`` (the cheap variant of eq:xd_obs_damage, which inflates the
    noise covariance instead of smoothing the detector).  Off by default: pass
    ``record_observations=True`` to :func:`simulate`.  This is what :mod:`degrade_2d_shrinkage_decay_v2_proto1`
    reconstructs from.

Step 12 (the box and budget constraints) belongs to the estimation NLP, not here.

**Step 9 runs before step 10, and the order is not cosmetic.** The mechanism is easy to state
wrongly, so precisely: the *unweighted* flux sum is zero in every case, whatever the ordering --
telescoping follows from the antisymmetry of eq:xd_flux alone and does not care when the decay
is applied. What the wrong order changes is the *weight* each end of a face carries into the
total. Decaying first means every face sees one common factor and the fluxes still cancel in
pairs; transporting first and decaying after leaves the flux contributing
``sum_faces (e_p - e_q) F_{p->q}``, which vanishes only if the decay factor ``e`` is uniform --
and it never is, because ``I_p`` varies. :func:`check_invariants` runs both orderings rather
than trusting either: measured at 0.0 for the correct order against ~9e-4 for the reverse.

Pure numpy + scipy over the repo's own cached ray geometry -- no Streamlit, no Pyomo, no solver.

Discretisation choices that differ from the manuscript's scratch reference, and why
------------------------------------------------------------------------------------
* **The photon balance is ray-based, not a plane wave.**  The reference script integrates a full
  parallel plane wave by bilinear resampling because it was verifying physics claims on a disc.
  eq:xd_local_intensity is stated over the pixels a ray crosses with chord lengths from
  ``C_v^loc``, which is exactly what :func:`senDOE.helpers.rays.ray_geometry` returns -- so this module
  reuses it and the (angle, offset, n_beams) bundles the rest of the app is built around.
* **Pixel units.**  ``dx = 1`` here, matching the app's geometry, where the reference used a
  ``[-1,1]`` box.  Every dimensionless diagnostic (Courant number, mass drift, the collapse
  error) is unaffected; only the natural size of ``c_q`` changes.  The ``1/Delta`` of
  eq:xd_flux is applied here when assembling the divergence.  (The spec omitted it until
  ``a2c5ee2``; this module always had it, so nothing changed when it was added.)
* **The box constraints of eq:xd_box / eq:xd_budget are NOT applied in the dynamics.**  They are
  inequality constraints of the estimation NLP, and step 12 says explicitly that clipping inside
  the forward map destroys both the conservation and the collapse.  Violations are reported in
  :class:`StepInfo` instead.
* **``eps_up`` is the constant of eq:xd_upwind and defaults to zero; ``eps_rel`` is the spec's
  relative form and is what the NLP needs.**  A constant ``eps_up`` does not vanish with the
  flow: at ``v = 0`` it still passes ``0.5*eps*(f_p - f_q)`` across every face, which breaks both
  the collapse and the ``I0 = 0`` identity.  Zero is exact, and is what the spec endorses for a
  simulator doing no sensitivity extraction -- but ``sqrt(v^2)`` is not differentiable at ``v =
  0``, so an NLP cannot use it.  ``eps_rel > 0`` selects ``eps_up^2 = eps_rel^2 * mean(|dx_k|^2)``
  instead, which is smooth, vanishes exactly where the flow does, and is grid independent.
  It is carried as ``eps_up^2`` throughout and never square-rooted, because eq:xd_upwind only
  ever needs the square -- and the square is a polynomial in ``dx_k``, where the root would
  reintroduce a kink at rest.

  **Two choices here are ours, not the note's.**  The spec writes ``eps_rel*||dx_k||`` with
  "a smooth norm over the grid" and never pins the norm, and it gives ``eps_rel`` no value
  anywhere.  The RMS is chosen because the plain 2-norm scales with the pixel count for a fixed
  displacement field, which would make the note's own two grids (64x64 for the checks, 10 for the
  closed-loop runs) incomparable; ``eps_rel`` is then relative to a *typical* displacement.
  ``1e-3`` is a working value with nothing behind it.

  **And the relative form has a defect the note does not record.**  Because ``eps_up^2`` is a
  positive-definite quadratic form in ``dx_k``, ``sqrt(v^2 + eps_up^2)`` is a norm of ``dx_k``
  and is NOT differentiable at ``dx_k = 0`` -- which a constant ``eps_up`` never was.  It is
  reachable: both of the spec's off switches put the field exactly at rest.  The estimation NLP
  handles it structurally by dropping the transport block there; see :mod:`degrade_2d_shrinkage_decay_v2_proto1`.

Tuning note
-----------
``c_cp`` serves two behaviours that want opposite values: shrinkage wants it large, a beam channel
that does not refill wants it small.  Measured upstream, against a channel floor of 0.20: a
constant ``c_cp`` gives 0.57 and a modulus that follows density gives 0.56, but a **dose-dependent**
``c_cp = c0*omega(Q)`` gives 0.35.  ``c_cp`` enters the load only, so nothing cancels it -- unlike
``E``, which cancels out of ``K dx = B dw`` almost exactly (1 against 1000 moves the displacement
by 9e-15).  The dose-dependent form is not part of section 3.2 and is not implemented here.

Reconstruction
--------------
The Pyomo reconstruction of this prototype is in the second half of this file, under the
``=== reconstruction ===`` banner.  Its own notes follow.

Pyomo transcription of the v2 damage model, and reconstruction of ``theta = f_0`` from it.

:mod:`degrade_2d_shrinkage_decay_v2_proto1` is the forward simulator: numpy, fast, and validated against the manuscript
by ``check_invariants`` / ``check_reference_numbers``.  This module writes the *same* ten steps
as algebraic constraints so IPOPT can run them backwards -- estimate the undamaged reference
field from the projections -- and so k_aug can differentiate the result, which is
eq:xd_composed_jacobian.

It is the v2 counterpart of :mod:`senDOE.models.tomography_pyomo_2d_pixel_intersection_uq`, and deliberately not a modification of it:
``senDOE/`` is a verbatim vendored snapshot (see ``SENDOE_VENDOR.md``) and v1's Pyomo model
carries only ``image[ix, iy, time]``, where v2 needs a dose state, a displacement field and a
mass balance as well.

What makes this checkable
-------------------------
The measured data comes from the numpy simulator, not from a Pyomo forward solve, so the two
implementations stay independent and can be compared.  :func:`check_forward` does exactly that,
at two levels:

* **residual** -- pin every variable to a ``degrade_2d_shrinkage_decay_v2_proto1.simulate`` trajectory and evaluate every
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
  ``degrade_2d_shrinkage_decay_v2_proto1.check_invariants``).  It is carried squared, as one variable per step, because
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
  ``senDOE.helpers.dose.degradation_dose_response`` shares the split, so it moves the 2D live picture
  and the 3D simulator, and because ``degrade_2d_shrinkage_decay_v2_proto1`` turns out to BE the "independent reproduction"
  quoted at the note's line 609 (it returns -0.93 / -3.63 / -7.95 for Rg, the printed digits
  exactly), the fix also edits three numbers already written into section 3.4.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyomo.environ as pyo
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from senDOE.helpers.rays import bundle_r_values, ray_geometry, ray_line_integral
from senDOE.helpers.dose import accumulate_dose, scale_to_optical_depth
from senDOE.helpers.rays import measurement_rays
from senDOE.helpers.shape_metrics import radius_of_gyration
from senDOE.helpers.solvers import solve_with_fallback





@dataclass(frozen=True)
class V2Params:
    """Parameters of the v2 model.  Defaults are a visible-but-gentle working point."""

    I0: float = 1.0          # incident beam intensity; I0 = 0 is the undamaged limit (M = id)
    c_q: float = 0.1         # fluence x path length -> absorbed dose
    Q_c: float = 1.0         # characteristic dose of the response
    omega_inf: float = 0.2   # residual attenuation fraction, in [0, 1)
    c_cp: float = 0.3        # fraction of created void the matrix closes: 1 compliant, 0 rigid
    # Mass loss, eq:xd_decay. Replaces the old gamma_esc sink: mass now leaves by the v1
    # multiplicative decay, driven by instantaneous fluence. a = b = 0 is the switch that
    # conserves mass exactly. b is kept at 0 -- see (M1) in the note on fractionation.
    a: float = 0.05
    b: float = 0.0
    eps_up: float = 0.0      # absolute upwind smoothing; 0 is exact (see the module docstring)
    eps_rel: float = 0.0     # relative smoothing, eq:xd_upwind's flow-scaled form. > 0 REPLACES
                             # eps_up with eps_rel*||dx_k||_rms, which vanishes with the flow.
                             # Needed only by the estimation NLP, which cannot differentiate
                             # sqrt(v^2) at v = 0; leave at 0 for forward simulation.
    E0: float = 1.0          # modulus of the undamaged matrix
    nu: float = 0.3          # Poisson ratio
    e_min_ratio: float = 1e-6  # ersatz soft background, E_min/E_0, so the free surface needs no
                               # explicit meshing
    clamp_bottom: bool = False  # substrate (clamp one edge) vs free-floating body
    dx: float = 1.0          # pixel pitch, in the app's geometry units

    def decay_factor(self, I_p):
        """eq:xd_decay's multiplier ``exp(-a*I - b*I^2)``.  Strictly positive, so f stays > 0."""
        I_p = np.asarray(I_p, dtype=float)
        return np.exp(-self.a * I_p - self.b * I_p ** 2)


@dataclass
class StepInfo:
    """Per-step diagnostics.  Cheap to compute and the only window into whether a run is sane."""

    courant: float        # max|u|/dx -- keep under ~0.5 or the upwind transport loses positivity
    mass: float           # sum_p f_p, the total attenuation M_k
    lost: float           # mass removed by eq:xd_decay this step; transport moves, never removes
    dw_max: float         # largest converted fraction anywhere
    q_max: float          # largest accumulated dose anywhere
    f_min: float          # most negative f, if the transport overshot
    energy_max: float     # max of the algebraic energy density E = f*Q
    eps_sq: float = 0.0   # the eq:xd_upwind smoothing actually used this step, squared


def omega(Q, omega_inf: float, Q_c: float):
    """Retained attenuation fraction -- eq:xd_response.  ``omega(0) = 1`` exactly."""
    return omega_inf + (1.0 - omega_inf) * np.exp(-np.asarray(Q, dtype=float) / Q_c)


def _q1_matrices(h: float, nu: float):
    """Bilinear Q1 element stiffness (8x8) and eigenstrain load (8,), 2x2 Gauss, plane stress.

    The load is per unit eigenstrain amplitude ``s``: ``fe = sum_g B^T D [s,s,0] detJ``.
    """
    D = (1.0 / (1.0 - nu ** 2)) * np.array(
        [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]]
    )
    g = 1.0 / np.sqrt(3.0)
    Ke = np.zeros((8, 8))
    Le = np.zeros(8)
    for xi in (-g, g):
        for et in (-g, g):
            dNdxi = 0.25 * np.array([-(1 - et), (1 - et), (1 + et), -(1 + et)])
            dNdet = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
            dNdx, dNdy = dNdxi * (2.0 / h), dNdet * (2.0 / h)   # J = (h/2) I
            B = np.zeros((3, 8))
            B[0, 0::2] = dNdx
            B[1, 1::2] = dNdy
            B[2, 0::2] = dNdy
            B[2, 1::2] = dNdx
            detJ = (h / 2.0) ** 2
            Ke += B.T @ D @ B * detJ
            Le += B.T @ D @ np.array([1.0, 1.0, 0.0]) * detJ
    return Ke, Le


class ElasticSolver:
    """``K u = B dw`` on the pixel grid, assembled and factored once.

    Under assumption (S2) the stiffness is built from the **initial** density and held fixed: a
    small-strain solve on a fixed reference domain.  The consequence, which is a known limit
    rather than an oversight, is that the skeleton never learns that material has gone.

    An ersatz soft background (``E_min/E_0 ~ 1e-6`` outside the sample) means the free surface
    needs no explicit meshing.  A free-floating body has three dofs pinned purely to kill the
    rigid-body modes -- the eigenstrain load is self-equilibrated, so the reactions there are
    zero and the solution is the correct free one.
    """

    def __init__(self, dens, nu: float, E0: float, e_min_ratio: float, dx: float,
                 clamp_bottom: bool = False):
        dens = np.asarray(dens, dtype=float)
        nr, nc = dens.shape
        if nr != nc:
            raise ValueError("ElasticSolver expects a square grid, got %r" % (dens.shape,))
        self.n = nr
        nn = nr + 1                      # nodes per side
        self.nn = nn
        self.ndof_total = 2 * nn * nn

        Ke, Le = _q1_matrices(dx, nu)
        self._Ke = Ke
        self._Le = Le

        # element (i, j) == pixel (i, j); node (i, j) -> dofs 2*(i*nn+j), +1
        edof = np.zeros((nr * nc, 8), dtype=int)
        for i in range(nr):
            for j in range(nc):
                nodes = ((i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j))
                edof[i * nc + j] = [d
                                    for (a, b) in nodes
                                    for d in (2 * (a * nn + b), 2 * (a * nn + b) + 1)]
        self._edof = edof

        e_min = e_min_ratio * E0
        self._Ee = e_min + (E0 - e_min) * np.clip(dens.ravel(), 0.0, 1.0)

        rows = np.repeat(edof, 8, axis=1).ravel()
        cols = np.tile(edof, (1, 8)).ravel()
        vals = (self._Ee[:, None] * Ke.ravel()[None, :]).ravel()
        K = sp.csc_matrix((vals, (rows, cols)),
                          shape=(self.ndof_total, self.ndof_total))
        # Kept so :mod:`degrade_2d_shrinkage_decay_v2_proto1` can write eq:xd_elastic_discrete as Pyomo rows off the
        # *same* matrix this factorisation uses, rather than assembling a second copy that
        # could drift from it.
        self.K = K

        fixed = []
        if clamp_bottom:                                   # substrate: one edge pinned
            for j in range(nn):
                fixed += [2 * j, 2 * j + 1]
        else:                                              # free body: kill rigid modes only
            c = nn // 2
            base = 2 * (c * nn + c)
            fixed += [base, base + 1, 2 * (c * nn + c + 1) + 1]
        self._free = np.setdiff1d(np.arange(self.ndof_total), np.array(fixed, dtype=int))
        self._lu = spla.splu(K[self._free][:, self._free].tocsc())

    def solve_nodal(self, dw, c_cp: float):
        """The raw nodal solution of ``K u = B dw``, length ``ndof_total``.

        Split out of :meth:`solve` so :mod:`degrade_2d_shrinkage_decay_v2_proto1` can pin its Pyomo ``u`` variables to
        exactly what this solves, rather than to a second implementation of the same assembly.
        """
        s = -0.5 * c_cp * np.asarray(dw, dtype=float).ravel()   # eigenstrain amplitude
        be = (self._Ee * s)[:, None] * self._Le[None, :]
        b = np.bincount(self._edof.ravel(), weights=be.ravel(), minlength=self.ndof_total)
        u = np.zeros(self.ndof_total)
        u[self._free] = self._lu.solve(b[self._free])
        return u

    def solve(self, dw, c_cp: float):
        """Per-pixel displacement ``(ux, uy)`` driven by the damage eigenstrain."""
        u = self.solve_nodal(dw, c_cp)
        nn = self.nn
        ux = u[0::2].reshape(nn, nn)
        uy = u[1::2].reshape(nn, nn)
        # nodal -> pixel centres
        cx = 0.25 * (ux[:-1, :-1] + ux[:-1, 1:] + ux[1:, :-1] + ux[1:, 1:])
        cy = 0.25 * (uy[:-1, :-1] + uy[:-1, 1:] + uy[1:, :-1] + uy[1:, 1:])
        return cx, cy


def upwind_flux_divergence(f, vx, vy, dx: float, eps_up: float, eps_sq=None):
    """``sum_q F_{p->q}`` with antisymmetric face fluxes and a smoothed upwind split.

    The split ``v_pm = 0.5*(sqrt(v^2 + eps^2) +- v)`` is smooth in ``v``, where the plain
    ``max(v, 0)`` is not -- differentiability is what the sensitivity extraction needs.  Each
    face contributes ``+F/dx`` to one cell and ``-F/dx`` to its neighbour, so the divergence sums
    to zero over the grid in exact arithmetic: that is what makes mass conservation exact.

    Note the smoothing is not free: at ``v = 0`` the split gives ``v_+ = v_- = eps/2`` rather
    than ``0``, so a stationary field still sees a flux ``0.5*eps*(f_L - f_R)``.  That is an
    ``O(eps)`` numerical diffusion across any gradient, and it is why the exact collapse holds
    only at ``eps_up = 0``; see :func:`check_invariants`.

    ``eps_sq`` overrides ``eps_up ** 2`` when given.  The relative form of eq:xd_upwind is
    naturally a *square* (``eps_rel^2 * mean|dx|^2``), and passing it as one avoids a
    square-root-then-square round trip -- which matters because the root has a kink at rest and
    the square does not.
    """
    f = np.asarray(f, dtype=float)
    e2 = eps_up ** 2 if eps_sq is None else float(eps_sq)

    def split(v):
        s = np.sqrt(v ** 2 + e2)
        return 0.5 * (s + v), 0.5 * (s - v)

    # faces normal to x, between column j and j+1
    vfx = 0.5 * (vx[:, :-1] + vx[:, 1:])
    px, mx = split(vfx)
    Fx = px * f[:, :-1] - mx * f[:, 1:]
    # faces normal to y, between row i and i+1
    vfy = 0.5 * (vy[:-1, :] + vy[1:, :])
    py, my = split(vfy)
    Fy = py * f[:-1, :] - my * f[1:, :]

    div = np.zeros_like(f)
    div[:, :-1] += Fx / dx
    div[:, 1:] -= Fx / dx
    div[:-1, :] += Fy / dx
    div[1:, :] -= Fy / dx
    return div


def step(f, Q, r_values, angle_rad: float, p: V2Params, solver: ElasticSolver,
         _decay_last: bool = False):
    """One measurement step: ``(f, Q) -> (f_next, Q_next, info)``.  Steps 1-10 in order.

    ``_decay_last`` swaps steps 9 and 10 -- transport the undecayed field and decay afterwards.
    It exists only so :func:`check_invariants` can *demonstrate* that the order matters instead
    of asserting it on faith; it is the wrong order and nothing else should use it.
    """
    f = np.asarray(f, dtype=float)
    Q = np.asarray(Q, dtype=float)

    dQ, I_p = accumulate_dose(f, r_values, angle_rad, p.I0, p.c_q)   # 1, 2
    Q_next = Q + dQ
    # 5. Relative converted fraction -- see the module docstring; absolute is the classic bug.
    dw = 1.0 - omega(Q_next, p.omega_inf, p.Q_c) / omega(Q, p.omega_inf, p.Q_c)
    dx_x, dx_y = solver.solve(dw, p.c_cp)                            # 6, 7 -> displacement

    # eq:xd_upwind's smoothing, carried squared.  The relative form is scaled to the flow, so it
    # is exactly 0 wherever dx_k is -- which is what restores the collapse and the I0 = 0
    # identity that a constant eps_up leaks (see check_invariants).
    if p.eps_rel > 0.0:
        eps_sq = p.eps_rel ** 2 * float(np.mean(dx_x ** 2 + dx_y ** 2))
    else:
        eps_sq = p.eps_up ** 2

    if _decay_last:                                                  # deliberately wrong order
        moved = f - upwind_flux_divergence(f, dx_x, dx_y, p.dx, p.eps_up, eps_sq)
        f_next = moved * p.decay_factor(I_p)
        # Deliberately measured the same way as the correct branch: the decay loss of the field
        # as it stood at the START of the step. Transport telescopes in either order, so a loss
        # measured *after* it would come out trivially consistent and hide the defect. What the
        # wrong order actually breaks is that the decay now weights a redistributed field, so
        # this expected loss no longer matches the real change -- and that gap is the leak.
        lost = float(f.sum() - (f * p.decay_factor(I_p)).sum())
    else:
        # 9 then 10: one common decay factor per face, so the fluxes still cancel in pairs and
        # the transport moves no mass at all; the entire change in the total is the decay.
        f_tilde = f * p.decay_factor(I_p)
        lost = float(f.sum() - f_tilde.sum())
        f_next = f_tilde - upwind_flux_divergence(f_tilde, dx_x, dx_y, p.dx, p.eps_up, eps_sq)

    info = StepInfo(
        courant=float(np.sqrt(dx_x ** 2 + dx_y ** 2).max() / p.dx),
        mass=float(f_next.sum()),
        lost=lost,
        dw_max=float(dw.max()),
        q_max=float(Q_next.max()),
        f_min=float(f_next.min()),
        energy_max=float((f_next * Q_next).max()),
        eps_sq=float(eps_sq),
    )
    return f_next, Q_next, info


def simulate(theta, seq, p: V2Params, image_res: int, _decay_last: bool = False,
             record_observations: bool = False, record_trajectory: bool = False):
    """Run a measurement sequence from the undamaged field ``theta``.

    ``seq`` is the app's ``_table_to_seq`` output -- ``(angle_deg, offset, n_beams)`` triples.
    Returns ``(f, Q, infos)``: the final attenuation field, the accumulated dose, and one
    :class:`StepInfo` per step.  The stiffness is factored once, from ``theta``.

    ``record_observations`` appends a fourth return value: step 11, one 1-D array of ray
    integrals per measurement, aligned with that step's ``bundle_r_values``.  Each is read off
    the field as it stood at the **start** of the step, which is the same ``f_k`` the photon
    balance of step 1 integrates -- so the observation and the shielding can never disagree.
    It is ragged across steps, since a bundle near the edge loses rays to the ``|r|`` clamp.

    ``record_trajectory`` appends the whole ``(f_k, Q_k)`` history, ``k = 0..K``, as two lists.
    Nothing in the app needs it; :mod:`degrade_2d_shrinkage_decay_v2_proto1` does, to check its Pyomo transcription of
    these same equations against this one.
    """
    theta = np.asarray(theta, dtype=float)
    f = theta.copy()
    Q = np.zeros_like(f)
    solver = ElasticSolver(theta, p.nu, p.E0, p.e_min_ratio, p.dx, p.clamp_bottom)
    infos = []
    obs = []
    f_hist, Q_hist = [f.copy()], [Q.copy()]
    for angle_deg, offset, n_beams in seq:
        angle_rad = np.deg2rad(float(angle_deg))
        r_values = bundle_r_values(float(offset), int(n_beams), int(image_res))
        if record_observations:
            # Step 11, taken BEFORE the step updates the field: y_k = C_v^loc f_k.
            obs.append(np.array([ray_line_integral(f, r, angle_rad) for r in r_values]))
        f, Q, info = step(f, Q, r_values, angle_rad, p, solver, _decay_last=_decay_last)
        infos.append(info)
        if record_trajectory:
            f_hist.append(f.copy())
            Q_hist.append(Q.copy())
    out = [f, Q, infos]
    if record_observations:
        out.append(obs)
    if record_trajectory:
        out.append((f_hist, Q_hist))
    return tuple(out)


# --- invariants -------------------------------------------------------------------------
# Three properties the model must have.  There is no test suite in this repo (see CLAUDE.md),
# so this is the verification path, in the same "run the module" style as senDOE.models.tomography_3d.

# The three settings of eq:xd_upwind's smoothing, and which of them (b) and (c) hold exactly
# for.  The constant form is the odd one out: it does not vanish with the flow, so it leaks a
# resting diffusion into corners where nothing should be moving at all.
_SMOOTHINGS = (
    ("eps_up = 0 (exact)", dict(eps_up=0.0)),
    ("eps_up = 1e-6 (const)", dict(eps_up=1e-6)),
    ("eps_rel = 1e-3 (flow)", dict(eps_rel=1e-3)),
)
_SUFFIX = {"eps_up = 0 (exact)": "", "eps_up = 1e-6 (const)": "_eps",
           "eps_rel = 1e-3 (flow)": "_rel"}
_EXACT = {"eps_up = 0 (exact)": True, "eps_up = 1e-6 (const)": False,
          "eps_rel = 1e-3 (flow)": True}


def _demo_sequence(n_steps: int = 12):
    """Evenly spaced full-fan projections -- the sequence the manuscript's checks use."""
    return tuple((180.0 * i / n_steps, 0.0, 0) for i in range(n_steps))


def check_invariants(image_res: int = 64, n_steps: int = 12, verbose: bool = True):
    """Assert the invariants of section 3.2.  Returns a dict of measured residuals.

    (a) **Exact mass conservation, and the step ordering that makes it hold.**  ``a = b = 0``
        removes eq:xd_decay, and then the total attenuation is conserved to machine precision
        for *any* ``c_cp`` -- transport moves mass, it never removes it.  With ``a > 0`` the
        whole change in the total must be the decay.  Not because the ordering restores
        telescoping -- the unweighted flux sum is zero either way, that being a property of
        eq:xd_flux alone -- but because decaying first gives both ends of every face one common
        factor, so the pairs still cancel.  Transport-then-decay leaves the flux contributing
        ``sum_faces (e_p - e_q) F_{p->q}``, zero only for a uniform decay.  The wrong order is *run* here rather than reasoned
        about, so the claim is demonstrated.

    (b) **Exact collapse to the model already in use.**  At ``c_cp = 0`` the eigenstrain
        vanishes, so ``Delta x = 0``, every flux with it, and the dynamics reduce to
        eq:xd_decay composed over ``K`` steps -- which is eq:xd_implicit_accumulation,
        ``f_K = f_0*exp(-a*sum_k I_k - b*sum_k I_k^2)``.

        This replaces the old check against ``theta*omega(Q)`` (eq:xd_reference_state).  That
        collapse is **no longer reachable**: the mass-loss channel is the fluence-driven decay
        and is no longer a function of ``omega``, so the old test would now fail against a
        correct implementation.  It would return only if eq:xd_decay were replaced by a sink
        proportional to ``dw``.

    (c) **``I0 = 0`` gives ``M = id``.**  No fluence, so no dose, no response, no eigenstrain,
        no decay, no transport.

    (b) and (c) are exact at ``eps_up = 0`` and, as the spec predicts, at any ``eps_rel`` --
    but *not* at a constant ``eps_up``, which gives ``v_+ = v_- = eps/2`` at ``v = 0``, so a
    motionless field still exchanges ``0.5*eps*(f_p - f_q)`` across every face.  All three are
    run: the constant form is reported (it leaks ~3e-7) and the other two are asserted exact.
    That matters because the estimation NLP cannot use ``eps_up = 0`` -- ``sqrt(v^2)`` has no
    derivative at ``v = 0`` -- so ``eps_rel`` is the only setting that is both differentiable
    and faithful, and this is what establishes it.
    """
    from skimage.data import shepp_logan_phantom
    from skimage.transform import resize

    theta = scale_to_optical_depth(
        resize(shepp_logan_phantom(), (image_res, image_res)).astype(float), 1.1, image_res)
    seq = _demo_sequence(n_steps)
    out = {}

    def say(msg):
        if verbose:
            print(msg)

    # (a) conservation at a = b = 0, for any c_cp --------------------------------------
    say("(a) mass conservation and step ordering")
    worst = 0.0
    for c_cp in (0.0, 0.3, 0.8):
        p = V2Params(a=0.0, b=0.0, c_cp=c_cp)
        f, _, infos = simulate(theta, seq, p, image_res)
        drift = abs(f.sum() - theta.sum()) / theta.sum()
        worst = max(worst, drift)
        say("    a=b=0  c_cp=%.1f   relative drift %.3e   Courant %.3f   min f %+.2e"
            % (c_cp, drift, max(i.courant for i in infos), min(i.f_min for i in infos)))
    out["mass_drift"] = worst
    assert worst < 1e-13, "mass not conserved at a=b=0: the face fluxes are not antisymmetric"

    # ... and with decay on, the entire change must be the decay -- which is what the
    # ordering buys. Run the wrong order too, so the difference is measured, not asserted.
    say("    with a=0.05, the whole change in the total must be eq:xd_decay:")
    totals = {}
    for label, wrong in (("decay then transport (correct)", False),
                         ("transport then decay (wrong)  ", True)):
        p = V2Params(a=0.05, b=0.0, c_cp=0.8)
        f, _, infos = simulate(theta, seq, p, image_res, _decay_last=wrong)
        resid = abs(f.sum() - (theta.sum() - sum(i.lost for i in infos))) / theta.sum()
        totals[wrong] = f.sum()
        out["order_wrong" if wrong else "order_right"] = resid
        say("        %s  unexplained mass %.3e" % (label, resid))
    out["order_total_gap"] = abs(totals[True] - totals[False]) / theta.sum()
    say("        the two orderings' totals differ by %.3e  [1e-4 to 3e-4]"
        % out["order_total_gap"])
    assert out["order_right"] < 1e-13, "the correct order is leaking mass through the flux"
    assert out["order_wrong"] > 1e-6, (
        "swapping steps 9 and 10 left the mass budget exact -- the ordering is not implemented")
    assert out["order_total_gap"] > 1e-6, "the two orderings are indistinguishable"

    # (b) collapse to the composed v1 decay at c_cp = 0 ---------------------------------
    say("(b) collapse to f_K = f_0*exp(-a*sum I - b*sum I^2)   [c_cp = 0]")
    for tag, smooth in _SMOOTHINGS:
        p = V2Params(a=0.05, b=0.01, c_cp=0.0, **smooth)
        solver = ElasticSolver(theta, p.nu, p.E0, p.e_min_ratio, p.dx, p.clamp_bottom)
        f_ref, Q_ref = theta.copy(), np.zeros_like(theta)
        sum_I, sum_I2 = np.zeros_like(theta), np.zeros_like(theta)
        for angle_deg, offset, n_beams in seq:
            rs = bundle_r_values(float(offset), int(n_beams), image_res)
            dQ, I_p = accumulate_dose(f_ref, rs, np.deg2rad(angle_deg), p.I0, p.c_q)
            sum_I += I_p
            sum_I2 += I_p ** 2
            f_ref, Q_ref, _ = step(f_ref, Q_ref, rs, np.deg2rad(angle_deg), p, solver)
        closed = theta * np.exp(-p.a * sum_I - p.b * sum_I2)
        err = float(np.abs(f_ref - closed).max())
        say("    %-22s max|f_K - closed form| = %.3e" % (tag, err))
        out["collapse" + _SUFFIX[tag]] = err
        if _EXACT[tag]:
            assert err < 1e-12, (
                "c_cp = 0 does not reduce to the composed decay at %s" % tag)

    # (c) I0 = 0 is the identity ---------------------------------------------------------
    say("(c) I0 = 0 gives M = identity")
    for tag, smooth in _SMOOTHINGS:
        p = V2Params(I0=0.0, a=0.05, c_cp=0.8, **smooth)
        f, Q, _ = simulate(theta, seq, p, image_res)
        err = float(np.abs(f - theta).max())
        say("    %-22s max|f - theta| = %.3e   max Q = %.3e" % (tag, err, Q.max()))
        out["identity" + _SUFFIX[tag]] = err
        if _EXACT[tag]:
            assert err == 0.0, "I0 = 0 is not the identity at %s" % tag

    say("all invariants hold")
    return out


def check_reference_numbers(image_res: int = 64, verbose: bool = True):
    """Reproduce the manuscript's re-measured numbers on its own test object.

    The invariants are structural -- they would pass for a model that was self-consistent and
    still wrong.  These are the quantitative cross-check against an independently written
    reference (plane-wave, bilinear resampling; this module is ray-based over the app's cached
    geometry), so agreement to a few percent is evidence the port is faithful rather than merely
    coherent.  Run at the reference's settings: a = 0.05, b = 0, peak optical depth 1.1,
    12 projections, nu = 0.3.

    Note the shrinkage baseline moved.  Radius of gyration used to be flat whenever
    ``c_cp = 0``, so it read as pure geometry; the fluence-driven decay fades the field
    *non-uniformly*, because ``I_p`` varies, so ``c_cp = 0`` now registers a small contraction
    with nothing having moved.  Rg mixes contraction with photometric reshaping and has to be
    read against that baseline rather than against zero.
    """
    lin = np.linspace(-1.0, 1.0, image_res)
    X, Y = np.meshgrid(lin, lin)
    disc = scale_to_optical_depth(np.where(np.hypot(X, Y) < 0.55, 1.0, 0.0), 1.1, image_res)
    seq = _demo_sequence(12)
    out = {}

    def say(msg):
        if verbose:
            print(msg)

    def run(**kw):
        p = V2Params(c_q=0.0317, **kw)
        f, _, infos = simulate(disc, seq, p, image_res)
        g0, g1 = radius_of_gyration(disc), radius_of_gyration(f)
        return (100.0 * (g1 - g0) / g0, f.sum() / disc.sum(),
                max(i.courant for i in infos), min(i.f_min for i in infos))

    say("reference numbers (manuscript value in brackets)")
    base, _, _, _ = run(a=0.05, b=0.0, c_cp=0.0)
    out["rg_baseline"] = base
    say("    a=.05 c_cp=0.0  Rg %+7.3f%%  [-0.7%%]  <- photometric reshaping, nothing moved"
        % base)
    for c_cp, want in ((0.3, -3.4), (0.8, -7.6)):
        rg, massf, cfl, fmin = run(a=0.05, b=0.0, c_cp=c_cp)
        out["rg_%.1f" % c_cp] = rg
        say("    a=.05 c_cp=%.1f  Rg %+7.3f%%  [%+.1f%%]   mass left %.1f%% [66%%]   Courant %.2f"
            % (c_cp, rg, want, 100 * massf, cfl))
        assert rg < base, "contraction must shrink the sample beyond the decay baseline"
        assert abs(rg - want) / abs(want) < 0.25, "shrinkage is off the reference value"
        assert fmin >= 0.0, "positivity lost"

    rg, massf, cfl, fmin = run(a=0.0, b=0.0, c_cp=0.8)
    out["rg_conserved"] = rg
    say("    a=b=0 c_cp=0.8  Rg %+7.3f%%  [-6.5%%]   mass left %.6f [1.000000]   Courant %.2f"
        % (rg, massf, cfl))
    assert abs(rg - (-6.5)) / 6.5 < 0.25, "conserved-mass shrinkage is off the reference value"
    assert abs(massf - 1.0) < 1e-13, "a=b=0 must conserve mass exactly"

    rg, massf, _, _ = run(a=0.0, b=0.0, c_cp=0.0)
    out["rg_null"] = rg
    say("    a=b=0 c_cp=0.0  Rg %+7.3f%% and mass %.6f  [both unchanged to 5 dp]"
        % (rg, massf))
    assert abs(rg) < 1e-5 and abs(massf - 1.0) < 1e-5, "the null corner moved something"

    say("reference numbers reproduced")
    return out


# ============================== reconstruction ==============================
# --- geometry: ray_walk and measurement_rays now live in senDOE.helpers.rays ------------------

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
            "degrade_2d_shrinkage_decay_v2_proto1.check_invariants), or set c_cp = 0 to switch transport off." % p.c_cp)

    m = pyo.ConcreteModel(name="degrade_2d_shrinkage_decay_v2_proto1")
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
    reading :func:`degrade_2d_shrinkage_decay_v2_proto1.accumulate_dose` takes, so the two differ only in the deposit index.

    This exists because agreeing with :mod:`degrade_2d_shrinkage_decay_v2_proto1` to 1e-16 proves only that two
    implementations share a convention -- including a wrong one.  The manuscript records exactly
    that failure mode for its own ordering test ("easy to write so that it passes vacuously").
    None of the three structural invariants in ``degrade_2d_shrinkage_decay_v2_proto1.check_invariants`` can catch a
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
    """Compare ``degrade_2d_shrinkage_decay_v2_proto1.accumulate_dose`` against :func:`reference_local_intensity`.

    Reports forward-ordered and antiparallel rays separately, because that is where they differ.
    Both families are asserted.  They did not always agree: antiparallel rays were out by a
    full ``I0`` until the deposit index was fixed, because each pixel was shielded by its own
    chord.  This is the check that found it.

    **What it does NOT certify, and this is the honest limit of it.**  The reference calls
    :func:`senDOE.helpers.rays.ray_geometry` for its crossing list, so it is independent of
    ``accumulate_dose`` only in the *walk* -- the travel order and the deposit index.  It shares
    the underlying chord/pixel attribution and is therefore blind to any error in it.  There is
    one: see :func:`check_chord_attribution`.  The lesson in the module docstring applies to this
    function too, one level down, which is worth saying plainly rather than leaving for someone
    to discover: a reference is only independent along the axes on which it does not reuse the
    thing it checks.
    """

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
            print("      is shielded by its own chord. senDOE.helpers.dose.degradation_dose_response has")
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

    ``strict`` is off: the fix belongs in :func:`senDOE.helpers.rays.ray_geometry` (the vendored file
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
    """Everything the Pyomo model has a variable for, computed by :mod:`degrade_2d_shrinkage_decay_v2_proto1`.

    Returns a dict of arrays keyed like the model's variables.  Used to pin the model for the
    residual check, to initialise it for a solve, and as the answer the forward solve is
    compared against.

    With ``solver`` left ``None`` this runs :func:`degrade_2d_shrinkage_decay_v2_proto1.simulate` unmodified -- which is
    the point, since the whole value of the cross-check is that the two implementations are
    independent.  Passing an :class:`ElasticSolver` instead steps the model by hand with *that*
    stiffness: needed only to initialise an inverse solve, where ``K`` is assembled from the
    reference density and not from the current guess, so a trajectory built the other way would
    start infeasible in the elasticity rows.
    """
    _omega = omega
    _step = step

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
            from senDOE.helpers.rays import ray_line_integral as _rli
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


# --- solving: solve_with_fallback and its helpers now live in senDOE.helpers.solvers ---------

# IPOPT stops on the *scaled* dual error, but a square forward run has no objective at all, so
# what decides its accuracy is constr_viol_tol -- which defaults to 1e-4.  Left alone it returns
# "optimal" with the fields still 1e-5 out, which reads as a model disagreement and is not one.
_FEASIBILITY_OPTIONS = {
    "constr_viol_tol": 1e-14,
    "acceptable_constr_viol_tol": 1e-14,
    "acceptable_tol": 1e-14,
    "bound_relax_factor": 0.0,
}


# --- the gate: does the Pyomo model reproduce the numpy forward model? -----------------------

def check_forward(image_res: int = 24, n_steps: int = 3, verbose: bool = True,
                  c_cp: float = 0.3, eps_rel: float = 1e-3, clamp_bottom: bool = False,
                  solve: bool = True, linear_solver: str = "ma27"):
    """Compare the Pyomo transcription against :func:`degrade_2d_shrinkage_decay_v2_proto1.simulate`, two ways.

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

    say("v2 Pyomo model vs degrade_2d_shrinkage_decay_v2_proto1.simulate   [%dx%d, %d steps, c_cp=%.2f, eps_rel=%g%s]"
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

    Data comes from :func:`degrade_2d_shrinkage_decay_v2_proto1.simulate`, so the measurements and the model that fits
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
            "The Pyomo model no longer reproduces degrade_2d_shrinkage_decay_v2_proto1.simulate on this geometry "
            "(residual %.3e at %s). Reconstructing against it would not mean anything; "
            "run degrade_2d_shrinkage_decay_v2_proto1.check_forward() to localise the disagreement." % (fwd_resid, where))

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
    """``python3 -m archives.degrade_2d_shrinkage_decay_v2_proto1`` checks the model; ``--reconstruct`` runs one.

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
        r = check_forward(image_res=a.image_res, n_steps=a.n_steps, c_cp=a.c_cp,
                          eps_rel=a.eps_rel, verbose=not a.quiet)
        return 0 if r["residual"] < 1e-10 else 1

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




# scaling_report now lives in senDOE.helpers.nlp_scaling.


if __name__ == "__main__":
    # `python3 -m archives.<this module> model` runs the forward-model checks; anything
    # else is the reconstruction CLI.
    if sys.argv[1:2] == ["model"]:
        check_invariants()
        print()
        check_reference_numbers()
    else:
        sys.exit(_cli())
