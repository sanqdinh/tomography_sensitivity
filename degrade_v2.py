"""v2 damage model: dose accumulation, saturating response, decay, and elastic transport.

Implements section 3.2 ("The system") of ``xray_degradation.tex`` at commit ``a2c5ee2`` of the
manuscript repo. Where v1 (:func:`dose_response.degradation_dose_response`) is a pure *local
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
    ``record_observations=True`` to :func:`simulate`.  This is what :mod:`degrade_v2_uq`
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
  ``C_v^loc``, which is exactly what :func:`dose_response.ray_geometry` returns -- so this module
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
  handles it structurally by dropping the transport block there; see :mod:`degrade_v2_uq`.

Tuning note
-----------
``c_cp`` serves two behaviours that want opposite values: shrinkage wants it large, a beam channel
that does not refill wants it small.  Measured upstream, against a channel floor of 0.20: a
constant ``c_cp`` gives 0.57 and a modulus that follows density gives 0.56, but a **dose-dependent**
``c_cp = c0*omega(Q)`` gives 0.35.  ``c_cp`` enters the load only, so nothing cancels it -- unlike
``E``, which cancels out of ``K dx = B dw`` almost exactly (1 against 1000 moves the displacement
by 9e-15).  The dose-dependent form is not part of section 3.2 and is not implemented here.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from dose_response import bundle_r_values, ray_geometry, ray_line_integral


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


def accumulate_dose(f, r_values, angle_rad: float, I0: float, c_q: float):
    """Steps 1 and 2: returns ``(dQ, I_sum)`` for one bundle.

    ``dQ`` is ``c_q * I_p * delta_p`` summed over the bundle's rays -- the dose increment of
    eq:xd_dose_state. ``I_sum`` is ``I_p`` itself, likewise summed, which eq:xd_decay needs
    separately: the decay is driven by the *instantaneous local fluence*, not by the dose, and
    the two differ by the chord length and ``c_q``.

    ``I_p`` is the Beer-Lambert intensity delivered to pixel ``p`` by one ray, with the sum in
    the exponent running strictly over the *upstream* pixels, so the entry pixel sees the full
    ``I0``.  This is the same travel-order walk :func:`dose_response.degradation_dose_response`
    performs -- the quantity v1 calls ``local`` *is* ``I_p`` -- but here it drives a dose
    accumulator rather than a multiplicative decay.

    Rays of one measurement are simultaneous, so each integrates ``f`` as it stood at the start
    of the step and their dose contributions add (the same superposition the 3D tab uses).
    Returns the dose increment, shaped like ``f``.
    """
    f = np.asarray(f, dtype=float)
    dQ = np.zeros_like(f)
    I_sum = np.zeros_like(f)
    for r in r_values:
        g = ray_geometry(float(r), float(angle_rad), f.shape[0], f.shape[1])
        if g is None:
            continue  # ray never enters the grid
        rows, cols, seg_lengths, forward = g
        n = len(rows)
        n_seg = len(seg_lengths)
        # radon[i] = chord_i * f at crossing i, exactly as the vendored routine builds it.
        radon = seg_lengths * f[rows[:n_seg], cols[:n_seg]]
        indices = range(n) if forward else range(n - 1, -1, -1)
        shielding = 0.0
        for i in indices:
            local = I0 * np.exp(-shielding)          # I_p, before this pixel attenuates anything
            seg = i if forward else i - 1            # chord traversed on leaving crossing i
            if 0 <= seg < n_seg:
                # Deposit into the pixel that OWNS this chord, which is rows[seg] -- not rows[i].
                # eq:xd_dose_state is Q_{k+1,p} = Q_{k,p} + c_q I_p delta_p: the deposit pixel and
                # the chord's owner are the same symbol p, in one line. They coincide on forward
                # rays (seg == i) and differ by one when travel runs against the vendored
                # ascending-(x, y) crossing order, where depositing into rows[i] made every pixel
                # shield ITSELF -- the chord added below belonged to the pixel written next.
                # Forward rays are bit-identical to before; only antiparallel rays move.
                dst = i if forward else seg
                dQ[rows[dst], cols[dst]] += c_q * local * seg_lengths[seg]
                I_sum[rows[dst], cols[dst]] += local
                shielding += radon[seg]
            # the last pixel in travel order has no chord in the vendored convention -> no dose
    return dQ, I_sum


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
        # Kept so :mod:`degrade_v2_uq` can write eq:xd_elastic_discrete as Pyomo rows off the
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

        Split out of :meth:`solve` so :mod:`degrade_v2_uq` can pin its Pyomo ``u`` variables to
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
    Nothing in the app needs it; :mod:`degrade_v2_uq` does, to check its Pyomo transcription of
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


def peak_optical_depth(theta, image_res: int, n_angles: int = 12) -> float:
    """Largest line integral ``sum_p f_p*delta_p`` over a full fan at ``n_angles`` angles.

    The sample's opacity, and the one number that decides whether the model is in a sensible
    regime at all.  ``f`` is an attenuation coefficient, i.e. a reciprocal *length*, so its
    numerical size is meaningless without the pixel pitch: this app measures geometry in pixels
    (``x_range = [-w/2, w/2]`` over ``w`` pixels, so a chord through one pixel is ~1), where the
    manuscript's reference script used a ``[-1, 1]`` box on a 64 grid (chord ~0.03).  A phantom
    with O(1) values therefore has a peak optical depth near 34 here against ~1.1 there, and
    ``exp(-34)`` means the beam is entirely absorbed within the first couple of pixels: all dose
    lands in a two-pixel entry rim, nothing downstream is ever measured, and every diagnostic the
    manuscript quotes is off by an order of magnitude.  Real tomography sits at ``mu*L`` of order
    one, which is what :func:`scale_to_optical_depth` restores.
    """
    theta = np.asarray(theta, dtype=float)
    best = 0.0
    for a in range(n_angles):
        angle = np.pi * a / n_angles
        for r in bundle_r_values(0.0, 0, int(image_res)):
            best = max(best, ray_line_integral(theta, r, angle))
    return float(best)


def scale_to_optical_depth(theta, target: float, image_res: int, n_angles: int = 12):
    """Rescale ``theta`` so its peak optical depth is ``target``; see :func:`peak_optical_depth`.

    Self-normalising, so the same ``target`` means the same physics at any grid resolution.
    """
    depth = peak_optical_depth(theta, image_res, n_angles)
    if depth <= 0.0:
        return np.asarray(theta, dtype=float).copy()
    return np.asarray(theta, dtype=float) * (float(target) / depth)


def radius_of_gyration(f) -> float:
    """``sqrt(sum f r^2 / sum f)`` about the field's own centroid, in pixels.

    The compactness diagnostic: it is what "the sample shrinks" means quantitatively.  At
    ``c_cp = 0`` nothing moves, so it must come out *exactly* unchanged; above zero it must fall.
    """
    f = np.asarray(f, dtype=float)
    nr, nc = f.shape
    yy, xx = np.mgrid[0:nr, 0:nc]
    m = f.sum()
    if m <= 0:
        return float("nan")
    xc, yc = (xx * f).sum() / m, (yy * f).sum() / m
    return float(np.sqrt((f * ((xx - xc) ** 2 + (yy - yc) ** 2)).sum() / m))


# --- invariants -------------------------------------------------------------------------
# Three properties the model must have.  There is no test suite in this repo (see CLAUDE.md),
# so this is the verification path, in the same "run the module" style as tomography_3d.py.

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


if __name__ == "__main__":
    check_invariants()
    print()
    check_reference_numbers()
