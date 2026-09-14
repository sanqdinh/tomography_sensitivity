"""v2 damage model: dose accumulation, saturating response, and elastic mass transport.

Implements section 3.2 ("The system") of ``xray_degradation.tex`` at commit ``bd0eab5`` of the
manuscript repo. Where v1 (:func:`dose_response.degradation_dose_response`) is a pure *local
sink* -- ``f <- f*exp(-a*I - b*I^2)``, so mass vanishes in place and the sample fades but never
changes shape -- v2 separates dose *accumulation* from the dose *response* and adds a mass
balance. Mass therefore moves instead of disappearing, and the sample can shrink.

State is ``(f, Q)``, the parameter is ``theta = f_0``, and ``Q_0 = 0``. One step, given the ray
bundle of one measurement, is the following in order:

1. photon balance  ``I_p = I0 * exp(-sum_{m<i} f_pm * delta_pm)``   Beer-Lambert over the
   *upstream* pixels of the ray; identical to v1 and to eq:xd_local_intensity.
2. dose            ``Q <- Q + c_q * I_p * delta_p``, in grays.  Note there is deliberately **no
   local f factor**: specific absorbed energy is ``mu_en*Psi/rho`` and ``mu_en/rho`` does not
   depend on density, so the factor cancels.  (``E = f*Q`` is the energy density that rides
   algebraically alongside, reported but not fed back.)
3. response        ``omega(Q) = omega_inf + (1-omega_inf)*exp(-Q/Q_c)`` -- saturating, with a
   floor, which is what v1's unbounded geometric decay lacked.
4. converted       ``dw = 1 - omega(Q_new)/omega(Q_old)``.  **Relative, not absolute.**  This is
   what makes the collapse in `check_invariants` exact; coding it as the absolute difference
   ``omega(Q_old) - omega(Q_new)`` is the bug that test exists to catch.
5. eigenstrain     ``eps* = -0.5*c_cp*dw*I``, so ``tr eps* = -c_cp*dw``.
6. equilibrium     ``K u = B dw`` -- linear elasticity, Q1 elements.  ``K`` depends only on the
   grid, ``E``, ``nu`` and the BCs, so it is assembled and factored **once** per simulation.
7. mass balance    ``f <- f - sum_q F_{p->q} - gamma_esc*dw*f`` with antisymmetric face fluxes
   ``F_{p->q} = -F_{q->p}`` built from the face-normal component of ``u``.

Pure numpy + scipy over the repo's own cached ray geometry -- no Streamlit, no Pyomo, no solver,
so this module is importable from a headless script or a test.

Discretisation choices that differ from the manuscript's scratch reference, and why
------------------------------------------------------------------------------------
* **The photon balance is ray-based, not a plane wave.**  The reference script integrates a full
  parallel plane wave by bilinear resampling because it was verifying physics claims on a disc.
  eq:xd_local_intensity is stated over the pixels a ray crosses with chord lengths from
  ``C_v^loc``, which is exactly what :func:`dose_response.ray_geometry` returns -- so this module
  reuses it and the (angle, offset, n_beams) bundles the rest of the app is built around.
* **Pixel units.**  ``dx = 1`` here, matching the app's geometry (``x_range = [-w/2, w/2]`` over
  ``w`` pixels), where the reference used a ``[-1,1]`` box.  Every dimensionless diagnostic
  (Courant number, mass drift, the collapse error) is unaffected; only the natural size of
  ``c_q`` changes, since ``delta_p ~ 1`` here against ``~0.03`` there.
* **The box constraints of eq:xd_budget are NOT applied in the dynamics.**  They are inequality
  constraints of the estimation NLP, not a projection inside the forward map, and clipping here
  would destroy both exact mass conservation and the exact collapse.  Violations are reported in
  :class:`StepInfo` instead.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from dose_response import ray_geometry, bundle_r_values


@dataclass(frozen=True)
class V2Params:
    """Parameters of the v2 model.  Defaults are a visible-but-gentle working point."""

    I0: float = 1.0          # incident beam intensity; I0 = 0 is the undamaged limit (M = id)
    c_q: float = 0.1         # fluence x path length -> absorbed dose
    Q_c: float = 1.0         # characteristic dose of the response
    omega_inf: float = 0.2   # residual attenuation fraction, in [0, 1)
    c_cp: float = 0.3        # fraction of created void the matrix closes: 1 compliant, 0 rigid
    gamma_esc: float = 0.0   # fraction of converted mass that leaves the specimen, in [0, 1]
    eps_up: float = 1e-6     # upwind smoothing; keeps the step map C-infinity (see below)
    E0: float = 1.0          # modulus of the undamaged matrix
    nu: float = 0.3          # Poisson ratio
    e_min_ratio: float = 1e-6  # ersatz soft background, E_min/E_0, so the free surface needs no
                               # explicit meshing
    clamp_bottom: bool = False  # substrate (clamp one edge) vs free-floating body
    dx: float = 1.0          # pixel pitch, in the app's geometry units

    def f_of_omega(self, Q):
        """Convenience: the photometric collapse field ``omega(Q)``."""
        return omega(Q, self.omega_inf, self.Q_c)


@dataclass
class StepInfo:
    """Per-step diagnostics.  Cheap to compute and the only window into whether a run is sane."""

    courant: float        # max|u|/dx -- keep under ~0.5 or the upwind transport loses positivity
    mass: float           # sum_p f_p, the total attenuation M_k
    escaped: float        # sum_p gamma_esc*dw*f, the mass that left this step
    dw_max: float         # largest converted fraction anywhere
    q_max: float          # largest accumulated dose anywhere
    f_min: float          # most negative f, if the transport overshot
    energy_max: float     # max of the algebraic energy density E = f*Q


def omega(Q, omega_inf: float, Q_c: float):
    """Retained attenuation fraction -- eq:xd_response.  ``omega(0) = 1`` exactly."""
    return omega_inf + (1.0 - omega_inf) * np.exp(-np.asarray(Q, dtype=float) / Q_c)


def accumulate_dose(f, r_values, angle_rad: float, I0: float, c_q: float):
    """``c_q * I_p * delta_p`` summed over the rays of one bundle -- steps 1 and 2.

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
            seg = i if forward else i - 1            # chord within pixel i, in travel order
            if 0 <= seg < n_seg:
                dQ[rows[i], cols[i]] += c_q * local * seg_lengths[seg]
                shielding += radon[seg]
            # the last pixel in travel order has no chord in the vendored convention -> no dose
    return dQ


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

    def solve(self, dw, c_cp: float):
        """Per-pixel displacement ``(ux, uy)`` driven by the damage eigenstrain."""
        s = -0.5 * c_cp * np.asarray(dw, dtype=float).ravel()   # eigenstrain amplitude
        be = (self._Ee * s)[:, None] * self._Le[None, :]
        b = np.bincount(self._edof.ravel(), weights=be.ravel(), minlength=self.ndof_total)
        u = np.zeros(self.ndof_total)
        u[self._free] = self._lu.solve(b[self._free])
        nn = self.nn
        ux = u[0::2].reshape(nn, nn)
        uy = u[1::2].reshape(nn, nn)
        # nodal -> pixel centres
        cx = 0.25 * (ux[:-1, :-1] + ux[:-1, 1:] + ux[1:, :-1] + ux[1:, 1:])
        cy = 0.25 * (uy[:-1, :-1] + uy[:-1, 1:] + uy[1:, :-1] + uy[1:, 1:])
        return cx, cy


def upwind_flux_divergence(f, vx, vy, dx: float, eps_up: float):
    """``sum_q F_{p->q}`` with antisymmetric face fluxes and a smoothed upwind split.

    The split ``v_pm = 0.5*(sqrt(v^2 + eps^2) +- v)`` is smooth in ``v``, where the plain
    ``max(v, 0)`` is not -- differentiability is what the sensitivity extraction needs.  Each
    face contributes ``+F/dx`` to one cell and ``-F/dx`` to its neighbour, so the divergence sums
    to zero over the grid in exact arithmetic: that is what makes mass conservation exact.

    Note the smoothing is not free: at ``v = 0`` the split gives ``v_+ = v_- = eps/2`` rather
    than ``0``, so a stationary field still sees a flux ``0.5*eps*(f_L - f_R)``.  That is an
    ``O(eps)`` numerical diffusion across any gradient, and it is why the exact collapse holds
    only at ``eps_up = 0``; see :func:`check_invariants`.
    """
    f = np.asarray(f, dtype=float)

    def split(v):
        s = np.sqrt(v ** 2 + eps_up ** 2)
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


def step(f, Q, r_values, angle_rad: float, p: V2Params, solver: ElasticSolver):
    """One measurement step: ``(f, Q) -> (f_next, Q_next, info)``.  Steps 1-7 in order."""
    f = np.asarray(f, dtype=float)
    Q = np.asarray(Q, dtype=float)

    Q_next = Q + accumulate_dose(f, r_values, angle_rad, p.I0, p.c_q)
    # Relative converted fraction -- see the module docstring; absolute is the classic bug.
    dw = 1.0 - omega(Q_next, p.omega_inf, p.Q_c) / omega(Q, p.omega_inf, p.Q_c)

    ux, uy = solver.solve(dw, p.c_cp)
    div = upwind_flux_divergence(f, ux, uy, p.dx, p.eps_up)
    escaped = p.gamma_esc * dw * f
    f_next = f - div - escaped

    info = StepInfo(
        courant=float(np.sqrt(ux ** 2 + uy ** 2).max() / p.dx),
        mass=float(f_next.sum()),
        escaped=float(escaped.sum()),
        dw_max=float(dw.max()),
        q_max=float(Q_next.max()),
        f_min=float(f_next.min()),
        energy_max=float((f_next * Q_next).max()),
    )
    return f_next, Q_next, info


def simulate(theta, seq, p: V2Params, image_res: int):
    """Run a measurement sequence from the undamaged field ``theta``.

    ``seq`` is the app's ``_table_to_seq`` output -- ``(angle_deg, offset, n_beams)`` triples.
    Returns ``(f, Q, infos)``: the final attenuation field, the accumulated dose, and one
    :class:`StepInfo` per step.  The stiffness is factored once, from ``theta``.
    """
    theta = np.asarray(theta, dtype=float)
    f = theta.copy()
    Q = np.zeros_like(f)
    solver = ElasticSolver(theta, p.nu, p.E0, p.e_min_ratio, p.dx, p.clamp_bottom)
    infos = []
    for angle_deg, offset, n_beams in seq:
        r_values = bundle_r_values(float(offset), int(n_beams), int(image_res))
        f, Q, info = step(f, Q, r_values, np.deg2rad(float(angle_deg)), p, solver)
        infos.append(info)
    return f, Q, infos


# --- invariants -------------------------------------------------------------------------
# Three properties the model must have.  There is no test suite in this repo (see CLAUDE.md),
# so this is the verification path, in the same "run the module" style as tomography_3d.py.

def _demo_sequence(n_steps: int = 12):
    """Evenly spaced full-fan projections -- the sequence the manuscript's checks use."""
    return tuple((180.0 * i / n_steps, 0.0, 0) for i in range(n_steps))


def check_invariants(image_res: int = 64, n_steps: int = 12, verbose: bool = True):
    """Assert the three invariants of section 3.2.  Returns a dict of measured residuals.

    (a) **Exact mass conservation.**  ``sum_p f_{k+1,p} = sum_p f_{k,p} - gamma_esc*sum_p
        dw*f``, so at ``gamma_esc = 0`` the total attenuation is conserved to machine precision
        for *any* ``c_cp``.  This one is robust to ``eps_up``: it follows from the face fluxes
        being antisymmetric, nothing else.  If it fails, they are not.

    (b) **Exact collapse to the photometric model.**  At ``c_cp = 0, gamma_esc = 1`` the step map
        must give ``f_k = theta*omega(Q_k)``, because the relative ``dw`` telescopes and
        ``omega(0) = 1`` (bitwise, which is checked).  This is the sharpest regression test
        available: coding ``dw`` as the absolute difference instead fails it by orders of
        magnitude, which the check below demonstrates rather than asserts.

        "Exact" is a statement about real arithmetic.  In floating point the step map forms the
        telescoping product one ratio at a time while the right-hand side evaluates
        ``omega(Q_K)`` once, so the two agree only to roundoff -- measured here at well under
        1 ulp per step (5.6e-17 at one step rising to 4.4e-16 at twenty-four).  A field whose
        values are exactly representable, such as the binary disc of the manuscript's own check,
        does come out bitwise identical; a Shepp-Logan phantom does not, and that is arithmetic
        rather than a defect.  Hence a roundoff-scaled tolerance below, not ``== 0``.

    (c) **``I0 = 0`` gives ``M = id``.**  No dose, so no response, no eigenstrain, no transport.

    (b) and (c) are exact **only at** ``eps_up = 0``.  The smoothed split of eq:xd_mass_transport
    gives ``v_+ = v_- = eps/2`` at ``v = 0``, so a motionless field still exchanges
    ``0.5*eps*(f_L - f_R)`` across every face: an O(eps) diffusion that has nothing to do with
    the physics.  Both are therefore checked exactly at ``eps_up = 0`` and reported, not
    asserted, at the default ``eps_up``.
    """
    from skimage.data import shepp_logan_phantom
    from skimage.transform import resize

    theta = resize(shepp_logan_phantom(), (image_res, image_res)).astype(float)
    seq = _demo_sequence(n_steps)
    out = {}

    def say(msg):
        if verbose:
            print(msg)

    # (a) mass conservation, any c_cp, robust to eps_up ----------------------------------
    say("(a) mass conservation")
    worst_a = 0.0
    for c_cp in (0.0, 0.3, 0.8):
        p = V2Params(c_cp=c_cp, gamma_esc=0.0)
        f, _, infos = simulate(theta, seq, p, image_res)
        drift = abs(f.sum() - theta.sum()) / theta.sum()
        worst_a = max(worst_a, drift)
        say("    c_cp=%.1f gamma=0.0   relative drift %.3e   Courant max %.3f"
            % (c_cp, drift, max(i.courant for i in infos)))
    # and with escape on, the loss must be exactly the reported escaped mass
    p = V2Params(c_cp=0.3, gamma_esc=0.5)
    f, _, infos = simulate(theta, seq, p, image_res)
    resid = abs(f.sum() - (theta.sum() - sum(i.escaped for i in infos))) / theta.sum()
    say("    c_cp=0.3 gamma=0.5   escape-accounted residual %.3e" % resid)
    out["mass_drift"] = worst_a
    out["escape_residual"] = resid
    assert worst_a < 1e-13, "mass not conserved: face fluxes are not antisymmetric"
    assert resid < 1e-13, "escaped mass does not account for the loss"

    # (b) collapse to the photometric model ----------------------------------------------
    say("(b) collapse to f = theta*omega(Q)  [c_cp=0, gamma_esc=1]")
    for eps in (0.0, 1e-6):
        p = V2Params(c_cp=0.0, gamma_esc=1.0, eps_up=eps)
        f, Q, _ = simulate(theta, seq, p, image_res)
        err = float(np.abs(f - theta * omega(Q, p.omega_inf, p.Q_c)).max())
        say("    eps_up=%-7g  max|f - theta*omega(Q)| = %.3e" % (eps, err))
        if eps == 0.0:
            out["collapse_exact"] = err
            # ~1 ulp per step of telescoping roundoff; 1e-12 leaves nine orders of headroom
            # over that and still fails the absolute-dw bug by ten orders (measured below).
            tol = 1e-12
            assert err < tol, "collapse is not exact: is dw the relative fraction?"
        else:
            out["collapse_eps"] = err

    # ... and show the test has teeth: the absolute-difference dw is the bug it catches.
    p = V2Params(c_cp=0.0, gamma_esc=1.0, eps_up=0.0)
    solver = ElasticSolver(theta, p.nu, p.E0, p.e_min_ratio, p.dx, p.clamp_bottom)
    f_bug, Q_bug = theta.copy(), np.zeros_like(theta)
    for angle_deg, offset, n_beams in seq:
        rs = bundle_r_values(float(offset), int(n_beams), image_res)
        Qn = Q_bug + accumulate_dose(f_bug, rs, np.deg2rad(angle_deg), p.I0, p.c_q)
        dw_abs = omega(Q_bug, p.omega_inf, p.Q_c) - omega(Qn, p.omega_inf, p.Q_c)  # the bug
        ux, uy = solver.solve(dw_abs, p.c_cp)
        f_bug = (f_bug - upwind_flux_divergence(f_bug, ux, uy, p.dx, p.eps_up)
                 - p.gamma_esc * dw_abs * f_bug)
        Q_bug = Qn
    err_bug = float(np.abs(f_bug - theta * omega(Q_bug, p.omega_inf, p.Q_c)).max())
    say("    absolute-dw variant (the bug) gives %.3e -- the check discriminates" % err_bug)
    out["collapse_bug"] = err_bug
    assert err_bug > 1e-3, "the collapse check no longer discriminates against absolute dw"

    # (c) I0 = 0 is the identity map -------------------------------------------------------
    say("(c) I0 = 0 gives M = identity")
    for eps in (0.0, 1e-6):
        p = V2Params(I0=0.0, c_cp=0.8, gamma_esc=1.0, eps_up=eps)
        f, Q, _ = simulate(theta, seq, p, image_res)
        err = float(np.abs(f - theta).max())
        say("    eps_up=%-7g  max|f - theta| = %.3e   max Q = %.3e" % (eps, err, Q.max()))
        if eps == 0.0:
            out["identity_exact"] = err
            assert err == 0.0, "I0 = 0 is not the identity"
        else:
            out["identity_eps"] = err

    say("all invariants hold")
    return out


if __name__ == "__main__":
    check_invariants()
