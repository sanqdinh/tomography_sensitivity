"""Parameterized re-implementation of ``Example2_simple_uq.py``.

This module exposes :func:`run_simple_uq`, a single function that reproduces the
tomographic uncertainty-quantification pipeline from the research example
(``sDOE_senNLP/examples/Example2/Example2_simple_uq.py``) but as a callable that:

* takes its knobs from a :class:`UQParams` dataclass instead of module-level constants,
* never calls ``plt.show()`` — it returns ``matplotlib`` ``Figure`` objects,
* renders headless (Agg backend), and
* can stream the IPOPT solver log to a caller-supplied ``log_callback``.

The actual physics/solver building blocks come from the vendored ``senDOE`` package
(see ``SENDOE_VENDOR.md``). This module does NOT import or modify the original example.

The projection geometry is user-defined: each :class:`BeamStep` is one projection (its own
angle, radial offset, and number of beams), replacing the old evenly-spaced ``linspace``
angles. The default ``beam_steps`` reproduce Example2 exactly (9 angles over [0, 180), a
full ``image_res``-wide fan per angle).

Pipeline (reconstruction/UQ identical in substance to Example2):
    1. forward IPOPT solve simulates measurements (sinogram) from a Shepp-Logan phantom,
    2. inverse IPOPT solve reconstructs ``image[:, :, 0]`` (RMSE + total-variation),
    3. k_aug extracts d(image0)/d(sinogram); posterior covariance = J·(σ²·I)·Jᵀ,
       from which the log10 covariance map and the D-optimality scalar are produced,
    4. a beam/measurement view shows the rays of the chosen geometry over the phantom.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless; belt-and-suspenders with MPLBACKEND=Agg in the container
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import pyomo.environ as pyo  # noqa: E402
from skimage.transform import resize  # noqa: E402
from skimage.data import shepp_logan_phantom  # noqa: E402

from senDOE.models.tomography_pyomo_pixel_intersection import (  # noqa: E402
    create_sample_model,
    load_image_to_sample,
    update_sinogram_rmse_expression,
    update_image_TV_expression,
    add_beam_constraints_pyomo,
    hamming_window,
    update_image_weigth,
)
from senDOE.helpers.statistics import d_optimality  # noqa: E402
from senDOE.sensitivity.pyomo_sensitivity import extract_sensitivity_matrix  # noqa: E402


# Linear solvers available in the IDAES IPOPT extensions, in preference order. ``ma27`` is
# the default; the rest are fallbacks tried only if the preferred one fails to *run*.
# NOTE: ``ma86`` is intentionally absent — it is NOT included in the IDAES IPOPT build used
# for deployment (only ma27/ma57/mumps are), even though a locally-built IPOPT may have it.
_FALLBACK_LINEAR_SOLVERS = ["ma27", "ma57", "mumps"]


@dataclass
class BeamStep:
    """One projection step of the user-defined geometry.

    Each step contributes ``n_beams`` parallel rays at ``angle_deg``, spaced one image-unit
    apart and centered at ``offset`` (radial position of the bundle). ``n_beams == 0`` is a
    sentinel meaning "use the full ``image_res``-wide fan" (resolved in :func:`run_simple_uq`).
    """

    angle_deg: float
    offset: float = 0.0  # radial center of the ray bundle, image units
    n_beams: int = 0  # 0 => default to image_res at build time


def _default_beam_steps() -> List[BeamStep]:
    """Default geometry: 9 evenly-spaced angles over [0, 180), full fan each.

    Reproduces Example2's old ``n_horizon=10`` (n_angle=9) evenly-spaced projections exactly.
    """
    return [BeamStep(float(a), 0.0, 0) for a in np.linspace(0.0, 180.0, 9, endpoint=False)]


@dataclass
class UQParams:
    """Tunable inputs for :func:`run_simple_uq` (defaults reproduce Example2 exactly)."""

    image_res: int = 30
    I0: float = 0.0  # initial beam intensity (0 => no dose degradation)
    alpha: float = 0.3  # alpha_Dose_Response (linear degradation coefficient)
    beta: float = 0.01  # beta_Rose_Response (quadratic degradation coefficient)
    tv_weight: float = 0.1  # total-variation regularization weight (1e-1 in Example2)
    noise_cov_scale: float = 10.0  # measurement noise covariance = scale * I
    ipopt_max_iter: int = 1000
    linear_solver: str = "ma27"
    # User-defined projection geometry; one BeamStep per time step (n_horizon = len + 1).
    beam_steps: List[BeamStep] = field(default_factory=_default_beam_steps)


@dataclass
class UQResults:
    """Everything :func:`run_simple_uq` produces: four figures + scalar metrics."""

    fig_phantom: "plt.Figure"
    fig_nlp: "plt.Figure"
    fig_covariance: "plt.Figure"
    fig_beams: "plt.Figure"
    d_optimality: float
    forward_solver_status: str
    inverse_solver_status: str
    forward_linear_solver: str
    inverse_linear_solver: str
    n_free_image0: int
    n_sinogram_measurements: int  # real measurement params fed to k_aug (one per ray)
    n_user_rays: int  # rays actually placed by the user geometry (after clamping)


class _LogWriter:
    """File-like sink that accumulates text and forwards each chunk to a callback."""

    def __init__(self, callback: Optional[Callable[[str], None]] = None):
        self._cb = callback
        self._chunks: list[str] = []
        self.muted = False  # when True, accumulate but do not forward to the callback

    def write(self, s: str) -> int:
        if s:
            self._chunks.append(s)
            if self._cb is not None and not self.muted:
                try:
                    self._cb(s)
                except Exception:
                    # A failing UI callback must never abort the solve.
                    pass
        return len(s) if s else 0

    def flush(self) -> None:  # pragma: no cover - required for file-like protocol
        pass

    def getvalue(self) -> str:
        return "".join(self._chunks)


def _resolve_ipopt() -> str:
    """Locate the IPOPT executable: env override -> common path -> PATH -> bare name."""
    exe = os.environ.get("IPOPT_EXECUTABLE")
    if exe and os.path.exists(exe):
        return exe
    if os.path.exists("/usr/local/bin/ipopt"):
        return "/usr/local/bin/ipopt"
    which = shutil.which("ipopt")
    if which:
        return which
    return "ipopt"  # let Pyomo resolve it from PATH


def _make_solver(params: UQParams) -> pyo.SolverFactory:
    solver = pyo.SolverFactory("ipopt", executable=_resolve_ipopt())
    solver.options["max_iter"] = int(params.ipopt_max_iter)
    # NOTE: Example2 also set print_info_string="yes", but that option makes the IDAES
    # IPOPT 3.13.2 build exit abnormally; it is purely cosmetic, so it is omitted here.
    return solver


def _solve_capturing(solver, model, writer: _LogWriter):
    """Run ``solver.solve(model, tee=True)`` capturing subprocess stdout to ``writer``.

    Uses Pyomo's ``capture_output`` (which redirects the fd that the IPOPT subprocess
    writes to). Falls back to a non-streaming solve if that helper is unavailable or its
    signature differs across Pyomo versions.
    """
    try:
        from pyomo.common.tee import capture_output
    except Exception:
        capture_output = None

    if capture_output is not None:
        try:
            cm = capture_output(writer)
        except TypeError:
            cm = None
        if cm is not None:
            with cm:
                return solver.solve(model, tee=True)

    # Fallback: solve without live streaming (still fully functional).
    return solver.solve(model, tee=False)


def _solve_with_fallback(solver, model, writer: _LogWriter, preferred: str):
    """Solve, trying linear solvers in order until one *runs* without raising.

    Matches Example2's behavior of proceeding with whatever the solve returns: we only
    switch linear solvers when the current one fails to execute (e.g. not available in the
    IPOPT build), not merely because the termination condition is non-optimal.
    """
    order = [preferred] + [s for s in _FALLBACK_LINEAR_SOLVERS if s != preferred]
    last_err = None
    for ls in order:
        solver.options["linear_solver"] = ls
        try:
            results = _solve_capturing(solver, model, writer)
        except Exception as exc:  # solver crashed / linear solver missing
            last_err = f"{ls}: {exc}"
            writer.write(f"\n[linear_solver '{ls}' failed to run: {exc}; trying next]\n")
            continue
        tc = str(results.solver.termination_condition)
        if ls != preferred:
            writer.write(f"\n[using fallback linear_solver '{ls}']\n")
        return results, ls, tc
    raise RuntimeError(f"All linear solvers failed to run. Last error: {last_err}")


def run_simple_uq(
    params: UQParams,
    log_callback: Optional[Callable[[str], None]] = None,
) -> UQResults:
    """Run the full forward/inverse/UQ tomography pipeline and return figures + metrics.

    Parameters
    ----------
    params : UQParams
        Run configuration (defaults reproduce Example2 exactly).
    log_callback : callable, optional
        Called with each chunk of IPOPT solver output as it streams. Use it to render a
        live log in a UI. Exceptions raised by the callback are swallowed.

    Returns
    -------
    UQResults
    """
    writer = _LogWriter(log_callback)
    # Mute everything up to the inverse solve (geometry setup + forward solve), so the live log
    # shows only the inverse solve onward. Unmuted just before the inverse-solve marker below.
    writer.muted = True

    image_res = int(params.image_res)
    steps = params.beam_steps
    if len(steps) < 1:
        raise ValueError("Need at least one beam step (the geometry is empty).")
    # Each step is one time index; the model needs an extra trailing time step (the final
    # dynamic-constraint target), so n_horizon = (#steps) + 1.  This reproduces the old
    # n_angle = n_horizon - 1 mapping.
    n_horizon = len(steps) + 1

    # --- geometry & phantom (Example2 lines 44-75) ------------------------------------
    phantom = shepp_logan_phantom()
    phantom = resize(phantom, (image_res, image_res))

    sample = create_sample_model(n_horizon=n_horizon, image_res=image_res)
    load_image_to_sample(sample, phantom)

    # Build each user-defined projection step.  Rays are spaced one image-unit apart,
    # centered at the step's offset; offset=0 / n_beams=image_res reproduces the old
    # full fan (np.linspace(-N/2+0.5, N/2-0.5, N)) exactly.  Rays whose distance from
    # center reaches the image boundary are dropped to avoid the vendored geometry's
    # empty-intersection IndexError (senDOE/helpers/geometry.py).
    _R_MAX = image_res / 2 - 0.5 + 1e-9
    measurement_set = []
    for i, step in enumerate(steps):
        n_beams = step.n_beams if step.n_beams and step.n_beams > 0 else image_res
        # Snap rays onto pixel centers (half-integers) so a beam passes through the middle of the
        # pixel it degrades rather than along an edge; geometry maps x -> col = floor(x + N/2), so
        # this centers the beam without changing which pixel is hit and keeps this solve in sync
        # with the live image (see app.py:_bundle_r_values). Even fans / the default full fan are
        # already half-integer and stay byte-identical.
        r_all = np.floor(step.offset + (np.arange(n_beams) - (n_beams - 1) / 2.0)) + 0.5
        r_vals = [float(r) for r in r_all if abs(r) <= _R_MAX]
        dropped = n_beams - len(r_vals)
        if dropped:
            writer.write(f"\n[step {i}: dropped {dropped} ray(s) outside the image]\n")
        if not r_vals:
            raise ValueError(
                f"Beam step {i} (angle={step.angle_deg}, offset={step.offset}, "
                f"n_beams={n_beams}) has no rays inside the image; reduce |offset| "
                f"or add beams."
            )
        measurement_set_k = [
            {"r": r, "theta": float(step.angle_deg * np.pi / 180), "time": i}
            for r in r_vals
        ]
        measurement_set = measurement_set + measurement_set_k
        sample = add_beam_constraints_pyomo(
            sample,
            measurement_set_k,
            injection_time=i,
            I0=params.I0,
            alpha_Dose_Response=params.alpha,
            beta_Rose_Response=params.beta,
            image_res=image_res,
        )

    # --- forward solve: simulate measurements (Example2 lines 83-87) ------------------
    # Writer is muted (set at creation), so this forward solve still runs and is captured but is
    # not forwarded to the live log.
    solver = _make_solver(params)
    writer.write("\n===== FORWARD SOLVE (simulate measurements) =====\n")
    _, fwd_ls, fwd_tc = _solve_with_fallback(solver, sample, writer, params.linear_solver)

    # --- harvest "measured" sinogram values (Example2 lines 116-120) -----------------
    sinogram_data = []
    for measurement in measurement_set:
        mid = [measurement["r"], measurement["theta"], measurement["time"]]
        sinogram_data.append(sample.sinogram[mid].value)

    # --- inverse solve: reconstruct image[:, :, 0] (Example2 lines 123-137) ----------
    sample = update_sinogram_rmse_expression(sample, measurement_set, sinogram_data)
    sample = update_image_TV_expression(sample, 0)

    sample.image[:, :, 0].free()
    for time in sample.time:
        sample.image[:, :, time].set_value(0.01)

    # Hamming weighting computed exactly as Example2 (its weighted term stays commented
    # out of the objective; preserved for behavioral fidelity).
    W_image = 1 - hamming_window(image_res, two_d=True)
    sample = update_image_weigth(model=sample, weight=W_image, time=0)
    sample.obj = pyo.Objective(
        expr=sample.rmse_sinogram_expression + params.tv_weight * sample.tv_expression
    )
    writer.muted = False  # stream the inverse solve onward to the live log
    writer.write("\n===== INVERSE SOLVE (reconstruct image) =====\n")
    _, inv_ls, inv_tc = _solve_with_fallback(solver, sample, writer, params.linear_solver)

    # --- collect reconstruction + free image0 vars (Example2 lines 139-144) ----------
    image_reconstruct = np.zeros((image_res, image_res))
    image0_vars = []
    for i in range(image_res):
        for j in range(image_res):
            image_reconstruct[i, j] = sample.image[i, j, 0].value
            image0_vars.append(sample.image[i, j, 0])

    fig_phantom, ax_phantom = plt.subplots(figsize=(10, 10))
    ax_phantom.imshow(phantom, cmap="gray")
    ax_phantom.set_title("Original Image")

    fig_nlp, ax_nlp = plt.subplots(figsize=(10, 10))
    ax_nlp.imshow(image_reconstruct, cmap="gray")
    ax_nlp.set_title("Reconstruction (NLP)")

    # --- beam / measurement view: the chosen geometry over the phantom ----------------
    # One line per ray (x*cos(theta) + y*sin(theta) = r), colored by projection step.
    half = image_res / 2.0
    fig_beams, ax_beams = plt.subplots(figsize=(10, 10))
    ax_beams.imshow(
        phantom, cmap="gray", extent=(-half, half, -half, half), origin="upper"
    )
    cmap = plt.get_cmap("viridis", max(len(steps), 1))
    seen_steps = set()
    for measurement in measurement_set:
        r = measurement["r"]
        theta = measurement["theta"]
        i_step = measurement["time"]
        ct, st = np.cos(theta), np.sin(theta)
        px, py = r * ct, r * st  # point on the line nearest the origin
        dx, dy = -st, ct  # direction along the line
        x0, y0 = px - image_res * dx, py - image_res * dy
        x1, y1 = px + image_res * dx, py + image_res * dy
        label = None
        if i_step not in seen_steps:
            label = f"step {i_step}: {steps[i_step].angle_deg:.0f}°"
            seen_steps.add(i_step)
        ax_beams.plot(
            [x0, x1], [y0, y1], lw=0.6, alpha=0.7, color=cmap(i_step), label=label
        )
    ax_beams.set_xlim(-half, half)
    ax_beams.set_ylim(-half, half)
    ax_beams.set_aspect("equal")
    ax_beams.set_title("Beam / measurement view (%d rays)" % len(measurement_set))
    if len(steps) <= 12:
        ax_beams.legend(loc="upper right", fontsize="small", framealpha=0.7)

    # --- sensitivity-based UQ via k_aug (Example2 lines 171-200) ----------------------
    writer.write("\n===== SENSITIVITY EXTRACTION =====\n")
    # k_aug computes one sensitivity column per parameter. Only the real measurement rays
    # appear in the objective; every other sinogram_data entry is a degenerate free variable
    # (a structurally-zero Jacobian column) that only inflates k_aug's backsolve + the
    # vendored parser's Python negation loop. Fix those entries (k_aug's own loop would have
    # anyway) so the factorized KKT system is identical to before, but hand k_aug only the
    # measured params. The dropped columns are all-zero, so Sigma = sigma^2 * J*J^T (and thus
    # the covariance map and D-optimality) is unchanged to floating point.
    measured_ids = {
        (m["r"], m["theta"], m["time"]) for m in measurement_set
    }
    sinogram_vars = []
    for idx, var in sample.sinogram_data.items():
        if idx in measured_ids:
            sinogram_vars.append(var)
        elif not var.fixed:
            var.fix()
    def _uq_failure(detail: str) -> RuntimeError:
        # Build a clear, actionable error for a failed sensitivity/covariance step and close the
        # figures already built above so a repeated (failing) Reconstruct does not leak them.
        for _f in (fig_phantom, fig_nlp, fig_beams):
            plt.close(_f)
        n_px = image_res * image_res
        n_rays = len(measurement_set)
        return RuntimeError(
            "Sensitivity / UQ step failed: " + detail + " The reduced KKT system is singular or "
            "ill-conditioned at the inverse solution, so the posterior covariance is undefined "
            f"[inverse solve status: {inv_tc}; {n_rays} measurement rays for {n_px} image pixels; "
            f"TV weight = {params.tv_weight:g}"
            + ("; I0 = 0 (no dose degradation)" if params.I0 == 0 else "")
            + "]. If the inverse solve above did not reach 'optimal', that non-convergence is the "
            "likely cause. Otherwise the problem is under-determined / under-regularized — raise "
            "the TV regularization weight, add more (and more evenly-spaced) projection angles or "
            "beams, and/or set I0 > 0; each improves conditioning so the sensitivity can be "
            "extracted."
        )

    try:
        dimage0_dsinogram = extract_sensitivity_matrix(
            model=sample,
            var_list=image0_vars,
            param_list=sinogram_vars,
            mode="k_aug",
            return_type="dense",
        )
    except (TypeError, ValueError, KeyError, IndexError) as exc:
        # k_aug produces no usable dsdp output when it cannot factor the KKT system at the solution
        # (singular / rank-deficient reduced Hessian). The vendored parser then fails cryptically:
        # np.fromstring(None, ...) -> TypeError "a bytes-like object is required, not 'NoneType'"
        # (pyomo_sensitivity.py:115), or a KeyError/ValueError/IndexError if it wrote an
        # inconsistent matrix. This one narrow call only fails this way, so translate any of these
        # into actionable guidance (the original is chained via ``from exc`` for debugging).
        raise _uq_failure("k_aug could not compute the sensitivity matrix.") from exc

    # A near-singular factorization can still emit a file full of inf/nan, which would otherwise
    # silently poison the covariance and D-optimality below — treat it like a failed extraction.
    if not np.all(np.isfinite(dimage0_dsinogram)):
        raise _uq_failure("k_aug returned a non-finite sensitivity matrix.")

    # noise covariance is sigma^2 * I, so Sigma = sigma^2 * (J*J^T) -- avoids building an
    # N x N identity and a redundant matmul against it.
    covariance_image0 = params.noise_cov_scale * (
        dimage0_dsinogram @ dimage0_dsinogram.T
    )
    covariance_image0_diag = np.diag(covariance_image0)
    covariance_image0_diag_2D = covariance_image0_diag.reshape((image_res, image_res))

    d_optimality_value = d_optimality(covariance_image0)

    fig_covariance, ax_cov = plt.subplots(figsize=(10, 10))
    with np.errstate(divide="ignore", invalid="ignore"):
        # Pixels that no ray constrains have exactly zero variance -> log10 = -inf (matplotlib
        # masks them); suppress the divide-by-zero warning so it does not spam the solver log.
        log_cov_diag_2D = np.log10(covariance_image0_diag_2D)
    im_cov = ax_cov.imshow(log_cov_diag_2D, cmap="viridis")
    ax_cov.set_title(
        "Covariance of Initial Image Variables (d=%.2f)" % d_optimality_value
    )
    cbar = fig_covariance.colorbar(im_cov, ax=ax_cov, fraction=0.046, pad=0.04)
    cbar.set_label("Covariance Value (Log10 Scale)")

    writer.write(
        "\n===== DONE (d-optimality = %.4f) =====\n" % float(d_optimality_value)
    )

    return UQResults(
        fig_phantom=fig_phantom,
        fig_nlp=fig_nlp,
        fig_covariance=fig_covariance,
        fig_beams=fig_beams,
        d_optimality=float(d_optimality_value),
        forward_solver_status=fwd_tc,
        inverse_solver_status=inv_tc,
        forward_linear_solver=fwd_ls,
        inverse_linear_solver=inv_ls,
        n_free_image0=len(image0_vars),
        n_sinogram_measurements=len(sinogram_vars),
        n_user_rays=len(measurement_set),
    )


if __name__ == "__main__":
    # Headless smoke test: run with defaults and print the scalar outcome.
    def _print(chunk: str) -> None:
        print(chunk, end="")

    res = run_simple_uq(UQParams(), log_callback=_print)
    print(
        "\nd_optimality=%.4f forward=%s(%s) inverse=%s(%s) "
        "free_image0=%d measurements=%d"
        % (
            res.d_optimality,
            res.forward_solver_status,
            res.forward_linear_solver,
            res.inverse_solver_status,
            res.inverse_linear_solver,
            res.n_free_image0,
            res.n_sinogram_measurements,
        )
    )
