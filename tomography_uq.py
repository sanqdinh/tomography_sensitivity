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

Pipeline (identical in substance to Example2):
    1. forward IPOPT solve simulates measurements (sinogram) from a Shepp-Logan phantom,
    2. inverse IPOPT solve reconstructs ``image[:, :, 0]`` (RMSE + total-variation),
    3. FBP (``iradon``) and SART reconstructions for comparison,
    4. k_aug extracts d(image0)/d(sinogram); posterior covariance = J·(σ²·I)·Jᵀ,
       from which the log10 covariance map and the D-optimality scalar are produced.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Callable, Optional

import matplotlib

matplotlib.use("Agg")  # headless; belt-and-suspenders with MPLBACKEND=Agg in the container
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import pyomo.environ as pyo  # noqa: E402
from skimage.transform import iradon, iradon_sart, resize  # noqa: E402
from skimage.data import shepp_logan_phantom  # noqa: E402

from senDOE.models.tomography_pyomo_pixel_intersection import (  # noqa: E402
    create_sample_model,
    load_image_to_sample,
    update_sinogram_rmse_expression,
    update_image_TV_expression,
    add_beam_constraints_pyomo,
    extract_sinogram_value,
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
class UQParams:
    """Tunable inputs for :func:`run_simple_uq` (defaults reproduce Example2 exactly)."""

    image_res: int = 30
    n_horizon: int = 10  # number of time steps; n_angle = n_horizon - 1
    I0: float = 0.0  # initial beam intensity (0 => no dose degradation)
    alpha: float = 0.3  # alpha_Dose_Response (linear degradation coefficient)
    beta: float = 0.01  # beta_Rose_Response (quadratic degradation coefficient)
    tv_weight: float = 0.1  # total-variation regularization weight (1e-1 in Example2)
    noise_cov_scale: float = 10.0  # measurement noise covariance = scale * I
    ipopt_max_iter: int = 1000
    linear_solver: str = "ma27"
    angle_start: float = 0.0  # degrees
    angle_stop: float = 180.0  # degrees (angles spaced over [start, stop), endpoint excluded)


@dataclass
class UQResults:
    """Everything :func:`run_simple_uq` produces: six figures + scalar metrics."""

    fig_sinogram: "plt.Figure"
    fig_phantom: "plt.Figure"
    fig_nlp: "plt.Figure"
    fig_fbp: "plt.Figure"
    fig_sart: "plt.Figure"
    fig_covariance: "plt.Figure"
    d_optimality: float
    forward_solver_status: str
    inverse_solver_status: str
    forward_linear_solver: str
    inverse_linear_solver: str
    n_free_image0: int
    n_sinogram_measurements: int


class _LogWriter:
    """File-like sink that accumulates text and forwards each chunk to a callback."""

    def __init__(self, callback: Optional[Callable[[str], None]] = None):
        self._cb = callback
        self._chunks: list[str] = []

    def write(self, s: str) -> int:
        if s:
            self._chunks.append(s)
            if self._cb is not None:
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

    image_res = int(params.image_res)
    n_horizon = int(params.n_horizon)
    n_angle = n_horizon - 1
    if n_angle < 1:
        raise ValueError("n_horizon must be >= 2 (need at least one projection angle).")

    # --- geometry & phantom (Example2 lines 44-75) ------------------------------------
    phantom = shepp_logan_phantom()
    phantom = resize(phantom, (image_res, image_res))

    sample = create_sample_model(n_horizon=n_horizon, image_res=image_res)
    load_image_to_sample(sample, phantom)

    r_interval_set = np.linspace(-image_res / 2 + 0.5, image_res / 2 - 0.5, image_res)
    angle_set = np.linspace(params.angle_start, params.angle_stop, n_angle, endpoint=False)

    measurement_set = []
    for time in range(n_angle):
        degree = angle_set[time]
        measurement_set_k = [
            {"r": float(r), "theta": float(degree * np.pi / 180), "time": time}
            for r in r_interval_set
        ]
        measurement_set = measurement_set + measurement_set_k
        sample = add_beam_constraints_pyomo(
            sample,
            measurement_set_k,
            injection_time=time,
            I0=params.I0,
            alpha_Dose_Response=params.alpha,
            beta_Rose_Response=params.beta,
            image_res=image_res,
        )

    # --- forward solve: simulate measurements (Example2 lines 83-87) ------------------
    solver = _make_solver(params)
    writer.write("\n===== FORWARD SOLVE (simulate measurements) =====\n")
    _, fwd_ls, fwd_tc = _solve_with_fallback(solver, sample, writer, params.linear_solver)

    # --- merged sinogram (Example2 lines 99-114) -------------------------------------
    sinogram_merged = extract_sinogram_value(sample, time=0)
    dx, dy = 0.5 * 180.0 / max(phantom.shape), 0.5 / phantom.shape[0]
    for i in range(1, n_angle):
        sinogram_merged = sinogram_merged + extract_sinogram_value(sample, time=i)

    fig_sinogram, ax_sino = plt.subplots(figsize=(10, 10))
    ax_sino.imshow(
        sinogram_merged,
        cmap="gray",
        extent=(-dx, 180.0 + dx, -dy, sinogram_merged.shape[0] + dy),
        aspect="auto",
        interpolation="none",
    )
    ax_sino.set_title("Sinogram (Merged over time)")

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

    # --- classical reconstructions for comparison (Example2 lines 158-168) -----------
    reconstruction_iradon = iradon(sinogram_merged, theta=angle_set)
    reconstruction_iradon_sart = iradon_sart(sinogram_merged, theta=angle_set)

    fig_fbp, ax_fbp = plt.subplots(figsize=(10, 10))
    ax_fbp.imshow(reconstruction_iradon, cmap="gray")
    ax_fbp.set_title("Reconstruction (FBP)")

    fig_sart, ax_sart = plt.subplots(figsize=(10, 10))
    ax_sart.imshow(reconstruction_iradon_sart, cmap="gray")
    ax_sart.set_title("Reconstruction (SART)")

    # --- sensitivity-based UQ via k_aug (Example2 lines 171-200) ----------------------
    writer.write("\n===== k_aug SENSITIVITY EXTRACTION =====\n")
    sinogram_id_vars = list(sample.sinogram_data.items())
    sinogram_vars = [var for _, var in sinogram_id_vars]
    dimage0_dsinogram = extract_sensitivity_matrix(
        model=sample,
        var_list=image0_vars,
        param_list=sinogram_vars,
        mode="k_aug",
        return_type="dense",
    )

    measurement_noise_covariance = params.noise_cov_scale * np.eye(len(sinogram_vars))
    covariance_image0 = (
        dimage0_dsinogram @ measurement_noise_covariance @ dimage0_dsinogram.T
    )
    covariance_image0_diag = np.diag(covariance_image0)
    covariance_image0_diag_2D = covariance_image0_diag.reshape((image_res, image_res))

    d_optimality_value = d_optimality(covariance_image0)

    fig_covariance, ax_cov = plt.subplots(figsize=(10, 10))
    im_cov = ax_cov.imshow(np.log10(covariance_image0_diag_2D), cmap="viridis")
    ax_cov.set_title(
        "Covariance of Initial Image Variables (d=%.2f)" % d_optimality_value
    )
    cbar = fig_covariance.colorbar(im_cov, ax=ax_cov, fraction=0.046, pad=0.04)
    cbar.set_label("Covariance Value (Log10 Scale)")

    writer.write(
        "\n===== DONE (d-optimality = %.4f) =====\n" % float(d_optimality_value)
    )

    return UQResults(
        fig_sinogram=fig_sinogram,
        fig_phantom=fig_phantom,
        fig_nlp=fig_nlp,
        fig_fbp=fig_fbp,
        fig_sart=fig_sart,
        fig_covariance=fig_covariance,
        d_optimality=float(d_optimality_value),
        forward_solver_status=fwd_tc,
        inverse_solver_status=inv_tc,
        forward_linear_solver=fwd_ls,
        inverse_linear_solver=inv_ls,
        n_free_image0=len(image0_vars),
        n_sinogram_measurements=len(sinogram_vars),
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
