"""Streamlit UI for the tomographic uncertainty-quantification pipeline.

The page has two tabs. The **2D** tab holds two modes that share it:

1. **Live dose-response simulator** (main area) — an interactive playground laid
   out as picture | dials/buttons | sequence table. The read-only table is the single measurement
   sequence; the picture is a *derived view* — the cumulative dose-response degradation
   (``pixel·exp(-α·I_local - β·I_local²)``) of the first ``view_k`` measurements. Sliders compose
   the next measurement (angle, radial offset, #beams) — previewed live as **red** dashes;
   **Take measurement** applies it and appends a table row, **Reset** clears the sequence.
   **Previous/Next Measurement** scrub ``view_k`` over ``0..N`` to replay the history: the
   picture shows that step, the viewed measurement's beams are traced in **blue**, and the table
   highlights that row. This runs entirely in the browser — no solver needed.

2. **Reconstruct** (button) — runs
   :func:`senDOE.models.tomography_pyomo_2d_pixel_intersection_uq.run_simple_uq` (a forward + inverse
   optimization plus a sensitivity extraction, minutes at the default size). It solves **exactly**
   the table's
   sequence (same conversion the live image uses) with the live I0/α/β values, then shows the
   reconstruction prominently plus the full UQ suite (posterior covariance, D-optimality, beam
   view, streaming solver log).

Streamlit reruns the whole script on every widget interaction, so the degraded image and the
solve results are stashed in ``st.session_state`` to survive reruns. The heavy solve runs ONLY
on the Reconstruct button press. This file is the only entrypoint and changes nothing in the
backend: it imports the vendored geometry primitives and the shared numpy helpers
(the dose-response degradation) for the live preview.
"""

import base64
import io
import os
import shutil
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly
import plotly.graph_objects as go

from senDOE.models.tomography_pyomo_2d_pixel_intersection_uq import (
    UQParams, BeamStep, run_simple_uq)

import matplotlib.pyplot as plt  # after the UQ module sets the Agg backend
import matplotlib.image as mpimg

# Vendored geometry primitives (importing them is not a backend change).
from senDOE.helpers.geometry import (
    get_line_abc_from_r_theta,
    line_grid_intersections,
    get_segment_polar,
)

# Dose-response physics lives in senDOE, Streamlit-free, so the 3D simulator shares it too
# (importing app.py would execute this whole page). Bound to the old private names so the rest
# of this file is unchanged.
from senDOE.helpers.rays import bundle_r_values as _bundle_r_values
from senDOE.helpers.dose import degradation_dose_response as _degradation_dose_response
from senDOE.helpers.solvers import SolverCancelled
from senDOE.models.tomography_3d import shepp_logan_3d, simulate_3d, detector_grid
from senDOE.helpers.dose import accumulate_dose as _accumulate_dose, scale_to_optical_depth
from senDOE.models.tomography_pyomo_2d_shrinkage_decay import (
    ShrinkageDecayUQParams as ShrinkageDecayUQParams,
    run_shrinkage_decay_reconstruction as run_shrinkage_decay_reconstruction,
    run_naive_shrinkage_reconstruction as run_naive_shrinkage_reconstruction)
# The shrinkage-decay tab's model is senDOE's shrinkage-decay model, imported under the tab's shrinkage-decay names.
from senDOE.models.tomography_2d_shrinkage_decay import (
    ShrinkageDecayParams as ShrinkageDecayParams, simulate as simulate_shrinkage_seq,
    simulate_simultaneous as simulate_shrinkage_sim, resolve as resolve_shrinkage,
    shape_diagnostics as shape_diagnostics_shrinkage,
    compaction_potential as _compaction_potential_shrinkage,
    select_eta as _select_eta_shrinkage)
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

st.set_page_config(page_title="Tomography Sensitivity UQ", layout="wide")

st.title("Tomographic reconstruction and uncertainty")
st.caption(
    "Use the simulator to take X-ray measurements. Each measurement is added to the sequence "
    "table on the right. Click **Reconstruct** to estimate the image from that sequence, then "
    "see the uncertainty for each pixel and a D-optimality score that summarizes how informative "
    "the measurements are."
)

# Fixed sim/reconstruction resolution (was the sidebar slider; equals the UQParams default).
IMAGE_RES = 30

# Live-simulator preview is a custom browser component (no-build static component): it holds the
# Angle/Offset/#Beams sliders + the beam overlay and redraws the red preview lines *while* dragging,
# client-side, so there is no server round-trip per drag (st.slider only reports on release). The
# component reports the values back on release; Python renders only the static background image and
# the committed (blue) bundle. See frontend/live_sim_component/index.html.
_LIVE_SIM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "live_sim_component")
_live_sim = components.declare_component("live_sim", path=_LIVE_SIM_DIR)

# The 3D Volume view is the same idea one level up: the plot itself lives in the component, so the
# beam curtains can be restyled in the browser while a slider is dragged. st.plotly_chart cannot do
# that -- it is server-rendered, so the earliest it can react is the release.
_VOLUME_SIM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "volume_sim_component")


def _ensure_plotly_asset() -> bool:
    """Put plotly.min.js next to the volume component's index.html; ``True`` if it is there.

    Copied out of the installed ``plotly`` package rather than pulled from a CDN or vendored into
    git: it is then guaranteed to be the build that produced the figure JSON we hand it, the app
    keeps working with no outbound network, and a 4.8 MB minified blob stays out of the history.
    The copy is idempotent and keyed on size, so a plotly upgrade refreshes it.

    Setting ``TOMO_VOLUME_SERVER_RENDER=1`` makes this report failure, which drops the Volume view
    back to ``st.plotly_chart`` plus ordinary sliders — the behaviour before the component existed.
    It is an escape hatch: the component is the only part of this app that cannot be exercised
    without a browser, so there is a way back that does not need a code change.
    """
    if os.environ.get("TOMO_VOLUME_SERVER_RENDER"):
        return False
    dst = os.path.join(_VOLUME_SIM_DIR, "plotly.min.js")
    src = os.path.join(os.path.dirname(plotly.__file__), "package_data", "plotly.min.js")
    try:
        if not os.path.exists(src):
            return False
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
            return True
        shutil.copyfile(src, dst)
        return True
    except OSError:
        return False


_PLOTLY_ASSET_OK = _ensure_plotly_asset()
_volume_sim = components.declare_component("volume_sim", path=_VOLUME_SIM_DIR)


def _live_background_uri(img, vmin, vmax) -> str:
    """PNG data-URI of the grayscale phantom view (no axes/lines) for the live component background.

    ``mpimg.imsave`` writes exactly the array's pixels (here image_res×image_res), so the data rect
    maps 1:1 to the component's image box and the JS overlay aligns to it. The component upscales it
    with ``image-rendering: pixelated`` to match the backend's ``interpolation="nearest"``.
    """
    buf = io.BytesIO()
    mpimg.imsave(buf, np.asarray(img), cmap="gray", vmin=vmin, vmax=vmax, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

# Solver-log box. A fixed-height st.container plus st.code -- deliberately NOT components.html.
#
# It used to be an iframe carrying a trailing <script> that force-scrolled to the bottom, because
# st.html strips scripts and so cannot auto-follow. That worked, but the box re-renders every
# 0.2 s for the whole solve, and on Streamlit versions that have deprecated components.v1.html
# in favour of st.iframe each render logged a deprecation notice -- five a second, interleaved
# line-for-line with the IPOPT output in the terminal, which is exactly the output it was burying.
# `global.suppressDeprecationWarnings` does NOT gate that particular notice (tried; it still
# printed), and filtering it out of `logging` is unreliable because Streamlit attaches its
# handlers after import and a logger-level filter does not catch records from child loggers.
#
# So: stop making the deprecated call. st.container(height=...) is a scrollable box with no
# script and no iframe, on every version from 1.31. Auto-follow is then unnecessary rather than
# unavailable -- showing the TAIL means the newest lines are already the visible ones.
_LOG_TAIL_LINES = 400
_LOG_BOX_HEIGHT = 300


def _render_log_box(slot, text: str) -> None:
    """Draw the rolling solver-log tail into ``slot`` (an ``st.empty()`` placeholder)."""
    tail = "".join(text).splitlines()[-_LOG_TAIL_LINES:]
    slot.empty()                     # drop the previous box so they do not stack
    with slot.container():
        with st.container(height=_LOG_BOX_HEIGHT):
            st.code("\n".join(tail), language=None)

def _term_echo(text: str) -> None:
    """Echo the solver log to the terminal Streamlit was started from.

    Writes to ``sys.__stdout__``, NOT ``sys.stdout``, and the distinction is the whole point:
    while the solve runs, Pyomo's ``capture_output`` has replaced ``sys.stdout`` with the very
    sink that calls this function, so a plain ``print`` would feed the log straight back into
    itself. ``sys.__stdout__`` is the process's original stdout and is untouched by that
    redirection.

    Flushed per chunk so the log appears live rather than in a lump when the solve ends, and
    wrapped because stdout can legitimately be closed or replaced (``streamlit run`` under a
    daemoniser, pytest capture, a detached process).
    """
    try:
        out = sys.__stdout__
        if out is not None and not out.closed:
            out.write(text)
            out.flush()
    except Exception:
        pass


def _current_script_ctx():
    """This session's ScriptRunContext, or None.  Must be called from the script thread.

    Deliberately NOT cached in a module global: app.py is the Streamlit entry script, so its
    module body re-executes on every rerun of every session, and a shared global would hold
    whichever session ran last. Handing that context to a worker thread serving a DIFFERENT
    session would write one user's solver log into another user's page.
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx()
    except Exception:
        return None


def _script_stop_requested(ctx) -> bool:
    """Read Streamlit's pending STOP request without consuming rerun requests.

    Streamlit 1.38 only checks this state at UI yield points. Native solver calls block the script
    thread, so the cancellable solver executor polls it directly while IPOPT is running.
    """
    requests = getattr(ctx, "script_requests", None) if ctx is not None else None
    lock = getattr(requests, "_lock", None)
    if requests is None or lock is None:
        return False
    try:
        with lock:
            state = getattr(requests, "_state", None)
            return getattr(state, "name", "") == "STOP"
    except Exception:
        return False


def _empty_beam_table() -> pd.DataFrame:
    """An empty measurement sequence (built up by taking measurements / editing the table)."""
    return pd.DataFrame(
        {
            "angle_deg": pd.Series([], dtype=float),
            "offset": pd.Series([], dtype=float),
            "n_beams": pd.Series([], dtype=int),
        }
    )


def _table_to_seq(df: pd.DataFrame) -> tuple:
    """Table → hashable ``((angle_deg, offset, n_beams), ...)`` measurement sequence.

    Drops rows with no angle and coerces exactly like the Reconstruct geometry build, so the
    derived live image and the solved geometry are guaranteed identical. ``n_beams == 0`` keeps
    the full-fan sentinel.
    """
    df = df.dropna(subset=["angle_deg"])
    seq = []
    for _, row in df.iterrows():
        nb = row["n_beams"]
        nb = 0 if pd.isna(nb) else max(int(nb), 0)
        off = 0.0 if pd.isna(row["offset"]) else float(row["offset"])
        seq.append((float(row["angle_deg"]), off, int(nb)))
    return tuple(seq)


# --- live simulator helpers (pure frontend; reuse vendored geometry) -------------------

@st.cache_data(show_spinner=False)
def _phantom(image_res: int) -> np.ndarray:
    """The same phantom the backend builds: ``resize(shepp_logan_phantom(), (N, N))``."""
    ph = shepp_logan_phantom()
    ph = resize(ph, (int(image_res), int(image_res)))
    return ph.astype(float)


def _live_figure(img, image_res, measurements_done, vmin, vmax,
                 preview=None, committed=None, beams_visible=True):
    """Large central grayscale image + colorbar + dynamic title with optional beam overlays.

    ``preview`` / ``committed`` are ``(angle_deg, offset, n_beams)`` bundles or ``None``. When
    ``beams_visible``: the committed (viewed) measurement is drawn in **blue** dashes and the
    next-measurement preview (live sliders) in **red** dashes, on top.
    """
    h, w = img.shape
    extent = [-w / 2, w / 2, -h / 2, h / 2]
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    cax = ax.imshow(
        img, cmap="gray", extent=extent, origin="upper",
        vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    fig.colorbar(cax, ax=ax, label="Intensity")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_title(f"Number of measurements: {measurements_done}")
    if beams_visible:
        seg_n = 200
        seg_range = [-(seg_n - 1) / 2.0, (seg_n - 1) / 2.0]

        def _draw(bundle, color):
            if bundle is None:
                return
            angle_deg, offset, n_beams = bundle
            for r in _bundle_r_values(offset, n_beams, image_res):
                seg = get_segment_polar(
                    r_distance=r, angle=np.deg2rad(angle_deg),
                    seg_range=seg_range, num_points=seg_n,
                )
                ax.plot(seg[:, 0], seg[:, 1], color=color, linestyle="--", linewidth=0.8)

        _draw(committed, "blue")  # measurement actually taken (the viewed row)
        _draw(preview, "red")     # next-measurement preview (sliders), drawn on top
    return fig


def _style_sequence(df: pd.DataFrame, k: int):
    """Read-only table styler that highlights the currently-viewed measurement (row k-1)."""
    sty = df.style
    if 1 <= k <= len(df):
        sty = sty.apply(
            lambda row: ["background-color: #ffe08a; color: #000" if row.name == k - 1 else ""
                         for _ in row],
            axis=1,
        )
    return sty


# Result figures: each backend figure is a square, equal-aspect image (optionally + a colorbar)
# with a title. We pin two canvases, both rendered with ``bbox_inches=None`` (not Streamlit's
# default "tight") so margins are honored verbatim:
#   • Covariance (has a colorbar): a WIDE canvas = square data box + a reserved right strip for the
#     colorbar and its word labels, with thin margins and a title row above.
#   • The three colorbar-less figures (phantom / reconstruction / beams): a NEAR-SQUARE canvas with
#     no right strip, so the image fills the width and there is no blank space on the right.
# Both canvases share the same data-box size (~8.04in), bottom margin, and title row, so the actual
# images render at the same scale; only the colorbar-less canvas is narrower. Consequence: the
# colorbar-less figures have a more-square aspect, so at equal column width they render a bit TALLER
# than the (wider) covariance figure — that is the cost of dropping the reserved strip.
_FIG_SIZE = (10.0, 8.8)                          # covariance canvas → square box + colorbar strip
_FIG_MAIN_RECT = (0.016, 0.0205, 0.804, 0.9136)  # data axes — square box, tight margins, title room above
_FIG_CBAR_RECT = (0.842, 0.0205, 0.024, 0.9136)  # colorbar slot — matches the image height
_FIG_SIZE_PLAIN = (8.36, 8.8)                    # colorbar-less canvas → square box, no right strip
_FIG_MAIN_RECT_PLAIN = (0.0191, 0.0205, 0.9617, 0.9136)  # same ~8.04in box, flush right (no blank strip)
# The covariance canvas is wider than the plain one (by its colorbar strip). Giving its grid column
# this same width ratio means both fill their columns at the SAME height and image size — the wider
# column exactly absorbs the colorbar, so no figure needs a blank reserved strip. (≈1.196.)
_CBAR_COL_RATIO = _FIG_SIZE[0] / _FIG_SIZE_PLAIN[0]


def _normalize_result_fig(fig):
    """Pin a backend result figure to a tight canvas+layout for the result grid.

    The backend figures are ``figsize=(10,10)`` with square, equal-aspect data; ``subplots`` makes
    the data axes ``fig.axes[0]`` and ``fig.colorbar`` appends the colorbar as ``fig.axes[1]``.
    Figures *with* a colorbar get the wide canvas (square data box + reserved colorbar strip);
    colorbar-less figures get the near-square canvas so the image fills the width with no blank
    strip on the right. Idempotent (adds no axes), so safe to call on every rerun. Must be rendered
    with ``bbox_inches=None`` so the chosen margins survive.
    """
    axes = fig.axes
    if not axes:
        return fig
    has_cbar = len(axes) > 1                 # fig.colorbar appended a second axes ⇒ this is covariance
    if has_cbar:
        fig.set_size_inches(*_FIG_SIZE)
        axes[0].set_position(_FIG_MAIN_RECT)
    else:
        fig.set_size_inches(*_FIG_SIZE_PLAIN)  # narrower canvas, no reserved colorbar strip
        axes[0].set_position(_FIG_MAIN_RECT_PLAIN)
    axes[0].set_xticks([])                  # drop the number labels on every result image
    axes[0].set_yticks([])                  # (the backend title is kept; canvas reserves a title row)
    for extra in axes[1:]:                  # colorbar axes (appended by fig.colorbar)
        extra.set_box_aspect(None)          # clear the colorbar's fixed length:width aspect (default 20)
        extra.set_aspect("auto")            # — otherwise it redraws at 20×width (≈4.8in) centered, far
        extra.set_position(_FIG_CBAR_RECT)  # shorter than the image; now it fills the strip's full height
    return fig


def _style_covariance_fig(fig):
    """Frontend-only restyle of the baked covariance figure: set a friendly title and replace the
    colorbar's numeric ticks with two words.

    Overrides the backend title ("Covariance of Initial Image Variables (d=…)") with a clearer one.
    Uses the ``Colorbar`` object API (``cbar.set_ticks``/``set_ticklabels``), which installs a
    fixed locator/formatter that survives the redraw ``st.pyplot`` triggers — setting ticks on the
    raw colorbar axes would be overridden on draw. Idempotent across reruns (overwrites, never
    appends).
    """
    axes = fig.axes
    if not axes:
        return fig
    axes[0].set_title("Posterior covariance: per-pixel uncertainty")
    imgs = axes[0].get_images()
    if imgs and imgs[0].colorbar is not None:   # relabel the colorbar with words, not values
        im = imgs[0]
        cbar = im.colorbar
        lo, hi = im.get_clim()
        if hi <= lo:                            # degenerate (constant covariance) guard
            hi = lo + 1e-9
        pad = 0.02 * (hi - lo)                  # inset off the extremes so labels aren't clipped
        cbar.set_ticks([lo + pad, hi - pad])
        cbar.set_ticklabels(["Low\nUncertainty", "High\nUncertainty"])  # two lines, narrower strip
        cbar.set_label("")
    return fig


def _render_results(slot, results):
    """Render the Reconstruct output (banner + two-column figure grid) into a fixed placeholder.

    Rendering through a slot lets a fresh Reconstruct clear the previous output up-front, so the
    old figures disappear while the new solve runs instead of lingering underneath.
    """
    with slot.container():
        if results is None:
            st.info("Tune the live simulator and build the sequence, then click **Reconstruct** "
                    "to run the solve.")
            return
        st.success(
            f"**D-optimality = {results.d_optimality:.4f}**  ·  "
            f"forward: {results.forward_solver_status} ({results.forward_linear_solver})  ·  "
            f"inverse: {results.inverse_solver_status} ({results.inverse_linear_solver})  ·  "
            f"{results.n_free_image0} free pixels  ·  "
            f"{results.n_user_rays} rays / {results.n_sinogram_measurements} sensitivity params"
        )
        # 2×2 grid: left = original / reconstruction, right = beam view / posterior covariance.
        # All four share the same ~8.04in data image; the covariance adds a reserved colorbar strip
        # (wider canvas), while the three colorbar-less figures use a near-square canvas with no
        # right strip. The covariance is additionally restyled (word ticks on the colorbar).
        for _f in (results.fig_phantom, results.fig_nlp, results.fig_covariance, results.fig_beams):
            _normalize_result_fig(_f)
        _style_covariance_fig(results.fig_covariance)
        # Built row by row (not as two stacked columns) so the covariance's column can be wider than
        # the plain columns by exactly its colorbar strip (_CBAR_COL_RATIO). Each figure fills its own
        # column (bbox_inches=None keeps its margins verbatim), so there is no blank reserved strip,
        # yet every figure renders at the same height and image size. The thin spacer in row 1 holds
        # the width that the covariance colorbar occupies in row 2, keeping the four images aligned.
        _spacer = _CBAR_COL_RATIO - 1.0
        row1 = st.columns([1.0, 1.0, _spacer])
        row1[0].pyplot(results.fig_phantom, use_container_width=True, bbox_inches=None)
        row1[0].caption("Original phantom")
        row1[1].pyplot(results.fig_beams, use_container_width=True, bbox_inches=None)
        row1[1].caption("Beam / measurement view: the chosen projection geometry over the phantom")
        row2 = st.columns([1.0, _CBAR_COL_RATIO])
        row2[0].pyplot(results.fig_nlp, use_container_width=True, bbox_inches=None)
        row2[0].caption("Reconstruction (NLP)")
        row2[1].pyplot(results.fig_covariance, use_container_width=True, bbox_inches=None)
        row2[1].caption("Posterior covariance: per-pixel uncertainty")


@st.cache_data(show_spinner=False)
def _degraded_image(seq: tuple, I0: float, alpha: float, beta: float,
                    image_res: int) -> np.ndarray:
    """Cumulative dose-response degradation of the phantom over the measurement sequence.

    Pure function of the table (``seq``) + global dose params, so the live image is always an
    exact view of the table. Cached: dragging the angle slider (overlay only) is a cache hit;
    only taking/editing a measurement or changing I0/α/β recomputes.
    """
    img = _phantom(image_res).copy()
    for angle_deg, offset, n_beams in seq:
        theta = np.deg2rad(angle_deg)
        for r in _bundle_r_values(offset, n_beams, image_res):
            img = _degradation_dose_response(img, r, theta, I0, alpha, beta)
    return img


# --- live simulator button callbacks (fire before the rerun body; read live_* keys) ----

def _sync_live_sim(prefix: str, widget_key: str):
    """Fold a live_sim component's released values into the canonical ``<prefix>_*`` keys.

    Runs as the component's ``on_change``, i.e. BEFORE the script body rebuilds its render args —
    which is the whole point. The component now follows the values Python sends it (so a second
    control surface can drive it); that is only safe if Python's copy is already up to date when
    it builds those args. Syncing after the call instead would hand the component back the
    *previous* value and snap its thumb.
    """
    val = st.session_state.get(widget_key)
    if not isinstance(val, dict):
        return
    st.session_state[prefix + "_angle"] = float(val["angle"])
    st.session_state[prefix + "_offset"] = float(val["offset"])
    st.session_state[prefix + "_nbeams"] = int(val["nbeams"])


# Zero-argument wrappers: a custom component's on_change is handed straight to register_widget
# with no args/kwargs (unlike st.slider's), so anything passed as args= would be swallowed by
# **kwargs and shipped to the frontend as a render arg instead of reaching the callback.
def _cb_sync_live_sim_2d():
    _sync_live_sim("live", "live_sim")


def _cb_sync_live_sim_3d():
    _sync_live_sim("live3d", "live_sim_3d")


def _cb_sync_volume_sim():
    _sync_live_sim("live3d", "volume_sim")


def _cb_step():
    """Take a measurement: append the current slider bundle as a row in the sequence table."""
    s = st.session_state
    new_row = pd.DataFrame(
        [{
            "angle_deg": float(s["live_angle"]),
            "offset": float(s["live_offset"]),
            "n_beams": int(s["live_nbeams"]),
        }]
    )
    s["beam_table"] = pd.concat([s["beam_table"], new_row], ignore_index=True)
    s["view_k"] = len(s["beam_table"])  # jump the view to the just-taken measurement


def _cb_reset():
    """Clear the sequence — back to the clean phantom."""
    st.session_state["beam_table"] = _empty_beam_table()
    st.session_state["view_k"] = 0


def _cb_view_prev(view_key: str):
    """Step a tab's view back one measurement."""
    st.session_state[view_key] = max(0, int(st.session_state[view_key]) - 1)


def _cb_view_next(view_key: str, n: int):
    """Step a tab's view forward one measurement, bounded by the sequence length."""
    st.session_state[view_key] = min(int(n), int(st.session_state[view_key]) + 1)


def _nav_block(view_key: str, n: int) -> int:
    """Previous / Next measurement scrubbing.  Returns the clamped view index.

    **Display only.**  It changes how much of the sequence is applied to the picture, not the
    sequence itself, and not what gets solved: the 2D tab's Reconstruct has always run the whole
    table however far the view is scrubbed back, and the 3D Reconstruct sub-tab does the same.

    Clamps on the way through, because the table may have shrunk since the index was set.
    """
    k = max(0, min(int(st.session_state[view_key]), n))
    st.session_state[view_key] = k
    nav = st.columns(2)
    nav[0].button("\u2b05 Previous Measurement", key="btn_prev_" + view_key,
                  on_click=_cb_view_prev, args=(view_key,),
                  use_container_width=True, disabled=(k == 0))
    nav[1].button("Next Measurement \u27a1", key="btn_next_" + view_key,
                  on_click=_cb_view_next, args=(view_key, n),
                  use_container_width=True, disabled=(k >= n))
    st.caption("Viewing measurement **%d** of **%d**." % (k, n))
    return k


def _cb_toggle():
    st.session_state["beams_visible"] = not st.session_state["beams_visible"]


# --- session state seeding -------------------------------------------------------------
if "beam_table" not in st.session_state:
    st.session_state["beam_table"] = _empty_beam_table()
st.session_state.setdefault("beams_visible", True)
st.session_state.setdefault("view_k", 0)  # number of measurements currently displayed
# Live control values live in session_state so the central figure (rendered before the
# widgets exist in script order) can read them, and callbacks have a single source of truth.
for _k, _v in {
    "live_angle": 45.0, "live_offset": 0.0, "live_nbeams": 30,
    "live_I0": 0.0, "live_alpha": 0.3, "live_beta": 0.01, "live_tv_weight": 0.1,
    "live_preset_lo": 0.0, "live_preset_hi": 180.0, "live_preset_n": 9,
}.items():
    st.session_state.setdefault(_k, _v)

# 3D tab keeps its own namespace so the two tabs never clobber each other's controls.
if "beam_table_3d" not in st.session_state:
    st.session_state["beam_table_3d"] = _empty_beam_table()
for _k, _v in {
    "live3d_angle": 45.0, "live3d_offset": 0.0, "live3d_nbeams": 30, "live3d_z": 0,
    "live3d_nslices": 16, "live3d_contrast": 1.0,
    "live3d_I0": 0.0, "live3d_alpha": 0.3, "live3d_beta": 0.01,
    "live3d_meas": 0, "live3d_opacity": 1.0, "live3d_isomin": 0.05,
    "live3d_isomax": 1.0, "live3d_band": "Whole head", "live3d_band_prev": None,
    "live3d_cutaxis": "x (col)", "live3d_cut": 0.5,
    "live3d_volsrc": "Degraded", "live3d_sinoview": "Per-slice sinogram",
    "live3d_showbeams": True,
    # Per-slice reconstruction (Reconstruct sub-tab). tv_weight matches the 2D default.
    "live3d_tv_weight": 0.1, "live3d_recon_stride": 1, "live3d_recon_z": 0,
    # Measurement preset: evenly spaced angles over [lo, hi).
    "live3d_preset_lo": 0.0, "live3d_preset_hi": 180.0, "live3d_preset_n": 9,
    "live3d_view_k": 0,
    "live3d_recsrc": "Reconstruction", "live3d_recband_pct": (2, 100),
}.items():
    st.session_state.setdefault(_k, _v)

# 2D implicit-transport shrinkage (shrinkage-decay). v5 with steps 5 and 6 changed: a softplus directed
# rate and one global sparse solve, so positivity and conservation are unconditional and the
# compaction number is gone. Forward only -- no tv_weight, no maxiter, no stored results.
if "beam_table_shrinkage" not in st.session_state:
    st.session_state["beam_table_shrinkage"] = _empty_beam_table()
for _k, _v in {
    "shrinkage_angle": 45.0, "shrinkage_offset": 0.0, "shrinkage_nbeams": 0, "shrinkage_res": 32,
    "shrinkage_showbeams": True,
    "shrinkage_depth": 1.1, "shrinkage_I0": 1.0,
    "shrinkage_c_omega": 0.1, "shrinkage_c_cp": 0.3, "shrinkage_a": 0.05, "shrinkage_b": 0.0,
    "shrinkage_reach": 7.0, "shrinkage_gamma": 100.0, "shrinkage_fref": 0.002, "shrinkage_eta": 1e-3,
    # eta is CHOSEN from the forward run by default -- no fixed value works, because max|dP|
    # spans 0.15 to 80 across this tab's own slider range. The slider is the override.
    "shrinkage_eta_auto": True,
    # Measurement schedule. SIMULTANEOUS is the default: the table's rows are one exposure
    # carrying every bundle, not one exposure each.
    "shrinkage_mode": "Simultaneous",
    # Reconstruct. tv_weight is the normalised trade-off ratio, NOT the 2D tab's scale.
    # maxiter 3000, NOT 500. It was cut to 500 on the belief that this solve never converges and
    # the cap only decided how fast you found that out. That belief predated ma97 + select_eta +
    # the sigma-bound fix: grid 32 / K=5 simultaneous now reaches `optimal` at iteration 1253
    # (219 s, inf_du 2.5e-14, theta 9.16%), so a cap of 500 does not reveal a failure, it CAUSES
    # one. IPOPT stops at convergence, so the cap costs nothing when the solve succeeds.
    "shrinkage_tv_weight": 0.01, "shrinkage_maxiter": 3000,
    "shrinkage_linear_solver": "ma97",
    "shrinkage_naive_method": "FBP",
    "shrinkage_preset_lo": 0.0, "shrinkage_preset_hi": 180.0, "shrinkage_preset_n": 10,
    "shrinkage_view_k": 0,
}.items():
    st.session_state.setdefault(_k, _v)


# --- 3D degradation tab (2.5D: same bundle through every slice; no reconstruction) ------
# Own session_state namespace (live3d_*, beam_table_3d) so the two tabs never clobber each
# other. The measurement table schema is identical to the 2D one, so _empty_beam_table /
# _table_to_seq are reused as-is.

_N_SLICES_MIN, _N_SLICES_MAX = 4, 48


@st.cache_data(show_spinner=False)
def _simulate_3d(seq: tuple, I0: float, alpha: float, beta: float,
                 image_res: int, n_slices: int, contrast: float):
    """``(original, degraded, sinogram)`` for the sequence (cached like the 2D image).

    Pure function of the table + dose params + volume size, so scrubbing z, switching sub-tab,
    rotating the volume and changing the measurement index are all cache hits; only taking a
    measurement or changing a parameter recomputes.
    """
    vol0 = shepp_logan_3d(image_res, n_slices, contrast=contrast)
    vol1, sino = simulate_3d(vol0, seq, I0, alpha, beta, image_res)
    return vol0, vol1, sino


@st.cache_data(show_spinner=False)
def _phantom_3d(image_res: int, n_slices: int, contrast: float):
    """The undamaged volume on its own, without running a degradation sequence."""
    return shepp_logan_3d(image_res, n_slices, contrast=contrast)


@st.cache_data(show_spinner=False, max_entries=512)
def _recon_slice_3d(seq: tuple, image_res: int, n_slices: int, contrast: float,
                    I0: float, alpha: float, beta: float, tv_weight: float, k: int,
                    _log_callback=None, _cancel_callback=None):
    """Reconstruct ONE z-slice exactly the way the 2D tab reconstructs its phantom.

    The slice handed over is the **undamaged** ``vol0[:, :, k]``, with ``I0/alpha/beta`` passed
    alongside, so the forward model applies the dose-response itself -- precisely what the 2D
    tab does. Reconstructing the already-degraded volume instead would be a different
    experiment: it would treat the damage as part of the object rather than as something the
    measurement caused.

    Cached per slice, which is what makes the stride control worth having: a coarse pass at
    stride 4 leaves its slices in the cache, so committing to stride 1 afterwards only pays for
    the ones that are new. ``_log_callback`` is underscore-prefixed so Streamlit leaves it out
    of the hash -- an uncached slice still streams into the log box, a cached one never runs.

    Returns plain arrays and scalars, never the ``UQResults``: it carries four matplotlib
    figures, and 48 slices of those would be ~192 live figures held in session state.
    """
    vol0 = _phantom_3d(image_res, n_slices, contrast)
    steps = [BeamStep(angle_deg=a, offset=o, n_beams=n) for (a, o, n) in seq]
    res = run_simple_uq(
        UQParams(image_res=image_res, I0=I0, alpha=alpha, beta=beta, tv_weight=tv_weight,
                 beam_steps=steps, phantom=vol0[:, :, int(k)]),
        log_callback=_log_callback,
        cancel_callback=_cancel_callback,
    )
    return (np.asarray(res.image_reconstruct, dtype=float),
            None if res.log_cov_diag_2D is None else np.asarray(res.log_cov_diag_2D, dtype=float),
            float(res.d_optimality), res.forward_solver_status, res.inverse_solver_status)


def _recon_slice_figure(original, recon, logcov, k: int, status: str):
    """Original | reconstruction | log-covariance for one slice, drawn from the stored arrays."""
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2))
    finite = np.isfinite(recon)
    vmax = float(original.max()) if original.size else 1.0
    for ax, (img, ttl, cmap) in zip(axes, (
        (original, "Original (undamaged) z=%d" % k, "gray"),
        (recon, "Reconstruction", "gray"),
        (logcov, "log10 variance", "viridis"),
    )):
        if not np.any(np.isfinite(img)):
            ax.text(0.5, 0.5, "not solved", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
        else:
            kw = dict(vmin=0.0, vmax=vmax) if cmap == "gray" else {}
            im = ax.imshow(img, cmap=cmap, interpolation="nearest", **kw)
            fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(ttl, fontsize=10)
    axes[1].set_xlabel(status, fontsize=8)
    if np.any(finite):
        err = float(np.nanmean(np.abs(recon - original)))
        axes[1].set_title("Reconstruction  (mean |err| %.4g)" % err, fontsize=10)
    fig.tight_layout()
    return fig


def _shrinkage_recon_figure(regular, naive):
    """Render independent naïve and regular results in the requested 2×3 layout."""
    truth = (naive.theta_true if naive is not None else
             regular.theta_true if regular is not None else None)
    if truth is None:
        return None
    shape = truth.shape
    nan_image = np.full(shape, np.nan, dtype=float)

    naive_truth = (getattr(naive, "theta_true", None) if naive is not None else
                   getattr(regular, "theta_true", None) if regular is not None else None)
    naive_truth = naive_truth if naive_truth is not None else nan_image
    naive_image = (getattr(naive, "theta_naive", None)
                   if naive is not None else None)
    naive_image = (naive_image if naive_image is not None
                   else nan_image)
    naive_error = (naive_image - naive_truth if np.any(np.isfinite(naive_image))
                   else nan_image)
    naive_method = getattr(naive, "method", "FBP") if naive is not None else "FBP"
    naive_titles = {
        "FBP": "Naïvely reconstructed sample (FBP)",
        "SART": "Naïvely reconstructed sample (SART, 1 pass)",
        "SART (1 pass)": "Naïvely reconstructed sample (SART, 1 pass)",
        "SART (5 passes)": "Naïvely reconstructed sample (SART, 5 passes)",
        "SART (10 passes)": "Naïvely reconstructed sample (SART, 10 passes)",
        "Pyomo": "Naïvely reconstructed sample (static Pyomo)",
    }
    naive_title = naive_titles.get(naive_method, "naive reconstruction (%s)" % naive_method)

    regular_truth = regular.theta_true if regular is not None else nan_image
    regular_image = regular.theta_hat if regular is not None else nan_image
    regular_error = (regular_image - regular_truth if regular is not None else nan_image)
    covariance = (np.where(np.isfinite(regular.log_cov_diag_2D), regular.log_cov_diag_2D, np.nan)
                  if regular is not None and regular.log_cov_diag_2D is not None else nan_image)

    def _vmax(image):
        finite = np.isfinite(image)
        return max(float(image[finite].max()), 1e-12) if np.any(finite) else 1.0

    def _span(image):
        finite = np.isfinite(image)
        return max(float(np.abs(image[finite]).max()), 1e-12) if np.any(finite) else 1.0

    error_span = max(_span(naive_error), _span(regular_error))
    panels = [
        (naive_truth, "Original sample (undamaged)", "gray",
         dict(vmin=0.0, vmax=_vmax(naive_truth))),
        (naive_image, naive_title, "gray",
         dict(vmin=0.0, vmax=_vmax(naive_truth))),
        (naive_error, "Naïve reconstruction error (reconstructed − original)", "coolwarm",
         dict(vmin=-error_span, vmax=error_span)),
        (covariance, "Reconstruction uncertainty (log10 posterior covariance diagonal)",
         "viridis", {}),
        (regular_image, "Reconstructed sample", "gray",
         dict(vmin=0.0, vmax=_vmax(regular_truth))),
        (regular_error, "Reconstruction error (reconstructed − original)", "coolwarm",
         dict(vmin=-error_span, vmax=error_span)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.8))
    for ax, (image, title, cmap, kwargs) in zip(axes.flat, panels):
        if not np.any(np.isfinite(image)):
            ax.text(0.5, 0.5, "unavailable", ha="center", va="center", transform=ax.transAxes)
        else:
            im = ax.imshow(image, cmap=cmap, interpolation="nearest", **kwargs)
            fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    return fig


def _cb3d_step():
    """Take a 3D measurement: append the current bundle to the 3D sequence table."""
    s = st.session_state
    new_row = pd.DataFrame(
        [{
            "angle_deg": float(s["live3d_angle"]),
            "offset": float(s["live3d_offset"]),
            "n_beams": int(s["live3d_nbeams"]),
        }]
    )
    s["beam_table_3d"] = pd.concat([s["beam_table_3d"], new_row], ignore_index=True)
    s["live3d_view_k"] = len(s["beam_table_3d"])  # jump the view to the just-taken measurement


def _preset_angles(lo: float, hi: float, n: int) -> list:
    """``n`` evenly spaced angles over **[lo, hi)** — the upper bound is exclusive.

    Exclusive because a projection at 180 deg traces the same line as one at 0 deg, so an
    inclusive sweep would spend a measurement re-measuring the start. This is also what makes
    the obvious case come out right: 0 to 180 in 3 gives 0, 60, 120 rather than 0, 90, 180.
    """
    n = max(int(n), 1)
    return [float(a) for a in np.linspace(float(lo), float(hi), n, endpoint=False)]


def _cb_preset(prefix: str, table_key: str, view_key: str, replace: bool):
    """Fill a sequence table with an evenly spaced angular sweep.

    Shared by all three tabs; ``prefix`` selects the namespace (``live``, ``live3d``, ``v2``).
    Offset and beam count come from that tab's current aim, so the preset sweeps *this* bundle
    through the angles rather than inventing a geometry of its own.
    """
    s = st.session_state
    rows = pd.DataFrame(
        [{"angle_deg": a,
          "offset": float(s[prefix + "_offset"]),
          "n_beams": int(s[prefix + "_nbeams"])}
         for a in _preset_angles(s[prefix + "_preset_lo"], s[prefix + "_preset_hi"],
                                 s[prefix + "_preset_n"])]
    )
    s[table_key] = rows if replace else pd.concat([s[table_key], rows], ignore_index=True)
    # Show what was just generated. Without this the view stays wherever it was -- at 0 on a
    # fresh tab -- and a freshly generated sweep would render as an untouched sample.
    s[view_key] = len(s[table_key])


def _preset_block(prefix: str, table_key: str, view_key: str, max_n: int = 60):
    """The Measurement Preset controls.  One implementation, rendered in each tab.

    Widget keys are namespaced by ``prefix`` because Streamlit renders every tab on every run,
    so a key may appear only once across all of them.
    """
    s = st.session_state
    st.markdown("**Measurement Preset**")
    pc1, pc2 = st.columns(2)
    pc1.number_input("From (deg)", step=5.0, format="%.1f", key=prefix + "_preset_lo")
    pc2.number_input("To (deg, exclusive)", step=5.0, format="%.1f", key=prefix + "_preset_hi")
    st.slider("Number of measurements", 1, max_n, step=1, key=prefix + "_preset_n")
    pre = _preset_angles(s[prefix + "_preset_lo"], s[prefix + "_preset_hi"],
                         s[prefix + "_preset_n"])
    # Show the actual angles: the exclusive upper bound is the one thing about this that can
    # surprise, and a preview settles it without anyone having to read a tooltip.
    st.caption(
        "\u2192 %s   \u00b7   at offset **%.1f**, **%s**"
        % (", ".join("%g\u00b0" % a for a in pre[:8]) + (" \u2026" if len(pre) > 8 else ""),
           float(s[prefix + "_offset"]),
           "full fan" if int(s[prefix + "_nbeams"]) == 0
           else "%d beams" % int(s[prefix + "_nbeams"]))
    )
    pb1, pb2 = st.columns(2)
    pb1.button("Generate", key="btn_preset_gen_" + prefix, on_click=_cb_preset,
               args=(prefix, table_key, view_key, True), use_container_width=True,
               help="Replace the sequence with this sweep.")
    pb2.button("Append", key="btn_preset_add_" + prefix, on_click=_cb_preset,
               args=(prefix, table_key, view_key, False), use_container_width=True,
               help="Add this sweep to the existing sequence.")


def _cb3d_reset():
    """Clear the 3D sequence — back to the clean volume."""
    st.session_state["beam_table_3d"] = _empty_beam_table()


def _cb3d_sync_z(source_key: str):
    """Fold one of the slice pickers back into the canonical ``live3d_z``.

    The Slice view and the per-slice sinogram both select a slice, but Streamlit renders every
    tab on every run, so one widget ``key`` cannot appear in both. Each view therefore owns its
    own slider and writes through to a single shared value, which is re-seeded into both before
    they are re-created. Two controls, one slice — they can never disagree.
    """
    st.session_state["live3d_z"] = int(st.session_state[source_key])


def _cb3d_sync_beam():
    """Fold the Volume sub-tab's duplicate bundle sliders back into the canonical ``live3d_*``.

    Same one-value-two-widgets pattern as :func:`_cb3d_sync_z`: Streamlit renders every tab on
    every run, so a widget ``key`` can appear only once across all of them and the Volume view
    cannot reuse the keys the Slice view's component owns. Each surface writes through to the
    canonical value, which is re-seeded into both before they are rebuilt — so they can never
    disagree, whichever one you touch.
    """
    s = st.session_state
    s["live3d_angle"] = float(s["live3d_angle_vol"])
    s["live3d_offset"] = float(s["live3d_offset_vol"])
    s["live3d_nbeams"] = int(s["live3d_nbeams_vol"])


def _cb3d_sync_recview():
    """Fold the Reconstruct view's opacity / cut-away duplicates into the canonical values.

    Same one-value-two-widgets pattern as :func:`_cb3d_sync_z`: Streamlit renders every tab on
    every run, so these keys cannot be reused across two sub-tabs. Sharing the *values* rather
    than giving the Reconstruct view its own is deliberate -- lining a cut-away up in the Volume
    view and then switching over to see the same cut through the reconstruction is the whole
    point of having both. The intensity band is NOT shared, because the two views are not in the
    same units: attenuation runs 0-1 and log10 variance is negative throughout.
    """
    s = st.session_state
    s["live3d_opacity"] = float(s["live3d_opacity_rec"])
    s["live3d_cutaxis"] = s["live3d_cutaxis_rec"]
    s["live3d_cut"] = float(s["live3d_cut_rec"])


def _bundle_sliders_3d(suffix: str, slots=None):
    """The Angle / Offset / # Beams trio, in the ranges the 3D tab uses.

    ``slots`` is three containers to place them in (e.g. ``st.columns(3)``); ``None`` stacks
    them in the current one.

    Seeds itself from the canonical values immediately before building the widgets, which is the
    last moment Streamlit allows a widget key to be written. Doing it here rather than at the top
    of the tab matters: this renders *after* the Slice view's component, so it picks up a value
    set over there within the same run, without depending on the component's on_change having
    fired first.
    """
    s = st.session_state
    s["live3d_angle" + suffix] = float(s["live3d_angle"])
    s["live3d_offset" + suffix] = float(s["live3d_offset"])
    s["live3d_nbeams" + suffix] = int(s["live3d_nbeams"])
    a, o, n = slots if slots is not None else (st, st, st)
    a.slider("Angle (deg)", 0.0, 360.0, step=1.0, key="live3d_angle" + suffix,
             on_change=_cb3d_sync_beam)
    o.slider("Offset", -float(IMAGE_RES) / 2, float(IMAGE_RES) / 2, step=0.5,
             key="live3d_offset" + suffix, on_change=_cb3d_sync_beam)
    n.slider("# Beams (0 = full fan)", 0, IMAGE_RES, step=1, key="live3d_nbeams" + suffix,
             on_change=_cb3d_sync_beam)


def _slice_figure(img, image_res, k, n_slices, n_meas, vmin, vmax, preview=None):
    """One z-slice as a grayscale image + colorbar, with the live beam bundle overlaid in red.

    Same recipe as the 2D live view (extent-mapped, ``origin="upper"``, nearest-neighbour so
    voxels stay square), so a slice here is directly comparable to the 2D tab's picture.
    """
    h, w = img.shape
    extent = [-w / 2, w / 2, -h / 2, h / 2]
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    cax = ax.imshow(img, cmap="gray", extent=extent, origin="upper",
                    vmin=vmin, vmax=vmax, interpolation="nearest")
    fig.colorbar(cax, ax=ax, label="Intensity")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_title("Slice z = %d / %d   ·   %d measurement%s applied"
                 % (k, n_slices - 1, n_meas, "" if n_meas == 1 else "s"))
    if preview is not None:
        angle_deg, offset, n_beams = preview
        seg_n = 200
        seg_range = [-(seg_n - 1) / 2.0, (seg_n - 1) / 2.0]
        for r in _bundle_r_values(offset, n_beams, image_res):
            seg = get_segment_polar(r_distance=r, angle=np.deg2rad(angle_deg),
                                    seg_range=seg_range, num_points=seg_n)
            ax.plot(seg[:, 0], seg[:, 1], color="red", linestyle="--", linewidth=0.8)
    return fig


def _sinogram_figure(panel, image_res, xlabel, xticklabels, title):
    """A sinogram panel: detector position (rows) against ``xlabel`` (columns).

    ``panel`` is a slice of the ``(detector, measurement, slice)`` array from
    :func:`senDOE.models.tomography_3d.simulate_3d`, so unmeasured cells are ``NaN``. A hand-built
    sequence samples only a few of the ``image_res`` detector slots, so most of the panel is genuinely
    unmeasured — those cells are drawn in a flat off-colour via ``set_bad`` so they read as
    "no data" rather than as a real low line-integral.

    The ramp is greyscale, matching the slice and volume views. That forces the ``set_bad``
    colour to be **chromatic**: greyscale already spans every lightness from black to white, so
    no shade of grey can mean "unmeasured" -- the old dark blue-grey would now read as a genuine
    low reading, which is the one thing this mask exists to prevent. Hue is the only channel
    left, hence the muted blue.
    """
    r_grid = detector_grid(image_res)
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("#3d5a80")  # unmeasured -- chromatic, so no grey level can be mistaken for it
    finite = np.isfinite(panel)
    im = ax.imshow(
        np.ma.masked_invalid(panel), cmap=cmap, aspect="auto", interpolation="nearest",
        origin="lower", vmin=(np.nanmin(panel) if finite.any() else 0.0),
        vmax=(np.nanmax(panel) if finite.any() else 1.0),
        extent=[-0.5, panel.shape[1] - 0.5, r_grid[0] - 0.5, r_grid[-1] + 0.5],
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Line integral  ∫ pixel dl")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Detector position r")
    ax.set_title(title)
    if xticklabels is not None and 0 < len(xticklabels) <= 24:
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels, fontsize=8, rotation=45 if len(xticklabels) > 8 else 0)
    if not finite.any():
        ax.text(0.5, 0.5, "no measurements yet", transform=ax.transAxes,
                ha="center", va="center", color="white", fontsize=13)
    return fig


# Corner offsets for each cube face, wound counter-clockwise seen from outside. Mask axes are
# (row, col, slice); vertices are emitted as (x=col, y=row, z=slice) to match _volume_figure's
# scene axes.
_VOXEL_FACES = (
    (1, True, ((0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5), (0.5, -0.5, 0.5))),
    (1, False, ((-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, 0.5), (-0.5, 0.5, -0.5))),
    (0, True, ((-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, -0.5))),
    (0, False, ((-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (-0.5, -0.5, 0.5))),
    (2, True, ((-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5))),
    (2, False, ((-0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (0.5, 0.5, -0.5), (0.5, -0.5, -0.5))),
)

_CUT_AXES = {"none": None, "x (col)": 1, "y (row)": 0, "z (slice)": 2}

# Where the flat beam preview is drawn. The volume occupies z = -0.5 … nz-0.5, so this sits
# clearly beneath it — a shadow of the bundle on the floor of the scene rather than a cage through
# the head. The z axis is given a margin below it so the line is never flush with the axis bound
# (plotly clips there), and that margin is unconditional so the box does not resize when the
# beams are toggled off.
_BEAM_FLOOR_Z = -1.0

# The flat preview's dash pattern is built as GEOMETRY, not as a line style: plotly has no
# "arrowhead dash", so each dash is drawn as a small arrow. That puts the travel direction along
# the whole ray instead of only at one end, and it is the dash pattern -- there is no separate
# glyph to keep in step. Lengths are in voxels, along the ray.
_ARROW_PERIOD = 4.0    # centre-to-centre spacing of successive arrows
_ARROW_SHAFT = 2.2     # length of one arrow's shaft (the rest of the period is the gap)
_ARROW_HEAD = 0.9      # how far the barbs sit behind the tip
# Half width must stay under half the ray spacing, which is 1 voxel: at 0.55 the heads of
# neighbouring rays overlapped and a full 30-ray fan rendered as a solid red mat.
_ARROW_HALF_W = 0.35


def _exposed_faces(mask, axis, positive):
    """Voxels in ``mask`` whose neighbour along ``axis`` is absent — i.e. that face is visible.

    Culling the faces between two drawn voxels is what keeps this affordable *and* legible: a
    solid region becomes a shell instead of a stack of coincident quads, so ~12k voxels emit
    ~5k faces rather than 76k.
    """
    nb = np.zeros_like(mask)
    dst, src = [slice(None)] * 3, [slice(None)] * 3
    if positive:
        dst[axis], src[axis] = slice(0, -1), slice(1, None)
    else:
        dst[axis], src[axis] = slice(1, None), slice(0, -1)
    nb[tuple(dst)] = mask[tuple(src)]
    return mask & ~nb


def _clip_ray_to_box(r, theta, half_w, half_h):
    """``(t0, t1)``: where the ray ``x·cosθ + y·sinθ = r`` enters and leaves the image box.

    The ray is parameterized along the beam-travel tangent,
    ``p(t) = (r·cosθ − t·sinθ, r·sinθ + t·cosθ)``, the same convention the vendored geometry and
    ``get_segment_polar`` use. Slab method: intersect the admissible ``t`` interval of the x and
    y slabs. Returns ``None`` if the ray never enters the box.

    Clipping analytically instead of drawing a long segment and letting plotly trim it keeps the
    endpoints exact and independent of how the renderer treats ``scene.*axis.range``.
    """
    c, sn = np.cos(theta), np.sin(theta)
    t0, t1 = -np.inf, np.inf
    for p0, d, lo, hi in ((r * c, -sn, -half_w, half_w), (r * sn, c, -half_h, half_h)):
        if abs(d) < 1e-12:          # parallel to this slab: in or out for every t
            if p0 < lo or p0 > hi:
                return None
            continue
        a, b = (lo - p0) / d, (hi - p0) / d
        if a > b:
            a, b = b, a
        t0, t1 = max(t0, a), min(t1, b)
    return (t0, t1) if t1 > t0 else None


def _arrow_dashes(x0, y0, x1, y1, z):
    """One ray's flat line as a row of arrows — the dash pattern, drawn as geometry.

    Arrows march from ``(x0, y0)`` to ``(x1, y1)``, which is the beam-travel direction: the
    caller derives both ends from ascending ``t``, whose tangent is ``(-sinθ, cosθ)``. Direction
    therefore comes from the two *scene* endpoints and is never recomputed from θ, so the flipped
    row axis cannot be applied twice.

    The spacing is stretched to divide the chord exactly, so the last arrow always lands on the
    exit point and the pattern does not crawl along the ray as the offset slider moves. A chord
    too short for one full arrow still gets one, scaled down.
    """
    dx, dy = x1 - x0, y1 - y0
    chord = float(np.hypot(dx, dy))
    if chord < 1e-9:
        return ()
    ux, uy = dx / chord, dy / chord
    px, py = -uy, ux                                   # in-plane perpendicular
    n = max(1, int(chord // _ARROW_PERIOD))
    step = chord / n
    scale = min(1.0, step / _ARROW_PERIOD)             # shrink to fit a short chord
    shaft, head, half_w = (_ARROW_SHAFT * scale, _ARROW_HEAD * scale, _ARROW_HALF_W * scale)
    segs = []
    for i in range(n):
        s = step * (i + 1)                             # tip of this arrow, measured from the entry
        tx, ty = x0 + s * ux, y0 + s * uy
        sx, sy = x0 + max(0.0, s - shaft) * ux, y0 + max(0.0, s - shaft) * uy
        bx, by = tx - head * ux, ty - head * uy
        segs.append(((sx, sy, z), (tx, ty, z)))                       # shaft
        segs.append(((bx + half_w * px, by + half_w * py, z),         # head: barb -> tip -> barb
                     (tx, ty, z),
                     (bx - half_w * px, by - half_w * py, z)))
    return tuple(segs)


def _beam_curtain_trace(r_values, angle_deg, nr, nc, nz, color, name=None, flat_z=None):
    """Each ray of a bundle as dashed lines through the volume.

    2.5D means one measurement fires the same bundle through *every* slice, so inside the volume
    a ray is not a line — it is a vertical plane. ``flat_z=None`` outlines that plane (bottom
    edge, top edge, two verticals), which shows the geometry honestly without a translucent sheet
    hiding the voxels behind it.

    ``flat_z`` instead draws **one ray per line at that height, dashed with arrows**. Full
    curtains read as a cage around the head and the rays nearest the camera sit in front of the
    very voxels you are trying to see; dropped to the floor below the volume they read as a
    shadow of the bundle, and the head is unobstructed.

    Each dash of that line is a small arrow rather than a plain stroke, because direction is
    physical and a plain line cannot show it: dose integrates along the travel tangent
    ``(-sinθ, cosθ)``, so 0° and 180° draw the identical line while depositing dose in opposite
    order. plotly has no arrowhead dash style, so the pattern is emitted as geometry and the line
    is drawn solid — the gaps between arrows *are* the dashes. Arrows point along increasing
    ``t``, which is that travel tangent by construction, and the last one lands on the exit point.

    Physical ``(x, y)`` become scene indices with the vendored convention from
    ``senDOE/helpers/geometry.py`` (``col = int(x + w/2)``, ``row = int(h/2 − y)``) evaluated at
    pixel centres — the same mapping ``_slice_figure``'s overlay uses, hence the −0.5 and the
    flipped row axis. Every segment goes into ONE trace separated by ``NaN`` breaks, so a 30-ray
    fan costs one trace rather than 120.

    Returns ``None`` when the bundle is empty (every ray clamped off the grid).
    """
    theta = np.deg2rad(float(angle_deg))
    c, sn = np.cos(theta), np.sin(theta)
    zb, zt = -0.5, nz - 0.5
    xs, ys, zs = [], [], []
    for r in r_values:
        clip = _clip_ray_to_box(float(r), theta, nc / 2.0, nr / 2.0)
        if clip is None:
            continue
        ends = []
        for t in clip:
            x, y = r * c - t * sn, r * sn + t * c
            ends.append((x + nc / 2.0 - 0.5, nr / 2.0 - 0.5 - y))
        (x0, y0), (x1, y1) = ends
        if flat_z is None:
            segs = (((x0, y0, zb), (x1, y1, zb)),      # bottom edge
                    ((x0, y0, zt), (x1, y1, zt)),      # top edge
                    ((x0, y0, zb), (x0, y0, zt)),      # the two verticals that close the curtain
                    ((x1, y1, zb), (x1, y1, zt)))
        else:
            segs = _arrow_dashes(x0, y0, x1, y1, float(flat_z))
        for seg in segs:
            for px, py, pz in seg:
                xs.append(px)
                ys.append(py)
                zs.append(pz)
            xs.append(np.nan)      # break the polyline between segments
            ys.append(np.nan)
            zs.append(np.nan)
    if not xs:
        # An empty trace, not None, when the caller named it: the browser restyles this trace BY
        # NAME while dragging, so it has to exist even at an offset where every ray is clamped off
        # the grid -- otherwise the curtains would vanish for good at the ends of that slider.
        if name is None:
            return None
        xs = ys = zs = []
    # dash="dot", not "dash". plotly maps line.dash through a fixed pattern table and then
    # SCALES it by the line width (plotly.min.js: DASHES.dash = [4, 1], each entry multiplied by
    # line.width * pixelRatio), so "dash" at width 2 is 8px on / 2px off -- an 80% duty cycle that
    # reads as a solid line. DASHES.dot = [1, 1] is the only even one, giving the broken look the
    # 2D overlay gets from stroke-dasharray "5 4".
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines", name=name,
        # Solid in flat mode: the arrows ARE the dashes, and plotly's pattern would chew them up.
        line=dict(color=color, width=2,
                  dash=("solid" if flat_z is not None else "dot")),
        hoverinfo="skip", showlegend=False,
    )


def _volume_figure(vol, opacity, isomin, isomax, title, colorscale, cmax,
                   cut_axis="none", cut_frac=0.0, beams=(), cmin=0.0):
    """Discrete voxel rendering: every voxel is a solid cube, no interpolation.

    ``go.Volume`` ray-marches through the data and blends between voxel centres, which smears
    exactly the small interior structures this view exists to show. Here each voxel above
    ``isomin`` becomes an actual cube (``go.Mesh3d`` + ``flatshading``), so blocks read as
    blocks and a ventricle keeps a hard edge.

    Two ways to see inside, because opaque blocks hide their own interior: drop ``opacity``, or
    ``cut_frac`` away the near part of an axis to expose a cut face. The cut is applied to the
    mask *before* face culling, so the exposed cross-section is drawn as real faces coloured by
    the voxel values there.

    ``isomin``/``isomax`` keep only voxels in that intensity band. Hiding *above* is what peels
    the uniform crust off: set it just under the crust value and the shell disappears, leaving
    the interior structures standing on their own.

    ``cmax`` is supplied by the caller and is deliberately **independent of the sliders**: the
    colour scale must not move when you hide, cut or fade voxels, or the same intensity would
    render as a different colour at every slider position and nothing could be compared. Fixing
    it costs some contrast when only the dim interior is left on screen — that is the intended
    trade: a stable, readable scale beats a pretty but meaningless one.

    ``beams`` are ready-made traces from :func:`_beam_curtain_trace`; the geometry is built
    outside so this stays a pure renderer.
    """
    nr, nc, nz = vol.shape
    mask = (vol >= isomin) & (vol <= isomax)
    ax = _CUT_AXES.get(cut_axis)
    if ax is not None and cut_frac > 0:
        keep = max(1, int(round(vol.shape[ax] * (1.0 - cut_frac))))
        sl = [slice(None)] * 3
        sl[ax] = slice(keep, None)
        mask[tuple(sl)] = False

    verts, vals = [], []
    for axis, positive, corners in _VOXEL_FACES:
        idx = np.argwhere(_exposed_faces(mask, axis, positive))
        if not len(idx):
            continue
        centres = np.stack([idx[:, 1], idx[:, 0], idx[:, 2]], axis=1).astype(float)
        verts.append(centres[:, None, :] + np.asarray(corners)[None, :, :])
        vals.append(vol[idx[:, 0], idx[:, 1], idx[:, 2]])

    fig = go.Figure()
    if verts:
        v = np.concatenate(verts).reshape(-1, 3)
        vals = np.concatenate(vals)
        base = np.arange(len(vals)) * 4
        fig.add_trace(
            go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2],
                # Each quad is two triangles: (0,1,2) and (0,2,3).
                i=np.stack([base, base], 1).ravel(),
                j=np.stack([base + 1, base + 2], 1).ravel(),
                k=np.stack([base + 2, base + 3], 1).ravel(),
                intensity=np.repeat(vals, 2), intensitymode="cell",
                colorscale=colorscale,
                # Fixed cmin -> cmax, set by the caller and never by the mask, so no slider can
                # rescale the bar. Intensity is physically non-negative, so 0 is the dark end.
                # cmin defaults to 0, which is right for attenuation (0 is air). The
                # reconstruction's log10-variance view is negative throughout, and a ramp
                # anchored at 0 would clamp every voxel to one colour, so that caller sets it.
                cmin=float(cmin),
                cmax=float(max(cmax, cmin + 1e-6)),
                opacity=float(opacity), flatshading=True,
                lighting=dict(ambient=0.62, diffuse=0.58, specular=0.12, roughness=0.7),
                lightposition=dict(x=2 * nc, y=-2 * nr, z=2 * nz),
                colorbar=dict(title="Intensity", thickness=14),
            )
        )
    else:
        title += " (no voxels in this intensity band)"

    for trace in beams:
        fig.add_trace(trace)

    fig.update_layout(
        title=title, height=620, margin=dict(l=0, r=0, t=40, b=0),
        # Pinned so the component's Plotly.react() keeps the camera across reruns; without it every
        # rerun would snap the view back to the default angle.
        uirevision="volume",
        scene=dict(
            xaxis_title="x (col)", yaxis_title="y (row)", zaxis_title="z (slice)",
            xaxis=dict(range=[-1, nc]), yaxis=dict(range=[-1, nr]),
            zaxis=dict(range=[_BEAM_FLOOR_Z - 0.25, nz]),
            # Equal x/y scaling; z is exaggerated for thin stacks so they are not a pancake.
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=max(0.35, min(1.0, nz / max(nr, 1)))),
        ),
    )
    return fig


# Named intensity windows for the Volume view. Peeling the crust with "Hide above" only exposes
# the brain, which is itself a closed mass wrapping everything -- reaching the structures needs
# "Hide below" raised past the brain too. These presets do both at once, and derive their
# thresholds from the volume's OWN distinct levels rather than hardcoded numbers, so they keep
# working as the contrast slider moves the tissue values around.
_BANDS = ("Whole head", "Crust off", "Structures only", "Brightest only", "Custom")

# "Hide below" must never reach 0. Air is exactly 0.0, so a threshold of 0 admits every empty
# voxel in the bounding box and the volume fills into a featureless block -- the opposite of
# what the control is for. 0.001 is below any real tissue value and above air.
_ISOMIN_FLOOR = 0.001


def _band_window(preset, vol, isomin, isomax):
    """``(lo, hi)`` intensity window for a named preset; falls back to the sliders on Custom."""
    if preset == "Custom":
        return float(isomin), float(isomax)
    levels = np.unique(np.round(vol[vol > 1e-9], 6))
    if levels.size == 0:
        return float(isomin), float(isomax)
    lo, hi = float(levels[0]) - 1e-6, float(levels[-1]) + 1e-6
    mid = lambda i: float(levels[i] + levels[i + 1]) / 2.0   # split between adjacent levels
    if preset != "Whole head" and levels.size >= 2:
        hi = mid(levels.size - 2)                            # drop the crust (top level)
    if preset == "Structures only" and levels.size >= 3:
        lo = mid(0)                                          # drop the brain (bottom level)
    if preset == "Brightest only" and levels.size >= 4:
        lo = mid(levels.size - 3)                            # keep only the top non-crust level
    return lo, hi


def _render_3d_tab():
    """The 3D degradation simulator: build a sequence, scrub slices, watch the dose land."""
    s = st.session_state
    st.caption(
        "**2.5D forward simulation.** The phantom is a true 3D Shepp-Logan volume, but each "
        "measurement fires the *same* beam bundle through every slice. Rays never cross "
        "slices. Dose still varies with depth because each slice attenuates the beam "
        "differently. **Reconstruct** solves each slice as an independent 2D problem and "
        "stacks the results."
    )
    left3d, mid3d, right3d = st.columns([3, 2, 2])

    # The full table is what Reconstruct solves; seq3d is the prefix the *pictures* show, so
    # every display below keeps working unchanged while Previous/Next scrubs history.
    seq_all = _table_to_seq(s["beam_table_3d"])
    n_all = len(seq_all)
    seq3d = seq_all[:max(0, min(int(s["live3d_view_k"]), n_all))]
    n_meas = len(seq3d)
    n_slices = int(s["live3d_nslices"])

    vol0, vol, sino = _simulate_3d(
        seq3d, float(s["live3d_I0"]), float(s["live3d_alpha"]), float(s["live3d_beta"]),
        IMAGE_RES, n_slices, float(s["live3d_contrast"]),
    )
    # Clamp the viewed slice / measurement: either may have shrunk since it was set.
    k = max(0, min(int(s["live3d_z"]), n_slices - 1))
    s["live3d_z"] = k
    if n_meas:
        s["live3d_meas"] = max(0, min(int(s["live3d_meas"]), n_meas - 1))
    # Seed both slice pickers from the canonical value. Safe because it happens before either
    # widget is instantiated this run; Streamlit only objects to writing a widget key after.
    # (The Volume sub-tab's duplicate bundle sliders seed themselves; see _bundle_sliders_3d.)
    s["live3d_z_slice"] = s["live3d_z_sino"] = k

    with left3d:
        _nav_block("live3d_view_k", n_all)
        view_slice, view_vol, view_sino, view_recon = st.tabs(
            ["Slice", "Volume", "Sinogram", "Reconstruct"]
        )

        with view_slice:
            # The 2D tab's live component, second instance. It OWNS the Angle/Offset/#Beams
            # sliders and redraws the red dashes client-side while the thumb is held; st.slider
            # only reports on release, so a Python-owned slider cannot track a drag at all. That
            # is why those three sliders moved out of the controls column to sit under this
            # picture -- the same arrangement the 2D tab already uses.
            _live_sim(
                image_uri=_live_background_uri(vol[:, :, k], 0.0, 1.0),
                image_res=IMAGE_RES,
                k=n_meas,
                angle=float(s["live3d_angle"]),
                offset=float(s["live3d_offset"]),
                nbeams=int(s["live3d_nbeams"]),
                # There is no view_k scrubbing here -- every measurement is always applied -- so
                # 2D's "the measurement shown" has no analogue. The last one taken is the useful
                # one to keep on screen next to the preview.
                committed=(list(seq3d[-1]) if n_meas else None),
                beams_visible=bool(s["live3d_showbeams"]),
                # This tab's own ranges, wider than 2D's: half-unit offsets and the BeamStep
                # "0 = full fan" case, both of which its st.sliders offered before the move.
                angle_range=[0, 360, 1],
                offset_range=[-float(IMAGE_RES) / 2, float(IMAGE_RES) / 2, 0.5],
                nbeams_range=[0, IMAGE_RES, 1],
                title="Slice z = %d / %d   \u00b7   %d measurement%s applied"
                      % (k, n_slices - 1, n_meas, "" if n_meas == 1 else "s"),
                hint='<b style="color:#ff2b2b">Red</b> dashes = next-measurement preview '
                     '(these sliders); <b style="color:#1f77ff">blue</b> dashes = the last '
                     'measurement taken. # Beams at 0 means the full fan.',
                legend="1.00",
                default={
                    "angle": float(s["live3d_angle"]),
                    "offset": float(s["live3d_offset"]),
                    "nbeams": int(s["live3d_nbeams"]),
                },
                key="live_sim_3d",
                on_change=_cb_sync_live_sim_3d,
            )
            # Return value deliberately unused -- see the 2D instance. Writing it back each run
            # would undo a change just made with the Volume sub-tab's sliders.
            # _cb_sync_live_sim_3d owns the sync, and runs early enough that the args above
            # already carry the result.
            # Stays server-side: a new z needs a new background PNG, so there is nothing a
            # client-side slider could redraw without a round-trip anyway.
            st.slider("Slice (z)", 0, max(n_slices - 1, 0), key="live3d_z_slice",
                      on_change=_cb3d_sync_z, args=("live3d_z_slice",))

        with view_vol:
            src = s["live3d_volsrc"]
            # Colour ceilings are pinned to the UNDEGRADED phantom, not to whatever is on
            # screen, so the bar holds still while you work the sliders. Original and Degraded
            # share one scale (same physical quantity), which also keeps them comparable as
            # dose accumulates and the degraded maximum drifts down. Dose removed is a
            # difference, bounded by the same maximum, so it uses it too.
            fixed_cmax = float(vol0.max())
            if src == "Original":
                data, cmap_name = vol0, "Gray"
                title = "Phantom (no dose applied)"
            elif src == "Reconstruction":
                # The stack built in the Reconstruct sub-tab, viewable here so it gets the
                # cut-away / opacity / intensity-band controls that already exist.
                _r3 = st.session_state.get("results_3d")
                data = vol0 * np.nan if _r3 is None else _r3["recon"]
                cmap_name = "Gray"
                title = ("Stacked reconstruction" if _r3 is not None
                         else "No reconstruction yet. See the Reconstruct sub-tab.")
            elif src == "Dose removed":
                data, cmap_name = vol0 - vol, "Inferno"
                title = "Dose removed (original − degraded): where the beams landed"
            else:
                data, cmap_name = vol, "Gray"
                title = "Degraded volume · %d measurement%s" % (
                    n_meas, "" if n_meas == 1 else "s")
            # Reset the Custom window whenever the band CHANGES into Custom, so it always opens
            # on the full 0.001-1.00 range instead of a stale window from an earlier visit.
            #
            # Keyed on the transition, not on "is a preset active": a session that was already
            # sitting in Custom when the page reloaded never leaves Custom, so a check of the
            # latter kind never fires and the stale value survives. The sentinel None start
            # makes the first run of any session count as a transition, which is what catches
            # that case -- including a hot reload, which keeps session_state across a code
            # change. Assignments land before the sliders are built, which is the only point
            # Streamlit allows a widget key to be written.
            if s["live3d_band_prev"] != s["live3d_band"]:
                if s["live3d_band"] == "Custom":
                    s["live3d_isomin"] = _ISOMIN_FLOOR
                    s["live3d_isomax"] = 1.0
                s["live3d_band_prev"] = s["live3d_band"]
            # Clamp a value carried over from before the floor existed, before the widget that
            # owns this key is created -- Streamlit rejects a value below min_value.
            s["live3d_isomin"] = max(_ISOMIN_FLOOR, float(s["live3d_isomin"]))
            band_lo, band_hi = _band_window(
                s["live3d_band"], vol0, s["live3d_isomin"], s["live3d_isomax"])
            # Beam curtains. Plotly is server-rendered, so unlike the Slice overlay these only
            # refresh when a slider is RELEASED -- never mid-drag.
            nz = vol0.shape[2]
            beams = []
            if s["live3d_showbeams"]:
                if n_meas:           # blue: the last measurement taken, as in the Slice view
                    _a, _o, _n = seq3d[-1]
                    _tr = _beam_curtain_trace(_bundle_r_values(_o, _n, IMAGE_RES), _a,
                                              IMAGE_RES, IMAGE_RES, nz, "#1f77ff",
                                              name="beam_committed")
                    if _tr is not None:
                        beams.append(_tr)
                # Red: the live preview. Python still builds it, so the plot is correct the moment
                # it loads and stays correct if the browser-side redraw never runs; the component
                # only restyles THIS trace (found by name) while a slider is dragged.
                beams.append(_beam_curtain_trace(
                    _bundle_r_values(s["live3d_offset"], s["live3d_nbeams"], IMAGE_RES),
                    s["live3d_angle"], IMAGE_RES, IMAGE_RES, nz, "#ff2b2b",
                    name="beam_preview", flat_z=_BEAM_FLOOR_Z))
            fig3d = _volume_figure(data, s["live3d_opacity"], band_lo, band_hi,
                                   title, cmap_name, fixed_cmax,
                                   s["live3d_cutaxis"], s["live3d_cut"], beams=beams)
            if _PLOTLY_ASSET_OK:
                _volume_sim(
                    figure=fig3d.to_json(),
                    image_res=IMAGE_RES, nr=IMAGE_RES, nc=IMAGE_RES, nz=nz,
                    flat_z=_BEAM_FLOOR_Z,
                    arrow=[_ARROW_PERIOD, _ARROW_SHAFT, _ARROW_HEAD, _ARROW_HALF_W],
                    angle=float(s["live3d_angle"]),
                    offset=float(s["live3d_offset"]),
                    nbeams=int(s["live3d_nbeams"]),
                    angle_range=[0, 360, 1],
                    offset_range=[-float(IMAGE_RES) / 2, float(IMAGE_RES) / 2, 0.5],
                    nbeams_range=[0, IMAGE_RES, 1],
                    default={
                        "angle": float(s["live3d_angle"]),
                        "offset": float(s["live3d_offset"]),
                        "nbeams": int(s["live3d_nbeams"]),
                    },
                    key="volume_sim",
                    on_change=_cb_sync_volume_sim,
                )
            else:
                # No plotly.min.js to hand the component: fall back to the server-rendered chart
                # with its own sliders, which works but only redraws on release.
                st.plotly_chart(fig3d, use_container_width=True)
                st.warning("Live beam preview is off: plotly.min.js was not found in the "
                           "installed plotly package, so the plot is rendered server-side.")
                st.markdown("**Aim the next measurement**")
                _bundle_sliders_3d("_vol", slots=st.columns(3))
            vc1, vc2, vc3, vc4 = st.columns(4)
            with vc1:
                st.radio("Show", ("Degraded", "Original", "Dose removed", "Reconstruction"),
                         key="live3d_volsrc",
                         help="'Dose removed' is the clearest view of where the beams "
                              "deposited energy.")
            with vc2:
                st.slider("Opacity", 0.02, 1.0, step=0.01, key="live3d_opacity",
                          help="Lower = more see-through. This is the 'look inside' knob.")
            with vc3:
                st.selectbox("Layers", _BANDS, key="live3d_band",
                             help="Which tissue to draw. 'Structures only' is the one that "
                                  "shows the floating ellipses: removing the crust alone is "
                                  "not enough, because the brain underneath is also solid. "
                                  "Thresholds come from the phantom's own levels, so these "
                                  "keep working as you move Phantom contrast.")
                if s["live3d_band"] == "Custom":
                    st.slider("Hide below", _ISOMIN_FLOOR, 1.0, step=0.001, format="%.3f",
                              key="live3d_isomin",
                              help="Drop voxels dimmer than this. Raise it past the brain "
                                   "value to free the structures inside. Floored at %.3f so it "
                                   "can never admit air (which is exactly 0)."
                                   % _ISOMIN_FLOOR)
                    st.slider("Hide above", 0.0, 1.0, step=0.01, key="live3d_isomax",
                              help="Drop voxels brighter than this, just under the crust "
                                   "value peels the outer shell.")
                else:
                    st.caption("window %.2f to %.2f" % (band_lo, band_hi))
            with vc4:
                st.selectbox("Cut away", tuple(_CUT_AXES), key="live3d_cutaxis",
                             help="Slice the volume open along an axis to expose the interior "
                                  "as a solid cut face.")
                st.slider("Cut amount", 0.0, 0.95, step=0.05, key="live3d_cut",
                          help="How much of the axis to slice away. ~0.5 puts the cut plane "
                               "through the middle of the head, which is where the ventricles "
                               "and blobs are; below ~0.35 it stays outboard of them and the "
                               "exposed face is solid brain.")
            st.caption(
                "Every voxel is a solid cube. There is no interpolation. Faces between two drawn voxels "
                "are culled, so you see surfaces rather than a fog of stacked quads. "
                "**To see the structures floating inside, use Layers, then Structures only.** "
                "Removing the crust alone will not do it: the brain underneath is a closed mass "
                "wrapping them, so it has to go too. The cut-away is the other route, but a "
                "cavity is only visible where the cut plane actually passes through it. "
                "The **red curtains** are the next measurement's bundle and the blue ones the "
                "last one taken: in 2.5D a ray fires through every slice, so each one is a "
                "vertical plane, not a line. They follow the sliders on release, not mid-drag "
                "(this view is rendered server-side). The Slice view is the live one. "
                "Drag to rotate · scroll to zoom · double-click to reset."
            )

        with view_sino:
            st.radio("View", ("Per-slice sinogram", "Per-measurement projection"),
                     key="live3d_sinoview", horizontal=True,
                     help="A sinogram is one slice's readings across every measurement. A "
                          "projection is one measurement's readings across every slice. It is "
                          "a 2D detector panel behind the volume would record.")
            if n_meas == 0:
                st.pyplot(_sinogram_figure(np.full((IMAGE_RES, 1), np.nan), IMAGE_RES,
                                           "Measurement", None,
                                           "Take a measurement to populate the sinogram"),
                          use_container_width=True)
            elif s["live3d_sinoview"] == "Per-slice sinogram":
                st.pyplot(
                    _sinogram_figure(
                        sino[:, :, k], IMAGE_RES, "Measurement (angle)",
                        ["%d\n%.0f°" % (i, a) for i, (a, _o, _n) in enumerate(seq3d)],
                        "Sinogram of slice z = %d / %d" % (k, n_slices - 1),
                    ),
                    use_container_width=True,
                )
                st.slider("Slice (z)", 0, max(n_slices - 1, 0), key="live3d_z_sino",
                          on_change=_cb3d_sync_z, args=("live3d_z_sino",),
                          help="Shared with the Slice view. Both track the same slice.")
            else:
                m = int(s["live3d_meas"])
                ang, off, nb = seq3d[m]
                st.pyplot(
                    _sinogram_figure(
                        sino[:, m, :], IMAGE_RES, "Slice z", None,
                        "2D projection of measurement %d  ·  %.0f°, offset %.1f, %s"
                        % (m, ang, off, "full fan" if nb == 0 else "%d beams" % nb),
                    ),
                    use_container_width=True,
                )
                if n_meas > 1:
                    st.slider("Measurement", 0, n_meas - 1, key="live3d_meas")
                st.caption(
                    "Every slice's reading for this one measurement, stacked into a 2D "
                    "detector panel behind the volume would record."
                )

        with view_recon:
            # 2.5D geometry is what makes this exact rather than an approximation: one
            # measurement fires the SAME (r, theta) bundle through every slice and rays never
            # cross slices, so each z-slice is an independent 2D tomography problem sharing one
            # geometry. Each is handed to run_simple_uq exactly as the 2D tab hands it its
            # phantom -- clean slice plus I0/alpha/beta, degradation applied inside the forward
            # model -- and the reconstructions are stacked back into a volume.
            st.caption(
                "Every z-slice is reconstructed **independently**, by the same "
                "`run_simple_uq` forward + inverse + k_aug pipeline the 2D tab uses, then "
                "stacked back into a volume. Needs IPOPT and k_aug; the rest of this tab does "
                "not."
            )
            stride = int(s["live3d_recon_stride"])
            targets = list(range(0, n_slices, stride))

            rc1, rc2, rc3 = st.columns([1, 1, 2])
            with rc1:
                st.select_slider(
                    "Stride", options=(1, 2, 4, 8), key="live3d_recon_stride",
                    help="1 is every slice. A coarse pass is cheap and its slices are cached, "
                         "so committing to stride 1 afterwards only pays for the new ones.",
                )
            with rc2:
                st.slider("TV weight", 0.0, 1.0, step=0.01, key="live3d_tv_weight",
                          help="Total-variation regularisation, the same dial the 2D tab has.")
            with rc3:
                if not n_all:
                    st.caption("Take at least one measurement first.")
                elif float(s["live3d_I0"]) > 0.0:
                    # Measured on this box: 9 measurements at IMAGE_RES=30 solve in 5.7 s at
                    # I0 = 0, and had still not converged after 8.5 MINUTES at I0 = 5. The
                    # dose-response makes the forward model nonlinear and the NLP far harder,
                    # so no estimate is offered here -- it would be off by two orders.
                    st.warning(
                        "**I0 = %.3g, so degradation is modelled in the solve and each slice "
                "gets dramatically more expensive**. It was measured at 5.7 s per slice "
                        "with I0 = 0 against over 8 minutes at I0 = 5, for the same geometry. "
                        "%d slices at that rate is hours. Reconstruct with a large stride "
                        "first, or set I0 = 0 to reconstruct the undamaged phantom."
                        % (float(s["live3d_I0"]), len(targets))
                    )
                else:
                    # ~0.6 s per measurement per slice at IMAGE_RES=30, I0 = 0, measured here.
                    est = 0.6 * n_all * len(targets)
                    st.caption(
                        "**%d** of %d slices · %d measurement%s · rough estimate "
                        "**%s** (cached slices are instant)."
                        % (len(targets), n_slices, n_all, "" if n_all == 1 else "s",
                           "%.0f s" % est if est < 90 else "%.1f min" % (est / 60.0))
                    )

            go = st.button("Reconstruct slices", type="primary", key="btn3d_recon",
                           disabled=(n_all == 0), use_container_width=False)

            # Signature of everything the stack depends on. Stored alongside the result so a
            # stale stack (parameters moved since) is reported rather than silently shown.
            recon_key = (seq_all, IMAGE_RES, n_slices, float(s["live3d_contrast"]),
                         float(s["live3d_I0"]), float(s["live3d_alpha"]),
                         float(s["live3d_beta"]), float(s["live3d_tv_weight"]))

            if go:
                prog = st.progress(0.0, text="Starting...")
                log_box = st.empty()
                log_lines: list[str] = []
                _last = [0.0]
                _ctx = _current_script_ctx()

                def _render_log() -> None:
                    _render_log_box(log_box, "".join(log_lines))

                def _log_cb(chunk: str) -> None:
                    # Append ONLY -- never render from in here. This callback runs inside the
                    # cached _recon_slice_3d, and a Streamlit element called from a cached
                    # function is recorded for replay; on a cache hit the replay fails with
                    # "a streamlit element is called", which would break exactly the slices the
                    # stride workflow is meant to reuse. The loop flushes between slices.
                    log_lines.append(chunk)

                # NaN, not 0: an unsolved slice must be distinguishable from a genuinely dark
                # one, the same reason the sinogram leaves unmeasured cells NaN.
                recon = np.full((IMAGE_RES, IMAGE_RES, n_slices), np.nan)
                logcov = np.full((IMAGE_RES, IMAGE_RES, n_slices), np.nan)
                dopt = [float("nan")] * n_slices
                status = ["not solved"] * n_slices

                for i, kz in enumerate(targets):
                    prog.progress(i / max(len(targets), 1),
                                  text="Slice %d of %d (z = %d)" % (i + 1, len(targets), kz))
                    try:
                        a, c, dv, fwd, inv = _recon_slice_3d(
                            seq_all, IMAGE_RES, n_slices, float(s["live3d_contrast"]),
                            float(s["live3d_I0"]), float(s["live3d_alpha"]),
                            float(s["live3d_beta"]), float(s["live3d_tv_weight"]), kz,
                            _log_callback=_log_cb,
                            _cancel_callback=lambda: _script_stop_requested(_ctx),
                        )
                        recon[:, :, kz] = a
                        if c is not None:
                            logcov[:, :, kz] = c
                        dopt[kz] = dv
                        status[kz] = "%s / %s" % (fwd, inv)
                    except SolverCancelled:
                        st.stop()
                    except Exception as exc:
                        # One slice must not lose the rest: k_aug's covariance here is
                        # intentionally rank-deficient and can fail on a starved geometry.
                        status[kz] = "failed: %s" % (str(exc).splitlines() or [""])[0][:120]
                    # Flush the accumulated solver output between slices (see _log_cb).
                    now = time.time()
                    if now - _last[0] >= 0.2:
                        _last[0] = now
                        _render_log()
                prog.progress(1.0, text="Done: %d slices" % len(targets))
                _render_log()
                st.session_state["results_3d"] = {
                    "recon": recon, "logcov": logcov, "dopt": dopt, "status": status,
                    "targets": targets, "stride": stride, "key": recon_key,
                }

            res3 = st.session_state.get("results_3d")
            if res3 is None:
                st.info("Build a measurement sequence, then press **Reconstruct slices**.")
            else:
                if res3["key"] != recon_key:
                    st.warning("Parameters or the sequence changed since this stack was "
                               "solved. Press **Reconstruct slices** again to refresh it.")
                solved = [k2 for k2 in res3["targets"]
                          if not str(res3["status"][k2]).startswith(("failed", "not solved"))]
                failed = [k2 for k2 in res3["targets"]
                          if str(res3["status"][k2]).startswith("failed")]
                if not solved:
                        st.error("Every slice failed. There is nothing to stack. See the status table "
                             "below; a geometry with too few rays starves the sensitivity step.")
                else:
                    if failed:
                        st.warning("%d of %d slices failed and are left empty in the stack: z = %s"
                                   % (len(failed), len(res3["targets"]),
                                      ", ".join(str(x) for x in failed)))
                    # "Show" sits above the figure because it chooses what the figure IS.
                    # Everything below the figure is *viewing* geometry and is deliberately
                    # independent of it, so switching source back and forth to compare the
                    # reconstruction against its variance leaves the view exactly as set.
                    st.radio("Show", ("Reconstruction", "Variance"), key="live3d_recsrc",
                             horizontal=True,
                             help="'Variance' is the posterior log10 variance from k_aug. "
                                  "where this geometry leaves the image uncertain.")

                    if s["live3d_recsrc"] == "Variance":
                        # -inf is a real output here: a pixel no ray constrains has exactly zero
                        # variance. Drop those to NaN so they are simply absent rather than
                        # dragging the colour ramp to negative infinity.
                        vdata = np.where(np.isfinite(res3["logcov"]), res3["logcov"], np.nan)
                        vcmap, vlabel = "Viridis", "Posterior log10 variance"
                    else:
                        vdata = res3["recon"]
                        vcmap, vlabel = "Gray", "Stacked reconstruction"
                    _fin = vdata[np.isfinite(vdata)]
                    _lo, _hi = ((float(_fin.min()), float(_fin.max())) if _fin.size else (0.0, 1.0))
                    _span = (_hi - _lo) or 1.0
                    # Read before the widgets are built: these hold the live values either way,
                    # since Streamlit applies widget state before re-running the script body.
                    _p0, _p1 = s["live3d_recband_pct"]
                    _isomin = _lo + _span * float(_p0) / 100.0
                    _isomax = _lo + _span * float(_p1) / 100.0

                    rv1, rv2 = st.columns([3, 2])
                    with rv1:
                        st.plotly_chart(
                            _volume_figure(
                                vdata, float(s["live3d_opacity"]), _isomin, _isomax,
                                "%s: %d of %d slices" % (vlabel, len(solved), n_slices),
                                vcmap, _hi,
                                cut_axis=s["live3d_cutaxis"], cut_frac=float(s["live3d_cut"]),
                                cmin=_lo,
                            ),
                            use_container_width=True,
                        )
                        st.caption("Range %.4g to %.4g; drawing %.4g to %.4g."
                                   % (_lo, _hi, _isomin, _isomax))
                    with rv2:
                        finite = [(z, res3["dopt"][z]) for z in solved
                                  if np.isfinite(res3["dopt"][z])]
                        if finite:
                            st.markdown("**D-optimality by slice**")
                            st.line_chart(
                                pd.DataFrame({"D-optimality": [v for _, v in finite]},
                                             index=[z for z, _ in finite]),
                                height=200,
                            )
                            st.caption("How well each depth is constrained by this geometry. "
                                       "The 2.5D bundle is identical for every slice, so the "
                                       "spread here is the phantom's, not the geometry's.")
                        st.dataframe(
                            pd.DataFrame({"z": res3["targets"],
                                          "status": [res3["status"][z] for z in res3["targets"]],
                                          "D-opt": [res3["dopt"][z] for z in res3["targets"]]}),
                            use_container_width=True, hide_index=True, height=180,
                        )

                    # View controls, under the figure and spanning the full width rather than
                    # squeezed into a quarter of it. Seeded from the canonical values
                    # immediately before the widgets are built -- the last moment Streamlit
                    # allows a widget key to be written, and late enough to pick up a change
                    # made in the Volume view earlier in this same run.
                    s["live3d_opacity_rec"] = float(s["live3d_opacity"])
                    s["live3d_cutaxis_rec"] = s["live3d_cutaxis"]
                    s["live3d_cut_rec"] = float(s["live3d_cut"])
                    qc1, qc2, qc3, qc4 = st.columns(4)
                    with qc1:
                        st.slider("Opacity", 0.02, 1.0, step=0.01, key="live3d_opacity_rec",
                                  on_change=_cb3d_sync_recview,
                          help="Shared with the Volume view. Both track one value.")
                    with qc2:
                        st.slider("Visible range (%)", 0, 100, step=1,
                                  key="live3d_recband_pct",
                                  help="Band to draw, as a percentage of the current view's own "
                                       "range. Percent rather than absolute because the two "
                                       "views are not in the same units: attenuation runs 0-1 "
                                       "and log10 variance is negative throughout, so one "
                                       "setting keeps meaning the same thing across both.")
                    with qc3:
                        st.selectbox("Cut away", tuple(_CUT_AXES), key="live3d_cutaxis_rec",
                                     on_change=_cb3d_sync_recview,
                                     help="Shared with the Volume view.")
                    with qc4:
                        st.slider("Cut amount", 0.0, 0.95, step=0.05, key="live3d_cut_rec",
                                  on_change=_cb3d_sync_recview)
                    zsel = st.slider("Inspect slice z", 0, max(n_slices - 1, 0),
                                     key="live3d_recon_z")
                    st.pyplot(
                        _recon_slice_figure(vol0[:, :, zsel], res3["recon"][:, :, zsel],
                                            res3["logcov"][:, :, zsel], zsel,
                                            str(res3["status"][zsel])),
                        use_container_width=True,
                    )

    with mid3d:
        st.markdown("**Next measurement**")
        # The three bundle sliders live inside the component under the Slice picture (they have
        # to -- see there). This is the read-out, so the current aim is legible from any sub-tab.
        st.caption(
            "Angle **%.0f\u00b0** \u00b7 offset **%.1f** \u00b7 **%s** \u2014 set these under the "
            "picture in the **Slice** view (live preview) or in the **Volume** view\'s *Aim* "
            "column."
            % (float(s["live3d_angle"]), float(s["live3d_offset"]),
               "full fan" if int(s["live3d_nbeams"]) == 0
               else "%d beams" % int(s["live3d_nbeams"]))
        )
        # One toggle for both overlays, kept here rather than in a sub-tab so it is reachable
        # whichever view is open (mirrors the 2D tab's beams_visible).
        st.checkbox("Show beams", key="live3d_showbeams",
                    help="Draw the bundle on the Slice picture and through the Volume view.")
        st.button("➕ Take measurement", key="btn3d_step", on_click=_cb3d_step,
                  use_container_width=True)
        st.button("Reset", key="btn3d_reset", on_click=_cb3d_reset,
                  use_container_width=True)

        st.markdown("**Volume & dose**")
        st.slider("Number of slices", _N_SLICES_MIN, _N_SLICES_MAX, step=1,
                  key="live3d_nslices",
                  help="How many z-slices the volume is built from. Not to be confused with "
                       "\"Slice (z)\" in the Slice / Sinogram views, which picks which one to show.")
        st.slider("Phantom contrast", 0.0, 2.0, step=0.05, key="live3d_contrast",
                  help="How far the interior structures stand above the brain background. "
                       "0 leaves a uniform 0.1 interior; 1 gives ventricles/blobs 0.3 and the "
                       "two floating spheres 0.4; above 1 exaggerates. Brain and skull are not "
                       "scaled, so the crust stays at 1.0 and the Volume colour scale never "
                       "moves. Nothing inside the skull is ever 0; that value means air.")
        st.number_input("I0 (0 = no degradation)", min_value=0.0, step=0.5,
                        key="live3d_I0")
        st.number_input("alpha", min_value=0.0, step=0.05, format="%.3f", key="live3d_alpha")
        st.number_input("beta", min_value=0.0, step=0.005, format="%.3f", key="live3d_beta")

    with right3d:
        _preset_block("live3d", "beam_table_3d", "live3d_view_k")

        st.markdown("**Measurement sequence**")
        st.dataframe(s["beam_table_3d"], use_container_width=True, hide_index=False)
        total0 = float(vol0.sum())
        total1 = float(vol.sum())
        lost = 0.0 if total0 == 0 else 100.0 * (1.0 - total1 / total0)
        st.metric("Intensity removed", "%.1f%%" % lost,
                  help="Total volume intensity lost to dose across all slices.")
        st.caption("Slice z=%d mean: %.4f" % (k, float(vol[:, :, k].mean())))


# --- 2D implicit-transport shrinkage (shrinkage-decay) tab: forward simulation only --------------------
# v5 with steps 5 and 6 changed. The explicit upwind flux and its compaction number are gone;
# eq:xd_implicit_transport is one sparse solve whose matrix is an M-matrix with unit column
# sums, so positivity and exact conservation hold with no step-size condition. What replaces
# C_k on the readout is the pair that CAN go wrong here: eta against the max |dP| it has to
# discriminate, and gamma(1-sigma)/varsigma inside the bulk.
# The measurement schedule. Same total exposure either way -- every row's bundle is fired with
# the same I0 -- and the two differ only in how it is split in time, which is the fractionation
# question experiment_shrinkage_fractionation.py measures (3.46x at c_omega = 0.4).
_SHRINKAGE_MODES = ("Simultaneous", "Sequential")
# TV weight ladder for the shrinkage-decay Reconstruct: 1-2-5 per decade from 1e-4 to 100.
#
# The top of this range is not decoration. shrinkage-decay's estimation NLP does not converge at grid 32 --
# 100 iterations leave inf_du at 1.2e+06 with 62% of iterations Hessian-regularised -- and a
# heavy TV term is one of the few levers that makes the objective more strongly convex, so the
# region above 1 is worth being able to reach even though it will visibly over-smooth.
_SHRINKAGE_TV_WEIGHTS = (0.0, 1e-4, 2e-4, 5e-4, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
                  0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
_SHRINKAGE_RESOLUTIONS = (32, 48, 64, 96)
_SHRINKAGE_NAIVE_METHODS = (
    "FBP", "SART", "SART (5 passes)", "SART (10 passes)",
    "Pyomo (1 simultaneous step)",
)
from senDOE.models.tomography_2d_shrinkage_decay import ETA_RATIO_TARGET as _ETA_TARGET_SHRINKAGE
_SHRINKAGE_ETAS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.5)
_SHRINKAGE_FREFS = (0.0002, 0.001, 0.002, 0.01, 0.05, 0.2)


def _cb_shrinkage_step():
    s = st.session_state
    s["beam_table_shrinkage"] = pd.concat(
        [s["beam_table_shrinkage"], pd.DataFrame([{
            "angle_deg": float(s["shrinkage_angle"]), "offset": float(s["shrinkage_offset"]),
            "n_beams": int(s["shrinkage_nbeams"])}])], ignore_index=True)
    s["shrinkage_view_k"] = len(s["beam_table_shrinkage"])


def _cb_shrinkage_reset():
    st.session_state["beam_table_shrinkage"] = _empty_beam_table()


def _cb_sync_live_sim_shrinkage():
    _sync_live_sim("shrinkage", "live_sim_shrinkage")


@st.cache_data(show_spinner=False)
def _simulate_shrinkage(seq: tuple, image_res: int, optical_depth: float, I0: float, c_omega: float,
                 c_cp: float, a: float, b: float, reach: float, gamma: float, fref: float,
                 eta: float, mode: str = "Simultaneous", auto_eta: bool = True):
    """``(theta, f, phi, summary)``.  Radii are about the FIXED initial centroid.

    Shape statistics come from ``senDOE.helpers.shape_metrics.shape_diagnostics`` rather than
    being re-derived here, so the tab cannot disagree with the module it mirrors.
    """
    theta = scale_to_optical_depth(_phantom(image_res), float(optical_depth), int(image_res))
    _mk = lambda e: ShrinkageDecayParams(I0=float(I0), c=float(c_omega), a=float(a), b=float(b),
                             c_cp=float(c_cp), reach=float(reach), gamma=float(gamma),
                             f_ref_frac=float(fref), eta=float(e))
    p = _mk(eta)
    # Resolve eta HERE, through the same select_eta the reconstruction uses, so the picture, the
    # metrics and the solve are one model. A tab whose live view ran one eta while Reconstruct
    # ran another would be showing physics the solve is not doing.
    eta_info = {"warning": ""}
    if auto_eta and seq:
        try:
            eta_used, eta_info = _select_eta_shrinkage(theta, seq, p, int(image_res),
                                                simultaneous=(mode == _SHRINKAGE_MODES[0]))
            p = _mk(eta_used)
        except Exception:
            pass
    # One step carrying every bundle, or one step per bundle. Both come from the model's
    # single step_bundles body, so the two schedules cannot drift apart.
    _run = simulate_shrinkage_sim if mode == _SHRINKAGE_MODES[0] else simulate_shrinkage_seq
    try:
        f, infos = _run(theta, seq, p, int(image_res))
        err = None
    except Exception as exc:                 # the potential's guards raise rather than return junk
        f, infos, err = theta.copy(), [], str(exc).split(":")[0]

    sup_pct, half_pct, flips = shape_diagnostics_shrinkage(theta, f)
    nr, nc = theta.shape
    yy, xx = np.mgrid[0:nr, 0:nc]
    m0 = float(theta.sum())
    cx0, cy0 = ((xx * theta).sum() / m0, (yy * theta).sum() / m0) if m0 > 0 else (0.0, 0.0)
    rad = np.sqrt((xx - cx0) ** 2 + (yy - cy0) ** 2)
    d = f - theta

    phi_panel = np.zeros_like(theta)
    if infos and seq:
        pr = resolve_shrinkage(p, theta)
        # Match the schedule: simultaneous accumulates EVERY bundle against the same field, so a
        # panel built from the last row alone would show a potential the run never solved.
        rows = seq if mode == _SHRINKAGE_MODES[0] else seq[-1:]
        cid = np.zeros_like(theta)
        I_p = np.zeros_like(theta)
        for ang, off, nb in rows:
            dq, ip = _accumulate_dose(f, _bundle_r_values(float(off), int(nb), int(image_res)),
                                      float(np.deg2rad(float(ang))), pr.I0, pr.c)
            cid = cid + dq
            I_p = I_p + ip
        try:
            phi_panel, _Pi, _sg = _compaction_potential_shrinkage(f * pr.decay_factor(I_p),
                                                           1.0 - np.exp(-cid), pr)
        except Exception:
            pass

    dP_max = max((i.dP_max for i in infos), default=0.0)
    summary = {
        "err": err, "mass0": m0, "mass1": float(f.sum()),
        "sup_pct": sup_pct, "half_pct": half_pct, "flips": flips,
        "inner": float(d[rad < 0.25 * nr].sum()), "outer": float(d[rad > 0.30 * nr].sum()),
        "f_min": min((i.state_min for i in infos), default=float(f.min())),
        "phi_cr": (infos[-1].phi_core_rim if infos else float("nan")),
        "dw_max": max((i.dw_max for i in infos), default=0.0),
        # What replaces C_k. colsum is conservation, mass_resid is prop:xd_mass measured per
        # step, absorp is the condition subsec:system does not state, eta_ratio is whether the
        # softplus is still resolving direction.
        "colsum": max((i.colsum_err for i in infos), default=0.0),
        "mass_resid": max((i.mass_residual for i in infos), default=0.0),
        "absorp": max((i.absorption_ratio for i in infos), default=0.0),
        "dP_max": dP_max,
        "rest_rate": max((i.rest_rate for i in infos), default=0.0),
        "eta_ratio": (float(p.eta) / dP_max) if dP_max > 0 else float("nan"),
        "eta_used": float(p.eta), "eta_auto": bool(auto_eta),
        "eta_warning": str(eta_info.get("warning", "")),
    }
    return theta, f, phi_panel, summary


def _render_2d_shrinkage_tab():
    """Implicit-transport shrinkage: unconditionally positive and conservative. Forward only."""
    s = st.session_state
    st.caption(
        "**Implicit-transport shrinkage model.** Build a measurement sequence, preview how "
        "decay and transport change the field, then reconstruct the undamaged field and its "
        "uncertainty. Transport is solved implicitly with a nonnegative softplus rate, which "
        "preserves positivity and mass conservation without a step-size limit."
    )
    left, mid, right = st.columns([3, 2, 2])
    res = int(s["shrinkage_res"])
    seq_all = _table_to_seq(s["beam_table_shrinkage"])
    n_all = len(seq_all)
    seq = seq_all[:max(0, min(int(s["shrinkage_view_k"]), n_all))]
    n_meas = len(seq)
    theta, f, phi, summary = _simulate_shrinkage(
        seq, res, float(s["shrinkage_depth"]), float(s["shrinkage_I0"]), float(s["shrinkage_c_omega"]),
        float(s["shrinkage_c_cp"]), float(s["shrinkage_a"]), float(s["shrinkage_b"]), float(s["shrinkage_reach"]),
        float(s["shrinkage_gamma"]), float(s["shrinkage_fref"]), float(s["shrinkage_eta"]), str(s["shrinkage_mode"]),
        bool(s["shrinkage_eta_auto"]))
    simultaneous = (str(s["shrinkage_mode"]) == _SHRINKAGE_MODES[0])

    view = "Attenuation"
    panel, vlo, vhi = f, 0.0, max(float(theta.max()), 1e-12)

    with left:
        _nav_block("shrinkage_view_k", n_all)
        _live_sim(
            image_uri=_live_background_uri(panel, vlo, vhi), image_res=res, k=n_meas,
            angle=float(s["shrinkage_angle"]), offset=float(s["shrinkage_offset"]),
            nbeams=int(s["shrinkage_nbeams"]),
            committed=(list(seq[-1]) if n_meas else None),
            beams_visible=bool(s["shrinkage_showbeams"]),
            angle_range=[0, 360, 1], offset_range=[-float(res)/2, float(res)/2, 0.5],
            nbeams_range=[0, res, 1],
            title="%s   ·   %d bundle%s %s   ·   %d×%d"
                  % (view, n_meas, "" if n_meas == 1 else "s",
                     "fired together" if simultaneous else "fired in sequence", res, res),
            hint='<b style="color:#ff2b2b">Red</b> = next-measurement preview; '
                 '<b style="color:#1f77ff">blue</b> = last measurement taken.',
            legend="%.3g" % vhi,
            default={"angle": float(s["shrinkage_angle"]), "offset": float(s["shrinkage_offset"]),
                     "nbeams": int(s["shrinkage_nbeams"])},
            key="live_sim_shrinkage", on_change=_cb_sync_live_sim_shrinkage)

    with mid:
        st.radio(
            "Measurement schedule", _SHRINKAGE_MODES, key="shrinkage_mode", horizontal=True,
            help="**Simultaneous** fires every row of the table in ONE exposure: the dose fields "
                 "are summed against the same starting field, then one decay, one potential "
                 "solve and one transport solve. **Sequential** fires one exposure per row, each "
                 "seeing the damage the previous ones did.\n\nSame total exposure either way, so "
                 "the difference is fractionation, not dose, and it is large because "
                 "the converted fraction saturates. Measured at c_omega = 0.4, 10 angles: "
                 "sequential contracts **3.46x** as much. Run the fractionation experiment "
                 "script for the full comparison.")
        act = st.columns(2)
        act[0].button("➕ Take measurement", on_click=_cb_shrinkage_step,
                      use_container_width=True, key="shrinkage_take")
        act[1].button("Reset", on_click=_cb_shrinkage_reset, use_container_width=True, key="shrinkage_clear")
        st.checkbox("Show beams", key="shrinkage_showbeams")
        with st.expander("Compaction reach and amplitude", expanded=True):
            st.slider("l: compaction reach (px)", 0.5, 32.0, step=0.5, key="shrinkage_reach",
                      help="A PHYSICAL length, so unlike c_cp it transfers across grids. As "
                           "as l approaches 0, the potential approaches the pointwise driver "
                           "the local potential driver; a reach comparable to "
                           "the specimen radius (~15 px here) gives whole-body contraction. "
                           "Below about R/2 the potential is still rim-peaked.")
            st.slider("c_cp: compaction amplitude", 0.0, 3.0, step=0.05, key="shrinkage_c_cp",
                      help="0 turns transport off. Higher values increase transport while the "
                           "implicit solve preserves positivity.")
        with st.expander("Beam, conversion and decay", expanded=True):
            st.slider("I0: incident intensity", 0.0, 5.0, step=0.1, key="shrinkage_I0")
            st.slider("c_omega: conversion", 0.0, 3.0, step=0.05, key="shrinkage_c_omega")
            st.slider("a: decay", 0.0, 0.5, step=0.005, format="%.3f", key="shrinkage_a",
                      help="a = b = 0 conserves mass EXACTLY, whatever c_cp does. That is the "
                           "clean shrinkage test: any change in support is then transport alone.")
            st.slider("b: quadratic decay", 0.0, 0.05, step=0.001, format="%.3f",
                      key="shrinkage_b")
        # Transport and potential numerics is intentionally hidden from the UI.
        # with st.expander("Transport and potential numerics", expanded=False):
        #     st.checkbox(
        #         "Auto η from the forward run", key="shrinkage_eta_auto",
        #         help="Sets eta = max|dP| / %g from a forward run. No fixed value works: max|dP| "
        #              "spans 0.15 to 80 across this tab's sliders. Too small and the estimation "
        #              "NLP degenerates (at eta = 1e-3, 97%% of IPOPT iterations were "
        #              "Hessian-regularised and inf_du climbed to 6.6e9); too large and the "
        #              "smoothing does the transport's job. Not circular: max|dP| is set in step "
        #              "4, which never reads eta." % _ETA_TARGET_SHRINKAGE)
        #     st.select_slider(
        #         "eta — softplus smoothing (manual)", options=_SHRINKAGE_ETAS, key="shrinkage_eta",
        #         disabled=bool(s["shrinkage_eta_auto"]),
        #         format_func=lambda v: "%.0e" % v,
        #         help="THE RATE FUNCTION IS NOT ZERO AT ZERO. phi_eta(0) = eta·log2, so at "
        #              "rest both directed rates are c_cp·eta·log2 and a motionless "
        #              "field still diffuses. Set I0 = 0 and raise eta to watch it: the spec says "
        #              "that case must be an exact identity, and it is not. Keep eta well under "
        #              "max |dP| below, or the rate stops discriminating direction.")
        #     st.select_slider("f_ref / f_max", options=_SHRINKAGE_FREFS, key="shrinkage_fref",
        #                      help="Density at which material starts conducting. The stated rule "
        #                           "— at most a fifth of the smallest interior value — "
        #                           "is NOT sufficient: what must be small is "
        #                           "gamma·exp(-f_int/f_ref)/varsigma, shown as absorp/vs "
        #                           "below. At 0.2 that ratio is ~33 and the specimen does not "
        #                           "move at all, silently.")
        #     st.select_slider("gamma — vacuum absorption", options=(1.0, 10.0, 100.0, 1000.0),
        #                      key="shrinkage_gamma", help="Sets how fast the potential decays into "
        #                                           "vacuum. Needs gamma >> 1, but raising it also "
        #                                           "raises absorp/vs.")
        #     st.select_slider("Grid", options=_SHRINKAGE_RESOLUTIONS, key="shrinkage_res")

    with right:
        _preset_block("shrinkage", "beam_table_shrinkage", "shrinkage_view_k")
        st.markdown("**Measurement sequence**")
        st.dataframe(
            s["beam_table_shrinkage"].rename(columns={
                "angle_deg": "Angle (degrees)",
                "offset": "Bundle offset",
                "n_beams": "Number of beams",
            }),
            use_container_width=True, height=180)
        st.subheader("Solver Tuning")
        rc = st.columns(4)
        with rc[0]:
            st.select_slider("TV regularization weight", options=_SHRINKAGE_TV_WEIGHTS,
                             key="shrinkage_tv_weight",
                             format_func=lambda v: "%g" % v,
                             help="Both objective terms are normalised to O(1) first, so this is a "
                                  "trade-off **ratio**, not the 2D tab's scale: a value of 1 "
                                  "means TV and the data fit carry equal weight.\n\nRange runs to "
                                  "100, well past the point of visible over-smoothing. That is "
                                  "deliberate: the estimation NLP does not converge at grid 32, and a heavy "
                                  "TV term is one of the few levers that makes the objective more "
                                  "strongly convex. Expect a smoother reconstructed sample at high values.")
        with rc[1]:
            st.number_input("Max IPOPT iterations", min_value=1500, max_value=8000, step=50,
                            key="shrinkage_maxiter",
                            help="Measured at grid 32 / K=5 simultaneous: `optimal` at iteration "
                                 "**1253**, 219 s (~0.175 s/iter on ma97). Grid 16 converges in "
                                 "520-642. A cap below ~1500 will cut grid 32 off before it gets "
                                 "there and report maxIterations, which reads as a failed solve "
                                 "rather than a truncated one.")
        with rc[2]:
            st.selectbox("Linear Solver", options=("ma97", "ma57", "ma27"),
                         key="shrinkage_linear_solver",
                         help="The selected solver is tried first. If it fails numerically or is "
                              "unavailable, reconstruction tries the other supported MA solvers.")
        with rc[3]:
            st.selectbox(
                "Naive reconstruction", options=_SHRINKAGE_NAIVE_METHODS,
                key="shrinkage_naive_method",
                help="Comparison panel only. FBP and the selected-pass SART option use the "
                     "direct damaged sinogram. Pyomo fits all measurements in one simultaneous step with "
                     "zero incident intensity and no degradation or transport.")
        actions = st.columns(2)
        go_shrinkage = actions[0].button(
            "Reconstruct", type="primary", key="btn_shrinkage_recon",
            disabled=(n_all == 0), use_container_width=True)
        go_shrinkage_naive = actions[1].button(
            "Reconstruct Naive", key="btn_shrinkage_naive_recon",
            disabled=(n_all == 0), use_container_width=True)
        # NO STANDING DIAGNOSTIC BLOCK. The metrics and the two numeric captions that used to sit
        # here (half-mass / support radius / sign changes / total attenuation, then min f,
        # |colsum-1|, transport residual, phi centre/rim, max |dP|, eta/max|dP|, resting rate,
        # absorp/vs) were removed as confusing: most of them are healthy at every setting the tab
        # offers, so they read as a wall of numbers demanding interpretation when there is
        # nothing to interpret. What is left below fires only when something actually needs the
        # user's attention. The numbers themselves are unchanged and still computed in
        # `_simulate_shrinkage`'s `summary`; the model module's check_invariants prints the full set.
        # The rim-peaked (centre/rim < 1) banner is gone too. It fired on essentially every
        # Shepp-Logan setting, because that phantom's mass sits in a bright ring and the void
        # source is weighted by material present, so phi peaks near r ~ 10 px and the ratio
        # saturates at 0.70 whatever the reach. A banner that is always on carries no
        # information and, worse, reads as a fault when nothing is wrong: positivity and
        # conservation are structural here and hold at every setting. The ratio is still in
        # StepInfo6.phi_core_rim for anyone who wants it.

    # Signature of everything the answer depends on, stored with it: a stale result is reported
    # rather than silently shown. Same guard the v2, v5 and 3D surfaces use.
    shrinkage_key = (seq_all, res, float(s["shrinkage_depth"]), float(s["shrinkage_I0"]), float(s["shrinkage_c_omega"]),
              float(s["shrinkage_c_cp"]), float(s["shrinkage_a"]), float(s["shrinkage_b"]), float(s["shrinkage_reach"]),
              float(s["shrinkage_gamma"]), float(s["shrinkage_fref"]), float(s["shrinkage_eta"]),
              str(s["shrinkage_mode"]), float(s["shrinkage_tv_weight"]), int(s["shrinkage_maxiter"]),
              str(s["shrinkage_linear_solver"]))
    naive_key = shrinkage_key + (str(s["shrinkage_naive_method"]),)

    # A full reconstruction refreshes the naïve result too, so Naive fit RMSE always refers to
    # the same current sequence and physics settings. The dedicated button still runs naïve only.
    run_naive = bool(go_shrinkage_naive or go_shrinkage)
    if go_shrinkage or run_naive:
        log_box = st.empty()
        log_lines: list[str] = []
        _ctx = _current_script_ctx()

        def _render_shrinkage_log() -> None:
            _render_log_box(log_box, "".join(log_lines))

        def _shrinkage_log(chunk: str) -> None:
            # Pyomo calls this on its reader thread. Streamlit elements are not thread-safe, so
            # only buffer and echo here; the script thread renders at solve boundaries below.
            _term_echo(chunk)
            log_lines.append(chunk)

        params_shrinkage = ShrinkageDecayUQParams(
            image_res=res, optical_depth=float(s["shrinkage_depth"]), beam_steps=seq_all,
            simultaneous=simultaneous,
            I0=float(s["shrinkage_I0"]), c=float(s["shrinkage_c_omega"]), a=float(s["shrinkage_a"]),
            b=float(s["shrinkage_b"]), c_cp=float(s["shrinkage_c_cp"]), reach=float(s["shrinkage_reach"]),
            gamma=float(s["shrinkage_gamma"]), f_ref_frac=float(s["shrinkage_fref"]),
            eta=(None if bool(s["shrinkage_eta_auto"]) else float(s["shrinkage_eta"])),
            tv_weight=float(s["shrinkage_tv_weight"]),
            naive_method=str(s["shrinkage_naive_method"]),
            noise_sigma=0.0, ipopt_max_iter=int(s["shrinkage_maxiter"]),
            linear_solver=str(s["shrinkage_linear_solver"]))
        if go_shrinkage:
            with st.spinner("Solving the estimation NLP, then k_aug..."):
                try:
                    out = run_shrinkage_decay_reconstruction(
                        params_shrinkage, log_callback=_shrinkage_log,
                        cancel_callback=lambda: _script_stop_requested(_ctx))
                    st.session_state["results_shrinkage"] = {"res": out, "key": shrinkage_key}
                except SolverCancelled:
                    st.stop()
                except Exception as exc:
                    st.session_state.pop("results_shrinkage", None)
                    st.error("Reconstruction failed: %s" % exc)
                finally:
                    _render_shrinkage_log()
        if run_naive:
            with st.spinner("Computing the selected naive reconstruction..."):
                try:
                    out = run_naive_shrinkage_reconstruction(
                        params_shrinkage, log_callback=_shrinkage_log,
                        cancel_callback=lambda: _script_stop_requested(_ctx))
                    st.session_state["results_shrinkage_naive"] = {
                        "res": out, "key": naive_key}
                except SolverCancelled:
                    st.stop()
                except Exception as exc:
                    st.session_state.pop("results_shrinkage_naive", None)
                    st.error("Naive reconstruction failed: %s" % exc)
                finally:
                    _render_shrinkage_log()

    stash_shrinkage = st.session_state.get("results_shrinkage")
    regular = stash_shrinkage["res"] if stash_shrinkage else None
    stash_naive = st.session_state.get("results_shrinkage_naive")
    naive = stash_naive["res"] if stash_naive else None
    naive_error_pct = "n/a"
    naive_fit_rms = "n/a"
    if naive is not None and getattr(naive, "theta_naive", None) is not None:
        naive_truth = np.asarray(naive.theta_true, dtype=float)
        naive_image = np.asarray(naive.theta_naive, dtype=float)
        if np.isfinite(naive_truth).all() and np.isfinite(naive_image).all():
            naive_scale = max(float(np.abs(naive_truth).max()), 1e-30)
            naive_error_pct = "%.3f%%" % (
                100.0 * float(np.sqrt(np.mean((naive_image - naive_truth) ** 2))) /
                naive_scale)
        if np.isfinite(getattr(naive, "fit_rms", np.nan)):
            naive_fit_rms = "%.3e" % naive.fit_rms
    if stash_shrinkage:
        out = regular
        if stash_shrinkage["key"] != shrinkage_key:
            st.warning("These results are STALE. A setting changed since they were "
                       "computed. Press Reconstruct again.")
        ok = out.status == "optimal"
        (st.success if ok else st.warning)(
            "%s in %.1f s on %s · %d variables, %d constraints · %s iterations"
            % (out.status, out.t_solve, out.linear_solver, out.n_vars, out.n_cons, out.iters))
        mm = st.columns(6)
        mm[0].metric("Reconstruction error", "%.3f%%" % out.theta_pct_peak,
                     help="RMS difference between the reconstructed sample and the original "
                          "sample, as a percent of the original sample peak.")
        mm[1].metric("Naïve reconstruction error", naive_error_pct,
                     help="RMS difference between the naïve reconstructed sample and the original "
                          "sample, as a percent of the original sample peak.")
        mm[2].metric("Fit RMSE", "%.3e" % out.obs_rms,
                     help="Residual of the projections the NLP actually minimised.")
        mm[3].metric("Naive fit RMSE", naive_fit_rms,
                     help="RMS projection residual after re-integrating the naïve image along "
                          "the measured rays.")
        mm[4].metric("D-optimality",
                     "%.4g" % out.d_optimality if out.d_optimality == out.d_optimality else "n/a",
                     help="Log-determinant of the posterior covariance derived from reconstruction "
                          "sensitivity to the measured projections. "
                          "Lower is a more informative design.")
        mm[5].metric("Half-mass change", "%+.3f%%" % out.half_pct,
                     help="What the damage did to the TRUE field. If this is ~0 the dynamics "
                          "barely moved anything and inverting them proved little.")
        if out.uq_error:
            st.info(
                "**k_aug did not return a covariance** (the reconstruction is kept): %s\n\n"
                "This is non-fatal by design. The covariance is intentionally rank deficient "
                ". A starved geometry leaves pixels that no ray constrains, so a "
                "singular KKT system is a statement about the *design*, not a bug. Add "
                "measurements or angles to condition it." % out.uq_error)
    if stash_naive:
        if stash_naive["key"] != naive_key:
            st.warning("Naive results are STALE. A setting changed since they were "
                       "computed. Press Reconstruct Naive again.")
        if getattr(naive, "error", ""):
            st.warning("Naive comparison (%s) was unavailable: %s" %
                       (getattr(naive, "method", "selected method"), naive.error))

    if regular is not None or naive is not None:
        st.pyplot(_shrinkage_recon_figure(regular, naive), use_container_width=True)
        st.caption(
            "**Reconstruct** refreshes both rows; **Reconstruct Naive** refreshes only the "
            "first row. The naïve result treats post-damage measurements as a static "
            "reconstruction problem and ignores the shrinkage-decay physics. "
            "The uncertainty panel shows the diagonal of the sensitivity-derived posterior "
            "covariance from k_aug."
        )


# --- three modes, three tabs ----------------------------------------------------------
tab_2d, tab_shrinkage, tab_3d = st.tabs(
    ["2D Decay Reconstruction", "2D Shrinkage Decay Reconstruction",
     "3D Decay Reconstruction"]
)

# The 3D tab is populated FIRST in script order: the 2D body below ends in a Reconstruct
# path that can call st.stop() (empty geometry), which would otherwise leave this tab blank
# on that rerun. Display order is set by the st.tabs() list above, not by population order.
with tab_3d:
    _render_3d_tab()

with tab_shrinkage:
    _render_2d_shrinkage_tab()

with tab_2d:
    # --- main: live dose-response simulator (the image is a derived view of the table) -----
    # image (left) | live-simulator dials + buttons (middle) | measurement-sequence table (right)
    left, mid, right = st.columns([3, 2, 2])

    _ph = _phantom(IMAGE_RES)
    _vmin, _vmax = float(_ph.min()), float(_ph.max())

    # The single source of truth: the table → sequence. The picture shows the first `_k`
    # measurements (Previous/Next scrub `_k` over 0..N); the table highlights measurement `_k`.
    _seq = _table_to_seq(st.session_state["beam_table"])
    _n = len(_seq)
    _k = max(0, min(int(st.session_state["view_k"]), _n))  # clamp (table may have shrunk)
    st.session_state["view_k"] = _k
    _view_image = _degraded_image(
        _seq[:_k], float(st.session_state["live_I0"]), float(st.session_state["live_alpha"]),
        float(st.session_state["live_beta"]), IMAGE_RES,
    )
    # Two overlays: red = live next-measurement preview (sliders); blue = the viewed measurement.
    _preview = (
        float(st.session_state["live_angle"]), float(st.session_state["live_offset"]),
        int(st.session_state["live_nbeams"]),
    )
    _committed = _seq[_k - 1] if _k >= 1 else None

    with left:
        nav = st.columns(2)
        nav[0].button("⬅ Previous Measurement", key="btn_prev_view_k",
                      on_click=_cb_view_prev, args=("view_k",),
                      use_container_width=True, disabled=(_k == 0))
        nav[1].button("Next Measurement ➡", key="btn_next_view_k",
                      on_click=_cb_view_next, args=("view_k", _n),
                      use_container_width=True, disabled=(_k >= _n))
        st.caption(f"Viewing measurement **{_k}** of **{_n}**.")
        # Interactive browser preview: the Angle/Offset/#Beams sliders live here and the red preview
        # dashes redraw *while* dragging (client-side). Python only supplies the static background image
        # and the committed (blue) bundle. On release the component returns the values so the rest of the
        # app (Take measurement / Reconstruct) reads them from session_state below.
        _live_sim(
            image_uri=_live_background_uri(_view_image, _vmin, _vmax),
            image_res=IMAGE_RES,
            k=_k,
            angle=float(st.session_state["live_angle"]),
            offset=float(st.session_state["live_offset"]),
            nbeams=int(st.session_state["live_nbeams"]),
            committed=(list(_committed) if _committed is not None else None),
            beams_visible=bool(st.session_state["beams_visible"]),
            default={
                "angle": float(st.session_state["live_angle"]),
                "offset": float(st.session_state["live_offset"]),
                "nbeams": int(st.session_state["live_nbeams"]),
            },
            key="live_sim",
            on_change=_cb_sync_live_sim_2d,
        )
        # Return value deliberately unused: it is the component's STANDING widget value and
        # persists across reruns, so writing it back each run would resurrect the last value the
        # component reported and overwrite anything set elsewhere. The 3D tab has a second control
        # surface where that is an outright bug; the rule is the same here so the two instances
        # cannot diverge. _cb_sync_live_sim_2d owns the sync.
        st.latex(
            r"\mathrm{pixel\_new} = \mathrm{pixel}\cdot"
            r"\exp\!\left(-\alpha\,I_{\mathrm{local}} - \beta\,I_{\mathrm{local}}^2\right)"
        )

    with mid:
        st.subheader("Live simulator")
        st.caption("Set the projection with the **Angle / Offset / # Beams** sliders under the image "
                   "(they update the red preview live), then **➕ Take measurement** to add it to the "
                   "sequence.")
        cc = st.columns(3)
        cc[0].number_input("I0", min_value=0.0, step=0.5, key="live_I0",
                    help="Beam intensity (≥ 0). At 0, there is no dose degradation (the image is not "
                                "darkened by measurements).")
        cc[1].number_input("alpha", step=0.05, format="%.3f", key="live_alpha")
        cc[2].number_input("beta", step=0.01, format="%.4f", key="live_beta")

        b = st.columns(3)
        b[0].button("➕ Take measurement", on_click=_cb_step, use_container_width=True,
                    help="Apply a measurement at the current angle/offset/#beams and append it to "
                         "the sequence table.")
        b[1].button("Reset", on_click=_cb_reset, use_container_width=True,
                    help="Clear the sequence and return to the clean phantom.")
        b[2].button("Toggle beams", on_click=_cb_toggle, use_container_width=True)

        st.subheader("Solver Tuning")
        st.slider("TV regularization weight", 0.0, 1.0, step=0.01, key="live_tv_weight",
                  help="Total-variation penalty in the reconstruction objective (higher = smoother; "
                       "0 disables it). Used only by Reconstruct. Very low values with few or clustered "
                       "angles can make the UQ/sensitivity step fail (singular system). Raise this or "
                       "add more evenly-spaced angles if Reconstruct reports a UQ failure.")

        reconstruct_clicked = st.button("Reconstruct", type="primary",
                                        use_container_width=True)
        st.caption("⏱️ Reconstruct solves exactly the table's sequence (takes a few minutes). It uses "
                   "the live I0/α/β; set I0=0 to reconstruct without modeling dose degradation.")

    with right:
        _preset_block("live", "beam_table", "view_k")

        st.subheader("Measurement sequence")
        st.caption(
            "Each measurement you take is recorded here: angle (°), offset (bundle center), #beams "
            "(0 = full fan). The **highlighted** row is the one currently shown on the left."
        )
        # Read-only view (st.dataframe, not st.data_editor) so the sequence can't be edited by an
        # accidental click — it is driven solely by the Take measurement / Reset buttons. The Styler
        # highlights the currently-viewed measurement (Previous/Next).
        st.dataframe(
            _style_sequence(st.session_state["beam_table"], _k),
            use_container_width=True,
            column_config={
                "angle_deg": st.column_config.NumberColumn("Angle °", format="%.2f"),
                "offset": st.column_config.NumberColumn("Offset", format="%.2f"),
                "n_beams": st.column_config.NumberColumn("# Beams", format="%d"),
            },
        )

    st.divider()

    # Fixed placeholders so a fresh Reconstruct clears the previous output up-front: reaching these
    # empty() slots on the rerun removes the old figures *before* the (minutes-long) solve, so they
    # vanish while it runs instead of lingering underneath. Results are filled back in below.
    _log_slot = st.empty()
    _results_slot = st.empty()

    # --- heavy solve: only on Reconstruct press --------------------------------------------
    if reconstruct_clicked:
        # Same table → sequence conversion the live image uses, so the solve matches the picture.
        steps = [BeamStep(angle_deg=a, offset=o, n_beams=n)
                 for (a, o, n) in _table_to_seq(st.session_state["beam_table"])]

        if not steps:
            st.session_state.pop("results", None)
            _results_slot.error("Take at least one measurement (or add a table row) before "
                                "reconstructing.")
            st.stop()

        # Cost guard: sensitivity parameters span the full (unique r × unique angle × time) product,
        # so fractional offsets that don't reuse the detector grid inflate cost quadratically.
        _rmax = IMAGE_RES / 2 - 0.5 + 1e-9
        uniq_r, uniq_ang, total_rays = set(), set(), 0
        for s in steps:
            nb = s.n_beams if s.n_beams > 0 else IMAGE_RES
            rv = [float(r) for r in s.offset + (np.arange(nb) - (nb - 1) / 2.0) if abs(r) <= _rmax]
            total_rays += len(rv)
            uniq_r.update(round(r, 6) for r in rv)
            uniq_ang.add(round(s.angle_deg, 6))
        param_cols = len(uniq_r) * len(uniq_ang) * (len(steps) + 1)

        # noise_cov_scale / ipopt_max_iter / linear_solver use the UQParams defaults.
        params = UQParams(
            image_res=IMAGE_RES,
            I0=float(st.session_state["live_I0"]),
            alpha=float(st.session_state["live_alpha"]),
            beta=float(st.session_state["live_beta"]),
            tv_weight=float(st.session_state["live_tv_weight"]),
            beam_steps=steps,
        )
        _ctx = _current_script_ctx()

        with _log_slot.container():
            if param_cols > 5000 or total_rays > 2000:
                st.warning(
                    f"⚠️ ~{param_cols:,} sensitivity parameter columns / {total_rays:,} rays. The "
                    "sensitivity + covariance step may take many minutes or run out of memory. "
                    "Tip: integer or 0.5-grid offsets reuse the detector grid and stay cheaper."
                )
            st.subheader("Solver log (inverse solve)")
            # Fixed-height scrolling box showing the newest lines (see _render_log_box).
            log_box = st.empty()
            log_lines: list[str] = []

            def _render_log() -> None:
                # Rolling tail (escaped) so very long solver logs stay responsive in the browser.
                _render_log_box(log_box, "".join(log_lines))

            def log_callback(chunk: str) -> None:
                # Called from Pyomo's reader thread: buffer and echo only. Rendering Streamlit
                # elements here causes cross-thread layout flicker while the script is blocked.
                log_lines.append(chunk)
                _term_echo(chunk)

            with st.spinner("Solving forward + inverse problem and extracting sensitivity..."):
                try:
                    results = run_simple_uq(
                        params, log_callback=log_callback,
                        cancel_callback=lambda: _script_stop_requested(_ctx))
                    st.session_state["results"] = results
                except SolverCancelled:
                    st.stop()
                except RuntimeError as exc:  # curated, user-facing guidance (e.g. singular-KKT UQ failure)
                    st.session_state.pop("results", None)
                    st.error(str(exc))  # message is already actionable; skip the scary chained traceback
                except Exception as exc:  # unexpected bug: surface the full traceback
                    st.session_state.pop("results", None)
                    st.error(f"Run failed: {exc}")
                    st.exception(exc)
                finally:
                    _render_log()  # final flush: last lines always shown and pinned to the bottom

    # Render current results into the fixed slot (new ones after a solve; persisted on a plain rerun).
    _render_results(_results_slot, st.session_state.get("results"))
