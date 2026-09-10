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

2. **Reconstruct** (button) — runs :func:`tomography_uq.run_simple_uq` (a forward + inverse
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
import html as _html
import io
import os
import time

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

from tomography_uq import UQParams, BeamStep, run_simple_uq

import matplotlib.pyplot as plt  # after tomography_uq sets the Agg backend
import matplotlib.image as mpimg

# Vendored geometry primitives (importing them is not a backend change).
from senDOE.helpers.geometry import (
    get_line_abc_from_r_theta,
    line_grid_intersections,
    get_segment_polar,
)

# Dose-response physics lives in a Streamlit-free module so the 3D simulator can import it too
# (importing app.py would execute this whole page). Bound to the old private names so the rest
# of this file is unchanged.
from dose_response import (
    bundle_r_values as _bundle_r_values,
    degradation_dose_response as _degradation_dose_response,
)
from tomography_3d import shepp_logan_3d, simulate_3d, detector_grid
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

st.set_page_config(page_title="Tomography Sensitivity UQ", layout="wide")

st.title("Tomographic reconstruction + sensitivity-based UQ")
st.caption(
    "Take X-ray measurements with the live simulator (each is logged to the sequence table on "
    "the right) → **Reconstruct** solves that exact sequence — a forward solve simulates the "
    "measurements, an inverse solve recovers the image, and a sensitivity analysis gives the "
    "per-pixel posterior covariance (uncertainty) and a scalar D-optimality information score."
)

# Fixed sim/reconstruction resolution (was the sidebar slider; equals the UQParams default).
IMAGE_RES = 30

# Live-simulator preview is a custom browser component (no-build static component): it holds the
# Angle/Offset/#Beams sliders + the beam overlay and redraws the red preview lines *while* dragging,
# client-side, so there is no server round-trip per drag (st.slider only reports on release). The
# component reports the values back on release; Python renders only the static background image and
# the committed (blue) bundle. See live_sim_component/index.html.
_LIVE_SIM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_sim_component")
_live_sim = components.declare_component("live_sim", path=_LIVE_SIM_DIR)


def _live_background_uri(img, vmin, vmax) -> str:
    """PNG data-URI of the grayscale phantom view (no axes/lines) for the live component background.

    ``mpimg.imsave`` writes exactly the array's pixels (here image_res×image_res), so the data rect
    maps 1:1 to the component's image box and the JS overlay aligns to it. The component upscales it
    with ``image-rendering: pixelated`` to match the backend's ``interpolation="nearest"``.
    """
    buf = io.BytesIO()
    mpimg.imsave(buf, np.asarray(img), cmap="gray", vmin=vmin, vmax=vmax, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

# Solver-log box: a scrollable monospace div force-scrolled to the bottom on every update. Rendered
# via components.html (sandboxed iframe) so the trailing <script> actually runs — st.html strips
# scripts, so it cannot auto-follow. ``{body}`` is the HTML-escaped rolling log tail.
_LOG_IFRAME = (
    '<div id="lb" style="height:300px;overflow-y:auto;white-space:pre-wrap;'
    'font-family:monospace;font-size:12px;line-height:1.3;background:#0e1117;'
    'color:#d6d6d6;padding:8px;border-radius:6px;">{body}</div>'
    "<script>var b=document.getElementById('lb');b.scrollTop=b.scrollHeight;</script>"
)


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
    axes[0].set_title("Posterior covariance — per-pixel uncertainty")
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
        row1[1].caption("Beam / measurement view — the chosen projection geometry over the phantom")
        row2 = st.columns([1.0, _CBAR_COL_RATIO])
        row2[0].pyplot(results.fig_nlp, use_container_width=True, bbox_inches=None)
        row2[0].caption("Reconstruction (NLP)")
        row2[1].pyplot(results.fig_covariance, use_container_width=True, bbox_inches=None)
        row2[1].caption("Posterior covariance — per-pixel uncertainty")


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


def _cb_prev():
    """Step the view back one measurement."""
    st.session_state["view_k"] = max(0, int(st.session_state["view_k"]) - 1)


def _cb_next():
    """Step the view forward one measurement (bounded by the sequence length)."""
    n = len(st.session_state["beam_table"])
    st.session_state["view_k"] = min(n, int(st.session_state["view_k"]) + 1)


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
}.items():
    st.session_state.setdefault(_k, _v)

# 3D tab keeps its own namespace so the two tabs never clobber each other's controls.
if "beam_table_3d" not in st.session_state:
    st.session_state["beam_table_3d"] = _empty_beam_table()
for _k, _v in {
    "live3d_angle": 45.0, "live3d_offset": 0.0, "live3d_nbeams": 30, "live3d_z": 0,
    "live3d_nslices": 16, "live3d_variant": "modified",
    "live3d_I0": 0.0, "live3d_alpha": 0.3, "live3d_beta": 0.01,
    "live3d_meas": 0, "live3d_opacity": 0.15, "live3d_isomin": 0.05,
    "live3d_volsrc": "Degraded", "live3d_sinoview": "Per-slice sinogram",
}.items():
    st.session_state.setdefault(_k, _v)


# --- 3D degradation tab (2.5D: same bundle through every slice; no reconstruction) ------
# Own session_state namespace (live3d_*, beam_table_3d) so the two tabs never clobber each
# other. The measurement table schema is identical to the 2D one, so _empty_beam_table /
# _table_to_seq are reused as-is.

_N_SLICES_MIN, _N_SLICES_MAX = 4, 48


@st.cache_data(show_spinner=False)
def _simulate_3d(seq: tuple, I0: float, alpha: float, beta: float,
                 image_res: int, n_slices: int, variant: str):
    """``(original, degraded, sinogram)`` for the sequence (cached like the 2D image).

    Pure function of the table + dose params + volume size, so scrubbing z, switching sub-tab,
    rotating the volume and changing the measurement index are all cache hits; only taking a
    measurement or changing a parameter recomputes.
    """
    vol0 = shepp_logan_3d(image_res, n_slices, variant=variant)
    vol1, sino = simulate_3d(vol0, seq, I0, alpha, beta, image_res)
    return vol0, vol1, sino


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


def _cb3d_reset():
    """Clear the 3D sequence — back to the clean volume."""
    st.session_state["beam_table_3d"] = _empty_beam_table()


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
    :func:`tomography_3d.simulate_3d`, so unmeasured cells are ``NaN``. A hand-built sequence
    samples only a few of the ``image_res`` detector slots, so most of the panel is genuinely
    unmeasured — those cells are drawn in a flat off-colour via ``set_bad`` so they read as
    "no data" rather than as a real low line-integral.
    """
    r_grid = detector_grid(image_res)
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#2b2b3a")  # unmeasured — deliberately not part of the viridis ramp
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


def _volume_figure(vol, opacity: float, isomin: float, title: str, colorscale: str):
    """Translucent 3D voxel rendering — the "look inside" view.

    Plotly composites the whole volume with per-voxel alpha, so interior structure shows
    through the skull instead of being hidden by it, and the user can drag to rotate. ``caps``
    are off so the volume reads as open rather than shrink-wrapped in solid end faces, and
    ``isomin`` clips the air around the head (which would otherwise fog the view).

    Axes follow the array: x = column, y = row, z = slice.
    """
    nr, nc, nz = vol.shape
    yy, xx, zz = np.meshgrid(np.arange(nr), np.arange(nc), np.arange(nz), indexing="ij")
    fig = go.Figure(
        data=go.Volume(
            x=xx.ravel(), y=yy.ravel(), z=zz.ravel(), value=vol.ravel(),
            isomin=float(isomin), isomax=float(max(vol.max(), isomin + 1e-6)),
            opacity=float(opacity),
            surface=dict(count=17),
            colorscale=colorscale,
            caps=dict(x_show=False, y_show=False, z_show=False),
            colorbar=dict(title="Intensity", thickness=14),
        )
    )
    fig.update_layout(
        title=title,
        height=620,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            xaxis_title="x (col)", yaxis_title="y (row)", zaxis_title="z (slice)",
            # Equal data scaling in x/y; z is exaggerated when there are few slices so a thin
            # stack is still legible rather than a pancake.
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=max(0.35, min(1.0, nz / max(nr, 1)))),
        ),
    )
    return fig


def _render_3d_tab():
    """The 3D degradation simulator: build a sequence, scrub slices, watch the dose land."""
    s = st.session_state
    st.caption(
        "**2.5D forward simulation.** The phantom is a true 3D Shepp-Logan volume, but each "
        "measurement fires the *same* beam bundle through every slice — rays never cross "
        "slices. Dose still varies with depth because each slice attenuates the beam "
        "differently. Reconstruction is not part of this tab."
    )
    left3d, mid3d, right3d = st.columns([3, 2, 2])

    seq3d = _table_to_seq(s["beam_table_3d"])
    n_meas = len(seq3d)
    n_slices = int(s["live3d_nslices"])

    vol0, vol, sino = _simulate_3d(
        seq3d, float(s["live3d_I0"]), float(s["live3d_alpha"]), float(s["live3d_beta"]),
        IMAGE_RES, n_slices, s["live3d_variant"],
    )
    # Clamp the viewed slice / measurement: either may have shrunk since it was set.
    k = max(0, min(int(s["live3d_z"]), n_slices - 1))
    s["live3d_z"] = k
    if n_meas:
        s["live3d_meas"] = max(0, min(int(s["live3d_meas"]), n_meas - 1))

    with left3d:
        view_slice, view_vol, view_sino = st.tabs(["Slice", "Volume", "Sinogram"])

        with view_slice:
            st.pyplot(
                _slice_figure(
                    vol[:, :, k], IMAGE_RES, k, n_slices, n_meas, 0.0, 1.0,
                    preview=(s["live3d_angle"], s["live3d_offset"], s["live3d_nbeams"]),
                ),
                use_container_width=True,
            )

        with view_vol:
            src = s["live3d_volsrc"]
            if src == "Original":
                data, cmap_name = vol0, "Gray"
                title = "Phantom (no dose applied)"
            elif src == "Dose removed":
                data, cmap_name = vol0 - vol, "Inferno"
                title = "Dose removed (original − degraded) — where the beams landed"
            else:
                data, cmap_name = vol, "Gray"
                title = "Degraded volume · %d measurement%s" % (
                    n_meas, "" if n_meas == 1 else "s")
            st.plotly_chart(
                _volume_figure(data, s["live3d_opacity"], s["live3d_isomin"], title, cmap_name),
                use_container_width=True,
            )
            vc1, vc2, vc3 = st.columns(3)
            with vc1:
                st.radio("Show", ("Degraded", "Original", "Dose removed"),
                         key="live3d_volsrc",
                         help="'Dose removed' is the clearest view of where the beams "
                              "deposited energy.")
            with vc2:
                st.slider("Opacity", 0.02, 1.0, step=0.01, key="live3d_opacity",
                          help="Lower = more see-through. This is the 'look inside' knob.")
            with vc3:
                st.slider("Hide below", 0.0, 0.9, step=0.01, key="live3d_isomin",
                          help="Clip low values (the air around the head) so they do not "
                               "fog the view.")
            st.caption("Drag to rotate · scroll to zoom · double-click to reset.")

        with view_sino:
            st.radio("View", ("Per-slice sinogram", "Per-measurement projection"),
                     key="live3d_sinoview", horizontal=True,
                     help="A sinogram is one slice's readings across every measurement. A "
                          "projection is one measurement's readings across every slice — what "
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
                    "Every slice's reading for this one measurement, stacked — the image a 2D "
                    "detector panel behind the volume would record."
                )

    with mid3d:
        st.markdown("**View**")
        st.slider("Slice (z)", 0, max(n_slices - 1, 0), key="live3d_z",
                  help="Drives both the Slice view and the per-slice sinogram.")

        st.markdown("**Next measurement**")
        st.slider("Angle (deg)", 0.0, 360.0, step=1.0, key="live3d_angle")
        st.slider("Offset", -float(IMAGE_RES) / 2, float(IMAGE_RES) / 2, step=0.5,
                  key="live3d_offset")
        st.slider("# Beams (0 = full fan)", 0, IMAGE_RES, step=1, key="live3d_nbeams")
        st.button("➕ Take measurement", key="btn3d_step", on_click=_cb3d_step,
                  use_container_width=True)
        st.button("Reset", key="btn3d_reset", on_click=_cb3d_reset,
                  use_container_width=True)

        st.markdown("**Volume & dose**")
        st.slider("Slices (z)", _N_SLICES_MIN, _N_SLICES_MAX, step=1, key="live3d_nslices")
        st.selectbox("Phantom contrast", ("modified", "classic"), key="live3d_variant",
                     help="'classic' is the textbook Kak & Slaney table; its interior "
                          "features sit ~2% above background and wash out under dose.")
        st.number_input("I0 (0 = no degradation)", min_value=0.0, step=0.5,
                        key="live3d_I0")
        st.number_input("alpha", min_value=0.0, step=0.05, format="%.3f", key="live3d_alpha")
        st.number_input("beta", min_value=0.0, step=0.005, format="%.3f", key="live3d_beta")

    with right3d:
        st.markdown("**Measurement sequence**")
        st.dataframe(s["beam_table_3d"], use_container_width=True, hide_index=False)
        total0 = float(vol0.sum())
        total1 = float(vol.sum())
        lost = 0.0 if total0 == 0 else 100.0 * (1.0 - total1 / total0)
        st.metric("Intensity removed", "%.1f%%" % lost,
                  help="Total volume intensity lost to dose across all slices.")
        st.caption("Slice z=%d mean: %.4f" % (k, float(vol[:, :, k].mean())))


# --- two modes, two tabs --------------------------------------------------------------
tab_2d, tab_3d = st.tabs(["2D dose-response + reconstruction", "3D degradation"])

# The 3D tab is populated FIRST in script order: the 2D body below ends in a Reconstruct
# path that can call st.stop() (empty geometry), which would otherwise leave this tab blank
# on that rerun. Display order is set by the st.tabs() list above, not by population order.
with tab_3d:
    _render_3d_tab()

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
        nav[0].button("⬅ Previous Measurement", on_click=_cb_prev,
                      use_container_width=True, disabled=(_k == 0))
        nav[1].button("Next Measurement ➡", on_click=_cb_next,
                      use_container_width=True, disabled=(_k >= _n))
        st.caption(f"Viewing measurement **{_k}** of **{_n}**.")
        # Interactive browser preview: the Angle/Offset/#Beams sliders live here and the red preview
        # dashes redraw *while* dragging (client-side). Python only supplies the static background image
        # and the committed (blue) bundle. On release the component returns the values so the rest of the
        # app (Take measurement / Reconstruct) reads them from session_state below.
        _live_val = _live_sim(
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
        )
        if isinstance(_live_val, dict):  # released slider values → sync so Take/Reconstruct use them
            st.session_state["live_angle"] = float(_live_val["angle"])
            st.session_state["live_offset"] = float(_live_val["offset"])
            st.session_state["live_nbeams"] = int(_live_val["nbeams"])
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
                           help="Beam intensity (≥ 0). 0 → no dose degradation (the image is not "
                                "darkened by measurements).")
        cc[1].number_input("alpha", step=0.05, format="%.3f", key="live_alpha")
        cc[2].number_input("beta", step=0.01, format="%.4f", key="live_beta")

        b = st.columns(3)
        b[0].button("➕ Take measurement", on_click=_cb_step, use_container_width=True,
                    help="Apply a measurement at the current angle/offset/#beams and append it to "
                         "the sequence table.")
        b[1].button("Reset", on_click=_cb_reset, use_container_width=True,
                    help="Clear the sequence — back to the clean phantom.")
        b[2].button("Toggle beams", on_click=_cb_toggle, use_container_width=True)

        st.subheader("Solver Tuning")
        st.slider("TV regularization weight", 0.0, 1.0, step=0.01, key="live_tv_weight",
                  help="Total-variation penalty in the reconstruction objective (higher = smoother; "
                       "0 disables it). Used only by Reconstruct. Very low values with few or clustered "
                       "angles can make the UQ/sensitivity step fail (singular system) — raise this or "
                       "add more evenly-spaced angles if Reconstruct reports a UQ failure.")

        reconstruct_clicked = st.button("Reconstruct", type="primary",
                                        use_container_width=True)
        st.caption("⏱️ Reconstruct solves exactly the table's sequence (takes a few minutes). It uses "
                   "the live I0/α/β; set I0=0 to reconstruct without modeling dose degradation.")

    with right:
        st.subheader("Measurement sequence")
        st.caption(
            "Each measurement you take is recorded here — angle (°), offset (bundle center), #beams "
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

        with _log_slot.container():
            if param_cols > 5000 or total_rays > 2000:
                st.warning(
                    f"⚠️ ~{param_cols:,} sensitivity parameter columns / {total_rays:,} rays — the "
                    "sensitivity + covariance step may take many minutes or run out of memory. "
                    "Tip: integer or 0.5-grid offsets reuse the detector grid and stay cheaper."
                )
            st.subheader("Solver log (inverse solve)")
            # Fixed-height box that auto-follows the newest line (see _LOG_IFRAME).
            log_box = st.empty()
            log_lines: list[str] = []
            _last_render = [0.0]

            def _render_log() -> None:
                # Rolling tail (escaped) so very long solver logs stay responsive in the browser.
                body = _html.escape("".join(log_lines)[-8000:])
                log_box.empty()  # drop the prior iframe so they don't stack
                with log_box.container():
                    components.html(_LOG_IFRAME.format(body=body), height=312, scrolling=False)

            def log_callback(chunk: str) -> None:
                log_lines.append(chunk)
                now = time.time()
                if now - _last_render[0] >= 0.2:  # throttle so the iframe rebuild doesn't flicker
                    _last_render[0] = now
                    _render_log()

            with st.spinner("Solving forward + inverse problem and extracting sensitivity…"):
                try:
                    results = run_simple_uq(params, log_callback=log_callback)
                    st.session_state["results"] = results
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
