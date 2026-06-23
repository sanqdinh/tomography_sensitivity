"""Streamlit UI for the tomographic uncertainty-quantification pipeline.

Two modes share one page:

1. **Live dose-response simulator** (main area) — an Example10-style interactive playground.
   A large central grayscale image starts as the Shepp-Logan phantom and progressively
   *degrades* as you apply X-ray measurements (``pixel·exp(-α·I_local - β·I_local²)``).
   Sliders set the projection angle, the radial offset of the ray bundle, and the number of
   beams; a red dashed overlay shows where the rays fall. **Step**/**Apply** degrade the image,
   **Reset** restores the phantom. This runs entirely in the browser process — no solver needed.

2. **Reconstruct** (button) — runs :func:`tomography_uq.run_simple_uq` (two IPOPT solves + a
   k_aug sensitivity extraction, minutes at the default size). It uses the multi-step geometry
   from the sidebar table and the live I0/α/β values, then shows the reconstruction prominently
   plus the full UQ suite (posterior covariance, D-optimality, beam view, streaming solver log).

Streamlit reruns the whole script on every widget interaction, so the degraded image and the
solve results are stashed in ``st.session_state`` to survive reruns. The heavy solve runs ONLY
on the Reconstruct button press. This file is the only entrypoint and changes nothing in the
backend: it imports the vendored geometry primitives and re-implements one small numpy helper
(the dose-response degradation) for the live preview.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import streamlit as st

from tomography_uq import UQParams, BeamStep, run_simple_uq

import matplotlib.pyplot as plt  # after tomography_uq sets the Agg backend

# Vendored geometry primitives (importing them is not a backend change).
from senDOE.helpers.geometry import (
    get_line_abc_from_r_theta,
    line_grid_intersections,
    get_segment_polar,
)
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

st.set_page_config(page_title="Tomography Sensitivity UQ", layout="wide")

st.title("Tomographic reconstruction + sensitivity-based UQ")
st.caption(
    "Tune the live dose-response simulator below (angle · offset · #beams, I0/α/β) → add steps "
    "to the geometry table → **Reconstruct** runs the forward + inverse IPOPT solve and k_aug "
    "sensitivity for the posterior covariance & D-optimality."
)


def _default_beam_table() -> pd.DataFrame:
    """9 evenly-spaced angles over [0, 180), full fan each — reproduces the old default."""
    angles = np.linspace(0.0, 180.0, 9, endpoint=False)
    return pd.DataFrame(
        {
            "angle_deg": [float(a) for a in angles],
            "offset": [0.0] * len(angles),
            "n_beams": [0] * len(angles),  # 0 => full image-width fan
        }
    )


def _geometry_preview_fig(df: pd.DataFrame, image_res: int):
    """Render a quick 'where do the beams point' dial from the current table (no solve)."""
    half = image_res / 2.0
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.add_patch(plt.Circle((0, 0), half, fill=False, color="0.6", lw=1.0))
    rows = df.dropna(subset=["angle_deg"])
    cmap = plt.get_cmap("viridis", max(len(rows), 1))
    for i, (_, row) in enumerate(rows.iterrows()):
        angle = float(row["angle_deg"])
        offset = 0.0 if pd.isna(row.get("offset")) else float(row["offset"])
        nb = row.get("n_beams")
        nb = image_res if (pd.isna(nb) or int(nb) <= 0) else int(nb)
        th = np.deg2rad(angle)
        ct, sn = np.cos(th), np.sin(th)
        dx, dy = -sn, ct  # direction along each ray
        for k, rr in enumerate((offset - (nb - 1) / 2.0, offset, offset + (nb - 1) / 2.0)):
            px, py = rr * ct, rr * sn
            x0, y0 = px - image_res * dx, py - image_res * dy
            x1, y1 = px + image_res * dx, py + image_res * dy
            ax.plot(
                [x0, x1], [y0, y1],
                lw=1.3 if k == 1 else 0.5,
                ls="-" if k == 1 else "--",
                color=cmap(i), alpha=0.85,
            )
    lim = half * 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Geometry preview", fontsize=9)
    return fig


# --- live simulator helpers (pure frontend; reuse vendored geometry) -------------------

@st.cache_data(show_spinner=False)
def _phantom(image_res: int) -> np.ndarray:
    """The same phantom the backend builds: ``resize(shepp_logan_phantom(), (N, N))``."""
    ph = shepp_logan_phantom()
    ph = resize(ph, (int(image_res), int(image_res)))
    return ph.astype(float)


def _degradation_dose_response(image, r, theta, I0, alpha, beta):
    """numpy port of ``util.HelperTools.degradation_Dose_Response`` (numpy branch).

    Degrades the image along the ray ``x·cosθ + y·sinθ = r`` using the dose-response model
    ``pixel·exp(-α·I_local - β·I_local²)`` with ``I_local = I0·exp(-Σ radon)``. Reuses the
    vendored ``get_line_abc_from_r_theta`` / ``line_grid_intersections``. Returns the image
    unchanged if the ray misses the grid (the vendored intersection routine raises IndexError
    on an empty hit, mirrored by the backend's |r| clamp).
    """
    h, w = image.shape
    a, b, c = get_line_abc_from_r_theta(r, theta)
    try:
        _, image_intersection, radon, _ = line_grid_intersections(
            a, b, c, image, x_range=[-w / 2, w / 2], y_range=[-h / 2, h / 2]
        )
    except IndexError:
        return image  # line never enters the grid → no-op
    if len(image_intersection) == 0:
        return image
    out = image.copy()
    for i in range(len(image_intersection)):
        ix = int(image_intersection[i, 0])  # row
        iy = int(image_intersection[i, 1])  # col
        local = I0 * np.exp(-sum(radon[j] for j in range(i)))
        out[ix, iy] = image[ix, iy] * np.exp(-alpha * local - beta * local**2)
    return out


def _bundle_r_values(offset: float, n_beams: int, image_res: int) -> list:
    """Radial positions of a ray bundle (BeamStep convention: n rays, 1 unit apart, centered).

    ``n_beams == 0`` ⇒ full ``image_res`` fan. Rays outside the grid are dropped with the same
    ``|r| <= image_res/2 - 0.5`` clamp the backend uses (guards the vendored empty-intersection
    IndexError).
    """
    n = int(n_beams) if int(n_beams) > 0 else int(image_res)
    r_max = image_res / 2 - 0.5 + 1e-9
    rs = offset + (np.arange(n) - (n - 1) / 2.0)
    return [float(r) for r in rs if abs(r) <= r_max]


def _live_figure(img, image_res, angle_deg, offset, n_beams, beams_visible,
                 measurements_done, vmin, vmax):
    """Large central grayscale image + colorbar + dynamic title + optional red beam overlay."""
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
        for r in _bundle_r_values(offset, n_beams, image_res):
            seg = get_segment_polar(
                r_distance=r, angle=np.deg2rad(angle_deg),
                seg_range=seg_range, num_points=seg_n,
            )
            ax.plot(seg[:, 0], seg[:, 1], "r--", linewidth=0.8)
    return fig


def _seed_live_state(image_res: int) -> None:
    """(Re)build the live degraded image as the phantom at this resolution; reset the counter."""
    st.session_state["current_image"] = _phantom(int(image_res)).copy()
    st.session_state["measurements_done"] = 0
    st.session_state["live_res"] = int(image_res)


# --- live simulator button callbacks (fire before the rerun body; read live_* keys) ----

def _degrade_once(img):
    """One measurement = degrade along every (clamped) ray in the current bundle."""
    s = st.session_state
    rs = _bundle_r_values(s["live_offset"], s["live_nbeams"], s["live_res"])
    theta = np.deg2rad(s["live_angle"])
    out = img
    for r in rs:
        out = _degradation_dose_response(
            out, r, theta, s["live_I0"], s["live_alpha"], s["live_beta"]
        )
    return out


def _cb_step():
    s = st.session_state
    s["current_image"] = _degrade_once(s["current_image"])
    s["measurements_done"] += 1


def _cb_reset():
    _seed_live_state(st.session_state["live_res"])


def _cb_apply():
    """Reach the Measurements-slider target (Example10 ``apply_n_iterations``)."""
    s = st.session_state
    target = int(s["live_measurements"])
    if target < s["measurements_done"]:  # going down ⇒ reset, then climb
        _cb_reset()
    while s["measurements_done"] < target:
        _cb_step()


def _cb_toggle():
    st.session_state["beams_visible"] = not st.session_state["beams_visible"]


def _cb_add_as_step():
    """Append the current live slider config to the multi-step geometry table."""
    s = st.session_state
    new_row = pd.DataFrame(
        [{
            "angle_deg": float(s["live_angle"]),
            "offset": float(s["live_offset"]),
            "n_beams": int(s["live_nbeams"]),
        }]
    )
    s["beam_table"] = pd.concat([s["beam_table"], new_row], ignore_index=True)


# --- session state seeding -------------------------------------------------------------
if "beam_table" not in st.session_state:
    st.session_state["beam_table"] = _default_beam_table()
st.session_state.setdefault("beams_visible", True)
# Live control values live in session_state so the central figure (rendered before the
# widgets exist in script order) can read them, and callbacks have a single source of truth.
for _k, _v in {
    "live_angle": 45.0, "live_offset": 0.0, "live_nbeams": 20,
    "live_I0": 2.0, "live_alpha": 0.3, "live_beta": 0.01, "live_measurements": 0,
}.items():
    st.session_state.setdefault(_k, _v)


# --- sidebar: solver/UQ params + multi-step geometry table -----------------------------
with st.sidebar:
    st.header("Run parameters")
    image_res = st.slider(
        "Image resolution (N×N)", min_value=16, max_value=48, value=30, step=2,
        help="Pixels per side. Cost grows ~ N² · (#steps); the default (30) matches Example2.",
    )

    st.divider()
    st.subheader("Projection geometry (Reconstruct steps)")
    st.caption(
        "One row per projection **step**: angle (°), offset (radial center of the ray "
        "bundle), and #beams (0 = full image-width fan). Use **➕ Add as step** below to push "
        "the live simulator's current bundle here."
    )

    with st.expander("Seed evenly-spaced angles", expanded=False):
        c1, c2 = st.columns(2)
        seed_count = c1.number_input("Count", min_value=1, max_value=64, value=9, step=1)
        seed_beams = c2.number_input("#Beams (0=full)", min_value=0, value=0, step=1)
        c3, c4 = st.columns(2)
        seed_start = c3.number_input("Start °", value=0.0, step=10.0)
        seed_stop = c4.number_input("Stop °", value=180.0, step=10.0)
        if st.button("Seed table", use_container_width=True):
            angs = np.linspace(
                float(seed_start), float(seed_stop), int(seed_count), endpoint=False
            )
            st.session_state["beam_table"] = pd.DataFrame(
                {
                    "angle_deg": [float(a) for a in angs],
                    "offset": [0.0] * len(angs),
                    "n_beams": [int(seed_beams)] * len(angs),
                }
            )

    edited = st.data_editor(
        st.session_state["beam_table"],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "angle_deg": st.column_config.NumberColumn("Angle °", step=1.0, format="%.2f"),
            "offset": st.column_config.NumberColumn("Offset", step=0.5, format="%.2f"),
            "n_beams": st.column_config.NumberColumn(
                "# Beams", min_value=0, step=1, format="%d",
                help="0 = full image-width fan",
            ),
        },
    )
    st.session_state["beam_table"] = edited

    _prev = _geometry_preview_fig(edited, int(image_res))
    st.pyplot(_prev, use_container_width=True)
    plt.close(_prev)  # avoid matplotlib figure leakage across reruns

    st.divider()
    st.subheader("Solver / UQ params")
    tv_weight = st.number_input("TV regularization weight", value=0.1, step=0.05,
                                format="%.4f")
    noise_scale = st.number_input("Measurement noise covariance (× I)", value=10.0,
                                  step=1.0)
    max_iter = st.number_input("IPOPT max_iter", value=1000, step=100, min_value=1)
    linear_solver = st.selectbox("Linear solver", ["ma27", "ma57", "mumps"],
                                 index=0,
                                 help="Falls back to the next option if one isn't available.")

# Rebuild the live image when the resolution changes (the degraded image is grid-specific).
if (st.session_state.get("live_res") != int(image_res)
        or "current_image" not in st.session_state):
    _seed_live_state(int(image_res))

# --- main: live dose-response simulator ------------------------------------------------
left, right = st.columns([3, 2])

_ph = _phantom(int(image_res))
_vmin, _vmax = float(_ph.min()), float(_ph.max())

with left:
    _live = _live_figure(
        st.session_state["current_image"], int(image_res),
        st.session_state["live_angle"], st.session_state["live_offset"],
        st.session_state["live_nbeams"], st.session_state["beams_visible"],
        st.session_state["measurements_done"], _vmin, _vmax,
    )
    st.pyplot(_live, use_container_width=True)
    plt.close(_live)
    st.latex(
        r"\mathrm{pixel\_new} = \mathrm{pixel}\cdot"
        r"\exp\!\left(-\alpha\,I_{\mathrm{local}} - \beta\,I_{\mathrm{local}}^2\right)"
    )

with right:
    st.subheader("Live simulator")
    st.slider("Angle (deg)", 0.0, 180.0, step=1.0, key="live_angle")
    st.slider("Offset (bundle center)", -24.0, 24.0, step=0.5, key="live_offset",
              help="Radial center of the ray bundle (image units). Rays outside the grid are "
                   "dropped.")
    st.slider("# Beams", 1, 48, step=1, key="live_nbeams",
              help="Rays spaced one image-unit apart, centered at the offset.")
    cc = st.columns(3)
    cc[0].number_input("I0", step=0.5, key="live_I0",
                       help="Beam intensity. 0 → no dose degradation (Example2 default).")
    cc[1].number_input("alpha", step=0.05, format="%.3f", key="live_alpha")
    cc[2].number_input("beta", step=0.01, format="%.4f", key="live_beta")
    st.slider("Measurements (Apply target)", 0, 30, step=1, key="live_measurements")

    b = st.columns(4)
    b[0].button("Apply", on_click=_cb_apply, use_container_width=True,
                help="Degrade up to the Measurements target.")
    b[1].button("Step", on_click=_cb_step, use_container_width=True,
                help="Apply one more measurement.")
    b[2].button("Reset", on_click=_cb_reset, use_container_width=True,
                help="Restore the phantom.")
    b[3].button("Toggle beams", on_click=_cb_toggle, use_container_width=True)

    a = st.columns(2)
    a[0].button("➕ Add as step", on_click=_cb_add_as_step, use_container_width=True,
                help="Append this angle/offset/#beams as a row in the geometry table.")
    reconstruct_clicked = a[1].button("Reconstruct", type="primary",
                                      use_container_width=True)
    st.caption("⏱️ Reconstruct (res 30, 9 steps) takes minutes. It uses the live I0/α/β; set "
               "I0=0 for the Example2-identical reconstruction.")

st.divider()

# --- heavy solve: only on Reconstruct press --------------------------------------------
if reconstruct_clicked:
    # Convert the edited table into beam steps (drop rows with no angle).
    df = st.session_state["beam_table"].dropna(subset=["angle_deg"])
    steps: list[BeamStep] = []
    for _, row in df.iterrows():
        nb = row["n_beams"]
        nb = 0 if pd.isna(nb) else max(int(nb), 0)  # 0 = full fan sentinel
        off = 0.0 if pd.isna(row["offset"]) else float(row["offset"])
        steps.append(BeamStep(angle_deg=float(row["angle_deg"]), offset=off, n_beams=nb))

    if not steps:
        st.error("Add at least one beam step — the geometry table is empty.")
        st.stop()

    # Cost guard: k_aug parameters span the full (unique r × unique angle × time) product,
    # so fractional offsets that don't reuse the detector grid inflate cost quadratically.
    _rmax = int(image_res) / 2 - 0.5 + 1e-9
    uniq_r, uniq_ang, total_rays = set(), set(), 0
    for s in steps:
        nb = s.n_beams if s.n_beams > 0 else int(image_res)
        rv = [float(r) for r in s.offset + (np.arange(nb) - (nb - 1) / 2.0) if abs(r) <= _rmax]
        total_rays += len(rv)
        uniq_r.update(round(r, 6) for r in rv)
        uniq_ang.add(round(s.angle_deg, 6))
    param_cols = len(uniq_r) * len(uniq_ang) * (len(steps) + 1)
    if param_cols > 5000 or total_rays > 2000:
        st.warning(
            f"⚠️ ~{param_cols:,} k_aug parameter columns / {total_rays:,} rays — the "
            "sensitivity + covariance step may take many minutes or run out of memory. "
            "Tip: integer or 0.5-grid offsets reuse the detector grid and stay cheaper."
        )

    params = UQParams(
        image_res=int(image_res),
        I0=float(st.session_state["live_I0"]),
        alpha=float(st.session_state["live_alpha"]),
        beta=float(st.session_state["live_beta"]),
        tv_weight=float(tv_weight),
        noise_cov_scale=float(noise_scale),
        ipopt_max_iter=int(max_iter),
        linear_solver=str(linear_solver),
        beam_steps=steps,
    )

    st.subheader("Solver log")
    log_box = st.empty()
    log_lines: list[str] = []

    def log_callback(chunk: str) -> None:
        log_lines.append(chunk)
        # Show a rolling tail so very long IPOPT logs stay responsive in the browser.
        log_box.code("".join(log_lines)[-8000:], language="text")

    with st.spinner("Solving forward + inverse NLP and extracting k_aug sensitivity…"):
        try:
            results = run_simple_uq(params, log_callback=log_callback)
            st.session_state["results"] = results
        except Exception as exc:  # surface failures instead of a blank page
            st.session_state.pop("results", None)
            st.error(f"Run failed: {exc}")
            st.exception(exc)

# --- render persisted results (survives reruns without re-solving) ---------------------
results = st.session_state.get("results")
if results is not None:
    st.success(
        f"**D-optimality = {results.d_optimality:.4f}**  ·  "
        f"forward: {results.forward_solver_status} ({results.forward_linear_solver})  ·  "
        f"inverse: {results.inverse_solver_status} ({results.inverse_linear_solver})  ·  "
        f"{results.n_free_image0} free pixels  ·  "
        f"{results.n_user_rays} rays / {results.n_sinogram_measurements} k_aug params"
    )

    # Reconstruction is the headline — centered, full width.
    hero = st.columns([1, 2, 1])
    hero[1].pyplot(results.fig_nlp, use_container_width=True)
    hero[1].caption("Reconstruction (NLP)")

    # Full UQ suite.
    row = st.columns(3)
    row[0].pyplot(results.fig_phantom, use_container_width=True)
    row[0].caption("Original phantom")
    row[1].pyplot(results.fig_covariance, use_container_width=True)
    row[1].caption("Posterior covariance (log10 diagonal)")
    row[2].pyplot(results.fig_beams, use_container_width=True)
    row[2].caption("Beam / measurement view — the chosen projection geometry over the phantom")
else:
    st.info("Tune the live simulator and build the geometry table, then click **Reconstruct** "
            "to run the solve.")
