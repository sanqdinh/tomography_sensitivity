"""Dose-response degradation physics, shared by the 2D and 3D simulators.

Extracted verbatim from ``app.py`` so it can be imported by non-Streamlit code: ``app.py`` is a
Streamlit script executed top-to-bottom with no ``__main__`` guard, so ``import app`` would run
the whole page. Keeping the physics here means the 2D live picture and the 3D volume simulator
call the *same* function and cannot drift apart.

Pure numpy + the vendored ``senDOE`` geometry — no Streamlit, no Pyomo, no solver. Safe to
import from a headless script or a test.

NOTE: ``bundle_r_values`` is mirrored in two other places that must stay in sync — the backend
copy at ``tomography_uq.py`` (inside the geometry build loop) and the JS ``bundleR()`` in
``live_sim_component/index.html``.
"""

from functools import lru_cache

import numpy as np

# Vendored geometry primitives (importing them is not a backend change).
from senDOE.helpers.geometry import (
    get_line_abc_from_r_theta,
    line_grid_intersections,
)


@lru_cache(maxsize=8192)
def ray_geometry(r: float, theta: float, h: int, w: int):
    """Pixels and chord lengths a ray crosses — ``(rows, cols, seg_lengths, forward)``.

    Depends only on the ray and the grid **shape**, never on pixel values, so one result is
    valid for every z-slice and for every point in time. That is what makes the 3D simulator
    affordable: the expensive part of the vendored ``line_grid_intersections`` (a Python set,
    a sort and a loop) runs once per distinct ray instead of once per ray *per slice*.

    ``rows``/``cols`` are the pixels at the ``len(rows)`` grid crossings in the vendored
    ascending-``(x, y)`` order; ``seg_lengths`` has ``len(rows) - 1`` entries, one per segment
    between consecutive crossings. The vendored ``radon`` is recovered exactly as
    ``seg_lengths * image[rows[:-1], cols[:-1]]``.

    ``forward`` says whether that ascending order already runs along the beam-travel tangent
    ``(-sinθ, cosθ)``. Returns ``None`` if the ray never enters the grid.
    """
    a, b, c = get_line_abc_from_r_theta(r, theta)
    probe = np.zeros((h, w))  # values are irrelevant here; we only want the geometry
    try:
        crossings, pixels, _radon, seg_lengths = line_grid_intersections(
            a, b, c, probe, x_range=[-w / 2, w / 2], y_range=[-h / 2, h / 2]
        )
    except IndexError:
        return None  # line never enters the grid
    if len(pixels) == 0:
        return None
    # The vendored list is always sorted by ascending (x, y) regardless of θ, so θ and θ+180
    # are the same line and would deposit dose in the same order. The points are colinear, so
    # that order is either aligned with the travel tangent or exactly reversed.
    dx = crossings[-1][0] - crossings[0][0]
    dy = crossings[-1][1] - crossings[0][1]
    forward = bool(dx * (-np.sin(theta)) + dy * np.cos(theta) >= 0)
    return (pixels[:, 0].astype(int), pixels[:, 1].astype(int),
            np.asarray(seg_lengths, dtype=float), forward)


def ray_line_integral(image, r, theta) -> float:
    """Line integral of one ray through a 2D image — the vendored ``sum(radon)``.

    This is the sinogram value for that ray. Returns 0.0 if the ray misses the grid.
    """
    g = ray_geometry(float(r), float(theta), *image.shape)
    if g is None:
        return 0.0
    rows, cols, seg_lengths, _ = g
    m = len(seg_lengths)
    return float(seg_lengths @ image[rows[:m], cols[:m]])


def ray_line_integral_stack(vol, r, theta):
    """Line integrals of one ray through **every** slice of a ``(row, col, slice)`` volume.

    The ray path is shared across slices, so all of them collapse into a single ``einsum``.
    Returns a length-``n_slices`` array, or ``None`` if the ray misses the grid.
    """
    g = ray_geometry(float(r), float(theta), vol.shape[0], vol.shape[1])
    if g is None:
        return None
    rows, cols, seg_lengths, _ = g
    m = len(seg_lengths)
    return np.einsum("i,ik->k", seg_lengths, vol[rows[:m], cols[:m], :])


def degradation_dose_response(image, r, theta, I0, alpha, beta):
    """numpy port of ``util.HelperTools.degradation_Dose_Response`` (numpy branch).

    Degrades the image along the ray ``x·cosθ + y·sinθ = r`` using the dose-response model
    ``pixel·exp(-α·I_local - β·I_local²)`` with ``I_local = I0·exp(-Σ radon)``, where Σ runs in
    the beam **travel direction** ``(-sinθ, cosθ)`` — so the entry pixel sees full ``I0`` and 0°
    (bottom-up) differs from 180° (top-down). The ray path comes from the cached
    :func:`ray_geometry` (which wraps the vendored intersection routine). Returns the image
    unchanged if the ray misses the grid, mirroring the backend's |r| clamp.
    """
    g = ray_geometry(float(r), float(theta), *image.shape)
    if g is None:
        return image  # line never enters the grid → no-op
    rows, cols, seg_lengths, forward = g
    # radon[i] = chord_length_i * pixel_value_at_crossing_i, exactly as the vendored routine
    # builds it — but from the cached geometry, and read off the ORIGINAL image so the walk
    # below (which writes into a copy) sees undamaged values, as it always has.
    values = image[rows, cols]
    radon = seg_lengths * values[: len(seg_lengths)]

    out = image.copy()
    n = len(rows)
    # Walk the pixels in beam-travel order so 0° (bottom-up) differs from 180° (top-down); see
    # ``ray_geometry`` for why the cached order may need reversing.
    indices = range(n) if forward else range(n - 1, -1, -1)
    dose = 0.0
    for i in indices:
        local = I0 * np.exp(-dose)
        out[rows[i], cols[i]] = values[i] * np.exp(-alpha * local - beta * local**2)
        seg = i if forward else i - 1  # segment crossed to reach the next pixel in travel order
        if 0 <= seg < len(radon):
            dose += radon[seg]
    return out


def bundle_r_values(offset: float, n_beams: int, image_res: int) -> list:
    """Radial positions of a ray bundle (BeamStep convention: n rays, 1 unit apart, centered).

    ``n_beams == 0`` ⇒ full ``image_res`` fan. Each ray is snapped onto the center of the pixel
    it falls in: pixel centers sit at half-integers because geometry maps ``x → col = floor(x +
    image_res/2)``, so without the snap an odd ray count / integer offset lands a beam on a pixel
    *edge* (it then degrades the pixel to its right while the drawn line runs along the boundary).
    Snapping leaves which pixel is hit unchanged but centers the beam on it; even fans / the
    default full fan are already half-integer, so they are byte-identical. Rays outside the grid
    are dropped with the same ``|r| <= image_res/2 - 0.5`` clamp the backend uses (guards the
    vendored empty-intersection IndexError).
    """
    n = int(n_beams) if int(n_beams) > 0 else int(image_res)
    r_max = image_res / 2 - 0.5 + 1e-9
    rs = np.floor(offset + (np.arange(n) - (n - 1) / 2.0)) + 0.5
    return [float(r) for r in rs if abs(r) <= r_max]

