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

import numpy as np

# Vendored geometry primitives (importing them is not a backend change).
from senDOE.helpers.geometry import (
    get_line_abc_from_r_theta,
    line_grid_intersections,
)




def degradation_dose_response(image, r, theta, I0, alpha, beta):
    """numpy port of ``util.HelperTools.degradation_Dose_Response`` (numpy branch).

    Degrades the image along the ray ``x·cosθ + y·sinθ = r`` using the dose-response model
    ``pixel·exp(-α·I_local - β·I_local²)`` with ``I_local = I0·exp(-Σ radon)``, where Σ runs in
    the beam **travel direction** ``(-sinθ, cosθ)`` — so the entry pixel sees full ``I0`` and 0°
    (bottom-up) differs from 180° (top-down). Reuses the vendored ``get_line_abc_from_r_theta`` /
    ``line_grid_intersections``. Returns the image unchanged if the ray misses the grid (the
    vendored intersection routine raises IndexError on an empty hit, mirrored by the backend's
    |r| clamp).
    """
    h, w = image.shape
    a, b, c = get_line_abc_from_r_theta(r, theta)
    try:
        intersection_result, image_intersection, radon, _ = line_grid_intersections(
            a, b, c, image, x_range=[-w / 2, w / 2], y_range=[-h / 2, h / 2]
        )
    except IndexError:
        return image  # line never enters the grid → no-op
    if len(image_intersection) == 0:
        return image
    out = image.copy()
    n = len(image_intersection)
    # The beam travels along the line tangent (-sinθ, cosθ); the vendored intersection list is
    # always sorted by ascending (x, y) regardless of θ, so θ and θ+180 are the same line and
    # would otherwise deposit dose in the same order. Walk the pixels in travel order so 0°=
    # bottom-up and 180°=top-down differ (and 0°==360°). The points are colinear, so the vendored
    # order is either aligned with the tangent or exactly reversed.
    dx = intersection_result[-1][0] - intersection_result[0][0]
    dy = intersection_result[-1][1] - intersection_result[0][1]
    forward = dx * (-np.sin(theta)) + dy * np.cos(theta) >= 0
    indices = range(n) if forward else range(n - 1, -1, -1)
    dose = 0.0
    for i in indices:
        ix = int(image_intersection[i, 0])  # row
        iy = int(image_intersection[i, 1])  # col
        local = I0 * np.exp(-dose)
        out[ix, iy] = image[ix, iy] * np.exp(-alpha * local - beta * local**2)
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

