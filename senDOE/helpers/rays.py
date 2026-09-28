"""Beam and ray geometry on a pixel grid: which pixels a ray crosses, and in what order.

Pure numpy on top of the vendored :mod:`senDOE.helpers.geometry` intersection routine. Nothing
here reads pixel values except :func:`ray_line_integral`, so the geometry is cached per
``(r, theta, grid shape)`` and reused across every slice and every time step.

``bundle_r_values`` is mirrored in JavaScript in ``frontend/live_sim_component/index.html`` and
``frontend/volume_sim_component/index.html``; the two must stay in sync.
"""

from functools import lru_cache

import numpy as np

from senDOE.helpers.geometry import get_line_abc_from_r_theta, line_grid_intersections


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


def ray_walk(r: float, angle_rad: float, res: int):
    """:func:`senDOE.helpers.dose.accumulate_dose`'s travel-order walk, as a list of records.

    One record per crossing that actually deposits dose:
    ``(dose_pixel, chord, shield_pixel)``, with pixels flattened to ``row * res + col``.
    The shielding after record ``t`` is ``sum_{s <= t} chord_s * f[shield_pixel_s]``, and its
    final value is the ray integral.

    Returns ``None`` if the ray never enters the grid, matching ``ray_geometry``.
    """
    g = ray_geometry(float(r), float(angle_rad), int(res), int(res))
    if g is None:
        return None
    rows, cols, seg, forward = g
    n, n_seg = len(rows), len(seg)
    walk = []
    for i in (range(n) if forward else range(n - 1, -1, -1)):
        s = i if forward else i - 1          # chord traversed on leaving crossing i
        if 0 <= s < n_seg:
            # Deposit pixel and shielding pixel are BOTH the chord's owner, rows[s]: that is
            # eq:xd_dose_state, where they are the same symbol p. They are kept as separate
            # fields only because the constraint builder reads them separately.
            dst = i if forward else s
            walk.append((int(rows[dst]) * res + int(cols[dst]),    # receives the dose
                         float(seg[s]),                            # chord for that dose
                         int(rows[s]) * res + int(cols[s])))       # shields the rest of the ray
    return walk or None


def measurement_rays(seq, image_res: int):
    """``[(angle_rad, [(r, walk), ...]), ...]`` -- one entry per measurement, rays that hit."""
    out = []
    for angle_deg, offset, n_beams in seq:
        ang = float(np.deg2rad(float(angle_deg)))
        rays = []
        for r in bundle_r_values(float(offset), int(n_beams), int(image_res)):
            w = ray_walk(r, ang, image_res)
            if w is not None:
                rays.append((float(r), w))
        out.append((ang, rays))
    return out
