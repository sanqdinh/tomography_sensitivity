"""3D Shepp-Logan phantom + dose-response degradation (forward simulation only).

This is the **2.5D** companion to the 2D live simulator in ``app.py``. The volume is a real
3D Shepp-Logan phantom, but the beam geometry is a stack of independent in-plane bundles:
each measurement fires the *same* ``(r, theta)`` bundle through every z-slice, and rays never
cross between slices. That makes the per-slice ray math identical to the established 2D path,
so this module degrades each slice by calling the very same
:func:`dose_response.degradation_dose_response` the 2D picture uses -- the two cannot drift.

Degradation still differs from slice to slice even though the geometry is shared, because each
slice presents different material: ``I_local = I0*exp(-sum radon)`` attenuates according to what
that particular slice's beam has already traversed.

Deliberately **not** here: reconstruction, any Pyomo model, any 3D voxel-intersection routine.
This is pure numpy and needs no IPOPT / k_aug, so it runs anywhere in seconds.

Run headless as a smoke test::

    python3 tomography_3d.py
"""

from __future__ import annotations

import numpy as np

from dose_response import (
    bundle_r_values,
    degradation_dose_response,
    ray_line_integral_stack,
)

# --- 3D Shepp-Logan ellipsoid tables ----------------------------------------------------
# One row per ellipsoid: (a, b, c, x0, y0, z0, phi_deg, theta_deg, psi_deg, value).
# a/b/c are semi-axes and x0/y0/z0 the center, all in the normalized [-1, 1] head frame;
# the three Euler angles orient the ellipsoid (ZYZ). Voxels inside an ellipsoid have its
# ``value`` ADDED, so the tables are read top-to-bottom: the skull is laid down first and the
# interior structures are carved out of it by later, negative entries.
#
# theta == 0 for every row in both tables, so the ZYZ rotation collapses to a rotation about z
# (i.e. within the slice plane) -- the general rotation is implemented anyway so a caller can add
# genuinely out-of-plane ellipsoids without touching the rasterizer.

# Textbook contrast (Kak & Slaney). Faithful to the literature, but the interior features sit
# within ~2% of the surrounding tissue, so they wash out as soon as degradation dims the image.
SHEPP_LOGAN_3D_CLASSIC = (
    (0.6900, 0.920, 0.810, 0.000, 0.0000, 0.00, 0.0, 0.0, 0.0, 1.00),
    (0.6624, 0.874, 0.780, 0.000, -0.0184, 0.00, 0.0, 0.0, 0.0, -0.98),
    (0.1100, 0.310, 0.220, 0.220, 0.0000, 0.00, -18.0, 0.0, 10.0, -0.02),
    (0.1600, 0.410, 0.280, -0.220, 0.0000, 0.00, 18.0, 0.0, 10.0, -0.02),
    (0.2100, 0.250, 0.410, 0.000, 0.3500, -0.15, 0.0, 0.0, 0.0, 0.01),
    (0.0460, 0.046, 0.050, 0.000, 0.1000, 0.25, 0.0, 0.0, 0.0, 0.01),
    (0.0460, 0.046, 0.050, 0.000, -0.1000, 0.25, 0.0, 0.0, 0.0, 0.01),
    (0.0460, 0.023, 0.050, -0.080, -0.6050, 0.00, 0.0, 0.0, 0.0, 0.01),
    (0.0230, 0.023, 0.020, 0.000, -0.6060, 0.00, 0.0, 0.0, 0.0, 0.01),
    (0.0230, 0.046, 0.020, 0.060, -0.6050, 0.00, 0.0, 0.0, 0.0, 0.01),
)

# Same geometry, boosted contrast (Toft-style). This is the default: the point of the app is to
# *see* dose damage, and structures at 2% of background vanish under the exp() dimming.
SHEPP_LOGAN_3D_MODIFIED = tuple(
    row[:9] + (value,)
    for row, value in zip(
        SHEPP_LOGAN_3D_CLASSIC,
        (1.0, -0.8, -0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
    )
)

PHANTOM_VARIANTS = {
    "modified": SHEPP_LOGAN_3D_MODIFIED,
    "classic": SHEPP_LOGAN_3D_CLASSIC,
}

# Half-height of the sampled z range, in the normalized head frame. The outer skull ellipsoid has
# c = 0.81, so sampling out to 0.8 keeps every slice inside the head (a full [-1, 1] span would
# make the end slices entirely empty, which reads as a bug rather than as anatomy).
DEFAULT_Z_EXTENT = 0.8


def _rotation_zyz(phi_deg: float, theta_deg: float, psi_deg: float) -> np.ndarray:
    """ZYZ Euler rotation matrix (degrees) mapping ellipsoid-frame axes into world axes."""
    p, t, s = np.deg2rad([phi_deg, theta_deg, psi_deg])
    cp, sp = np.cos(p), np.sin(p)
    ct, st = np.cos(t), np.sin(t)
    cs, ss = np.cos(s), np.sin(s)
    rz1 = np.array([[cp, -sp, 0.0], [sp, cp, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]])
    rz2 = np.array([[cs, -ss, 0.0], [ss, cs, 0.0], [0.0, 0.0, 1.0]])
    return rz1 @ ry @ rz2


def shepp_logan_3d(
    image_res: int,
    n_slices: int,
    variant: str = "modified",
    z_extent: float = DEFAULT_Z_EXTENT,
) -> np.ndarray:
    """Rasterize the 3D Shepp-Logan phantom onto an ``(image_res, image_res, n_slices)`` grid.

    Axis order is ``[row, col, slice]`` so that ``vol[:, :, k]`` is a plain 2D image in exactly
    the layout the 2D code expects: axis 0 is the image row (``+y`` at the top, matching
    ``imshow(origin="upper")``) and axis 1 is the column (``+x`` to the right).

    Values are clipped to ``[0, 1]`` to match the range of the 2D ``shepp_logan_phantom`` and to
    keep the dose-response model on non-negative pixels.
    """
    if image_res < 1 or n_slices < 1:
        raise ValueError("image_res and n_slices must both be >= 1")
    try:
        table = PHANTOM_VARIANTS[variant]
    except KeyError:
        raise ValueError(
            "variant must be one of %s, got %r" % (sorted(PHANTOM_VARIANTS), variant)
        ) from None

    # Rows run +y (top) -> -y (bottom); columns run -x -> +x. A single slice is the mid-plane.
    row_y = np.linspace(1.0, -1.0, image_res)
    col_x = np.linspace(-1.0, 1.0, image_res)
    sl_z = (
        np.zeros(1)
        if n_slices == 1
        else np.linspace(-z_extent, z_extent, n_slices)
    )
    yy, xx, zz = np.meshgrid(row_y, col_x, sl_z, indexing="ij")

    vol = np.zeros((image_res, image_res, n_slices), dtype=float)
    for a, b, c, x0, y0, z0, phi, theta, psi, value in table:
        rot = _rotation_zyz(phi, theta, psi)
        dx, dy, dz = xx - x0, yy - y0, zz - z0
        # Express the offset in the ellipsoid's own frame: R^T @ d (columns of R dotted with d).
        xr = rot[0, 0] * dx + rot[1, 0] * dy + rot[2, 0] * dz
        yr = rot[0, 1] * dx + rot[1, 1] * dy + rot[2, 1] * dz
        zr = rot[0, 2] * dx + rot[1, 2] * dy + rot[2, 2] * dz
        vol[(xr / a) ** 2 + (yr / b) ** 2 + (zr / c) ** 2 <= 1.0] += value

    return np.clip(vol, 0.0, 1.0)


# --- detector axis ----------------------------------------------------------------------
# ``bundle_r_values`` snaps every ray onto a half-integer, for any offset, so the set of
# reachable detector positions is a FIXED lattice that does not depend on the geometry the user
# builds. That gives every measurement a common r axis to plot against.


def detector_grid(image_res: int) -> np.ndarray:
    """The complete detector lattice: ``image_res`` half-integer positions, spacing 1.0.

    For ``image_res = 30`` that is ``-14.5, -13.5, ..., 14.5``. Any ray bundle, at any offset,
    is a subset of this.
    """
    return np.arange(image_res) - image_res / 2.0 + 0.5


def detector_index(r: float, image_res: int) -> int:
    """Row of :func:`detector_grid` that a ray at radius ``r`` lands on."""
    return int(round(r + image_res / 2.0 - 0.5))


def simulate_3d(vol, seq, I0: float, alpha: float, beta: float, image_res: int = None):
    """Degrade a volume over a measurement sequence and record what each measurement saw.

    Returns ``(degraded_vol, sino)``.

    ``sino`` has shape ``(image_res, n_measurements, n_slices)`` and holds the line integral
    (the vendored ``sum(radon)``) for every measured ray. Cells the user never sampled are
    **NaN, not 0** -- a 0 would be indistinguishable from a genuine low reading, which is the
    trap that makes the vendored ``extract_sinogram_value`` misleading. A hand-built sequence
    samples only a handful of detector slots, so most of the array is legitimately unmeasured.

    Two useful views fall straight out of it:

    * ``sino[:, :, k]``  -- the sinogram of slice ``k`` (detector r vs measurement).
    * ``sino[:, m, :]``  -- the 2D projection of measurement ``m`` (detector r vs slice z),
      i.e. what a 2D detector panel behind the volume would record.

    **Timing:** every ray of one measurement integrates the volume as it stood at the *start*
    of that measurement, so rays within a measurement do not see each other's damage. That
    treats a measurement as simultaneous and matches the Pyomo backend, where each ray of a
    step reads ``image_array[injection_time]`` and their intensities superpose. Degradation
    itself stays strictly sequential, exactly as the 2D live picture applies it.
    """
    if vol.ndim != 3:
        raise ValueError("expected a (row, col, slice) volume, got shape %r" % (vol.shape,))
    if image_res is None:
        image_res = vol.shape[1]

    out = vol.copy()
    n_slices = out.shape[2]
    seq = tuple(seq)
    sino = np.full((image_res, len(seq), n_slices), np.nan, dtype=float)

    for m, (angle_deg, offset, n_beams) in enumerate(seq):
        theta = np.deg2rad(angle_deg)
        rs = bundle_r_values(offset, n_beams, image_res)

        # Measure first, against the volume as it is BEFORE this measurement damages it.
        snapshot = out.copy()
        for r in rs:
            integrals = ray_line_integral_stack(snapshot, r, theta)
            if integrals is not None:
                sino[detector_index(r, image_res), m, :] = integrals

        # Then apply the dose, ray by ray, slice by slice (the established 2D order).
        for r in rs:
            for k in range(n_slices):
                out[:, :, k] = degradation_dose_response(
                    out[:, :, k], r, theta, I0, alpha, beta
                )

    return out, sino


def degrade_volume(
    vol: np.ndarray,
    seq,
    I0: float,
    alpha: float,
    beta: float,
    image_res: int = None,
) -> np.ndarray:
    """Cumulative dose-response degradation of a volume over a measurement sequence.

    ``seq`` is the same hashable ``((angle_deg, offset, n_beams), ...)`` the 2D path uses, so the
    measurement table needs no extra columns: one measurement fires one bundle through *every*
    slice.

    Each slice is degraded by the shared 2D routine, in the same (step, ray) order the 2D live
    picture uses -- so a one-slice volume reproduces the 2D result exactly. Slices are mutually
    independent, so the order of the slice loop is irrelevant. Returns a new array.

    Thin wrapper over :func:`simulate_3d` that discards the sinogram — recording it costs
    almost nothing, since the ray geometry is cached and shared across slices.
    """
    return simulate_3d(vol, seq, I0, alpha, beta, image_res)[0]


if __name__ == "__main__":
    # Headless smoke test: build a volume, degrade it, and report that dose actually landed.
    IMAGE_RES, N_SLICES = 30, 16
    SEQ = ((0.0, 0.0, 0), (45.0, 2.0, 7), (90.0, -3.5, 5), (135.0, 0.0, 0))
    I0, ALPHA, BETA = 5.0, 0.3, 0.01

    vol0 = shepp_logan_3d(IMAGE_RES, N_SLICES)
    print(
        "phantom: shape=%s range=[%.4f, %.4f] nonzero=%.1f%%"
        % (vol0.shape, vol0.min(), vol0.max(), 100.0 * np.count_nonzero(vol0) / vol0.size)
    )

    vol1, sino = simulate_3d(vol0, SEQ, I0, ALPHA, BETA, IMAGE_RES)
    print(
        "after %d measurements (I0=%.1f): range=[%.4f, %.4f]  total %.3f -> %.3f (-%.2f%%)"
        % (
            len(SEQ), I0, vol1.min(), vol1.max(), vol0.sum(), vol1.sum(),
            100.0 * (1.0 - vol1.sum() / vol0.sum()),
        )
    )
    assert np.all(vol1 <= vol0 + 1e-12), "degradation must never brighten a voxel"
    assert vol1.sum() < vol0.sum(), "degradation must remove intensity"

    print("per-slice mean (before -> after):")
    for k in range(N_SLICES):
        print(
            "  z=%2d  %.4f -> %.4f" % (k, vol0[:, :, k].mean(), vol1[:, :, k].mean())
        )

    # I0 == 0 is the app default and must be an exact no-op.
    assert np.array_equal(degrade_volume(vol0, SEQ, 0.0, ALPHA, BETA, IMAGE_RES), vol0)

    # --- sinogram ------------------------------------------------------------------------
    measured = ~np.isnan(sino)
    assert sino.shape == (IMAGE_RES, len(SEQ), N_SLICES), sino.shape
    print(
        "\nsinogram: shape=%s  measured cells=%d of %d (%.0f%%)  range=[%.3f, %.3f]"
        % (sino.shape, measured.sum(), sino.size,
           100.0 * measured.sum() / sino.size,
           np.nanmin(sino), np.nanmax(sino))
    )
    # Exactly the sampled detector slots are non-NaN, and nothing else.
    for m, (angle_deg, offset, n_beams) in enumerate(SEQ):
        want = {detector_index(r, IMAGE_RES) for r in bundle_r_values(offset, n_beams, IMAGE_RES)}
        got = {int(i) for i in np.where(measured[:, m, 0])[0]}
        assert want == got, (m, sorted(want ^ got))
    print("  detector slots recorded match bundle_r_values for every measurement")

    print("\nOK: I0=0 is a no-op; degradation is monotone and strictly lossy.")
