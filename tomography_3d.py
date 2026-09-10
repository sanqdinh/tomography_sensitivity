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
from scipy.ndimage import binary_erosion

from dose_response import (
    bundle_r_values,
    degradation_dose_response,
    ray_line_integral_stack,
)

# --- 3D Shepp-Logan ellipsoid tables ----------------------------------------------------
# One row per ellipsoid: (a, b, c, x0, y0, z0, phi_deg, theta_deg, psi_deg, value).
# a/b/c are semi-axes and x0/y0/z0 the center, all in the normalized [-1, 1] head frame;
# the three Euler angles orient the ellipsoid (ZYZ). Voxels inside an ellipsoid have its
# ``value`` ADDED, so the table is read top-to-bottom: the skull is laid down first, the second
# row pulls the interior down to the brain background, and each later row lifts its structure
# above that background. The table entry is therefore NOT the tissue value -- see
# :func:`shepp_logan_table`.
#
# theta == 0 for every row, so the ZYZ rotation collapses to a rotation about z
# (i.e. within the slice plane) -- the general rotation is implemented anyway so a caller can add
# genuinely out-of-plane ellipsoids without touching the rasterizer.

# Geometry and provenance: the textbook Kak & Slaney table. The first NINE columns -- semi-axes,
# centre and Euler angles -- are used VERBATIM and are never modified. The tenth column is the
# published intensity, kept for reference only; what this module actually renders comes from
# _TISSUE_TARGETS below.
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

# --- intensity scheme ---------------------------------------------------------------------
# Tissue values at contrast = 1. Every interior structure sits ABOVE the brain background, so no
# voxel inside the skull is ever 0. That is deliberate: 0 is also the value of air, so a
# zero-valued cavity is indistinguishable from empty space in the Volume view. The textbook
# table put the ventricles at exactly brain - 0.2 = 0.0, which made the interior read as part
# solid, part empty and hid the structures behind a featureless shell.
CRUST_VALUE = 1.0      # outer skull; also the intensity the crust shell is painted with
BRAIN_VALUE = 0.1      # the bulk of the skull interior
FEATURE_VALUE = 0.3    # ventricles, upper blob, bottom cluster
EYE_VALUE = 0.4        # the two small floating spheres at z = +0.25

# Tissue target per ellipsoid, in table order. These are the values a voxel ENDS UP with, not
# the additive table entries -- shepp_logan_table() converts between the two.
_TISSUE_TARGETS = (
    CRUST_VALUE,     # 1  outer skull
    BRAIN_VALUE,     # 2  inner skull -> sets the interior background
    FEATURE_VALUE,   # 3  right ventricle
    FEATURE_VALUE,   # 4  left ventricle
    FEATURE_VALUE,   # 5  upper blob
    EYE_VALUE,       # 6  floating sphere ("eye")
    EYE_VALUE,       # 7  floating sphere ("eye")
    FEATURE_VALUE,   # 8  bottom cluster
    FEATURE_VALUE,   # 9  bottom cluster
    FEATURE_VALUE,   # 10 bottom cluster
)

# Contrast scales how far each interior structure sits above the brain background. 0 leaves a
# uniform BRAIN_VALUE interior, 1 gives the targets above, higher exaggerates. The skull and the
# brain are NOT scaled, so the crust -- and therefore the Volume view's colour ceiling -- holds
# still, and no structure ever passes back through the background value on the way.
DEFAULT_CONTRAST = 1.0


def shepp_logan_table(contrast: float = DEFAULT_CONTRAST) -> tuple:
    """Ellipsoid table whose value column yields :data:`_TISSUE_TARGETS` at ``contrast=1``.

    Values are ADDED where ellipsoids overlap, so a structure's table entry is its offset from
    whatever is already there, not its final intensity:

    * row 1 (outer skull) is absolute -- ``CRUST_VALUE``;
    * row 2 (inner skull) is ``BRAIN_VALUE - CRUST_VALUE``, which pulls everything inside the
      skull down to the brain background;
    * every later row is ``contrast * (target - BRAIN_VALUE)`` -- its height above that
      background, scaled by the slider.

    Rows 1 and 2 are deliberately left unscaled so the crust and the brain are the same at every
    contrast, and every feature approaches the background from one side only.
    """
    t = float(contrast)
    rows = []
    for n, (row, target) in enumerate(zip(SHEPP_LOGAN_3D_CLASSIC, _TISSUE_TARGETS)):
        if n == 0:
            value = CRUST_VALUE
        elif n == 1:
            value = BRAIN_VALUE - CRUST_VALUE
        else:
            value = t * (target - BRAIN_VALUE)
        rows.append(row[:9] + (value,))
    return tuple(rows)


# Half-height of the sampled z range, in the normalized head frame. The outer skull ellipsoid has
# c = 0.81, so sampling out to 0.8 keeps every slice inside the head (a full [-1, 1] span would
# make the end slices entirely empty, which reads as a bug rather than as anatomy).
DEFAULT_Z_EXTENT = 0.8

# Thickness, IN VOXELS, of the uniform outer crust. The analytic skull is the gap between the
# two outermost ellipsoids (a = 0.690 vs 0.6624), which at image_res = 30 is 0.41 voxels wide --
# under a voxel, so rasterizing it directly gives a broken, aliased ring that changes with every
# resolution and slice count. A morphological shell instead gives a closed, uniform crust that is
# always exactly this many voxels thick, whatever the grid.
DEFAULT_CRUST_VOXELS = 1


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
    contrast: float = DEFAULT_CONTRAST,
    z_extent: float = DEFAULT_Z_EXTENT,
    crust_voxels: int = DEFAULT_CRUST_VOXELS,
    crust_value: float = None,
) -> np.ndarray:
    """Rasterize the 3D Shepp-Logan phantom onto an ``(image_res, image_res, n_slices)`` grid.

    Axis order is ``[row, col, slice]`` so that ``vol[:, :, k]`` is a plain 2D image in exactly
    the layout the 2D code expects: axis 0 is the image row (``+y`` at the top, matching
    ``imshow(origin="upper")``) and axis 1 is the column (``+x`` to the right).

    Values are clipped to ``[0, 1]`` to match the range of the 2D ``shepp_logan_phantom`` and to
    keep the dose-response model on non-negative pixels.

    Tissue values at ``contrast=1``: crust/skull 1.0, brain 0.1, ventricles + upper blob +
    bottom cluster 0.3, and the two floating "eye" spheres 0.4. **Nothing inside the skull is
    0** -- 0 is air, and a zero-valued cavity is indistinguishable from empty space in the
    Volume view. ``contrast`` scales each structure's height above the 0.1 background; see
    :func:`shepp_logan_table`.

    Resolution caveat: the eye spheres are r = 0.046, about 0.7 voxel at ``image_res=30``, so
    they land on only ~2 voxels at the app's default grid (4 at ``n_slices=48``); the three
    bottom-cluster ellipsoids are smaller still and render **no** voxels at any grid this app
    offers. Their values are therefore nominal at low resolution -- the published geometry is
    kept as-is rather than inflated to suit the display.

    **Crust.** The outermost ``crust_voxels`` layer of the head is forced to one constant
    intensity (``crust_value``, defaulting to the table's skull value). The analytic skull --
    the gap between the two outer ellipsoids -- is thinner than one voxel at the resolutions
    this app uses, so rasterizing it gives a broken ring that looks different at every grid
    size. Eroding the head mask instead yields a closed shell of exactly the requested voxel
    thickness in *all three* directions, so it looks the same whatever ``image_res`` and
    ``n_slices`` are. Pass ``crust_voxels=0`` for the raw ellipsoid phantom.
    """
    if image_res < 1 or n_slices < 1:
        raise ValueError("image_res and n_slices must both be >= 1")
    table = shepp_logan_table(contrast)

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
    head_mask = inner_mask = None
    for n, (a, b, c, x0, y0, z0, phi, theta, psi, value) in enumerate(table):
        rot = _rotation_zyz(phi, theta, psi)
        dx, dy, dz = xx - x0, yy - y0, zz - z0
        # Express the offset in the ellipsoid's own frame: R^T @ d (columns of R dotted with d).
        xr = rot[0, 0] * dx + rot[1, 0] * dy + rot[2, 0] * dz
        yr = rot[0, 1] * dx + rot[1, 1] * dy + rot[2, 1] * dz
        zr = rot[0, 2] * dx + rot[1, 2] * dy + rot[2, 2] * dz
        inside = (xr / a) ** 2 + (yr / b) ** 2 + (zr / c) ** 2 <= 1.0
        vol[inside] += value
        if n == 0:
            head_mask = inside      # outer skull ellipsoid = the head boundary
        elif n == 1:
            inner_mask = inside     # inner skull ellipsoid = start of brain tissue

    vol = np.clip(vol, 0.0, 1.0)

    if crust_voxels > 0 and head_mask is not None:
        # 1. Flatten the analytic skull band to brain. At these resolutions it is a sub-voxel,
        #    aliased ring; leaving it would sit under the crust as a second, ragged shell.
        brain = max(float(table[0][9] + table[1][9]), 0.0)
        vol[head_mask & ~inner_mask] = brain
        # 2. Paint a closed shell exactly crust_voxels thick. border_value=0 treats outside the
        #    array as background, so a head touching the grid edge is still crusted there.
        core = binary_erosion(head_mask, iterations=int(crust_voxels), border_value=0)
        vol[head_mask & ~core] = float(
            table[0][9] if crust_value is None else crust_value
        )

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
