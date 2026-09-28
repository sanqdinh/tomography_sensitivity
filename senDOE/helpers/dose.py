"""Dose along a ray: the multiplicative dose-response degradation, dose accumulation, and
optical-depth normalisation of a phantom.

Pure numpy. The ray paths come from :func:`senDOE.helpers.rays.ray_geometry`.

:func:`degradation_dose_response` is the dose-response model ``pixel*exp(-a*I - b*I^2)`` that the
2D live picture and :mod:`senDOE.models.tomography_3d` both call, so the two cannot drift apart.
"""

import numpy as np

from senDOE.helpers.rays import bundle_r_values, ray_geometry, ray_line_integral


def accumulate_dose(f, r_values, angle_rad: float, I0: float, c_q: float):
    """Steps 1 and 2: returns ``(dQ, I_sum)`` for one bundle.

    ``dQ`` is ``c_q * I_p * delta_p`` summed over the bundle's rays -- the dose increment of
    eq:xd_dose_state. ``I_sum`` is ``I_p`` itself, likewise summed, which eq:xd_decay needs
    separately: the decay is driven by the *instantaneous local fluence*, not by the dose, and
    the two differ by the chord length and ``c_q``.

    ``I_p`` is the Beer-Lambert intensity delivered to pixel ``p`` by one ray, with the sum in
    the exponent running strictly over the *upstream* pixels, so the entry pixel sees the full
    ``I0``.  This is the same travel-order walk the dose-response model performs -- the
    quantity it calls ``local`` *is* ``I_p`` -- but here it drives a dose accumulator rather
    than a multiplicative decay.

    Rays of one measurement are simultaneous, so each integrates ``f`` as it stood at the start
    of the step and their dose contributions add (the same superposition the 3D tab uses).
    Returns the dose increment, shaped like ``f``.
    """
    f = np.asarray(f, dtype=float)
    dQ = np.zeros_like(f)
    I_sum = np.zeros_like(f)
    for r in r_values:
        g = ray_geometry(float(r), float(angle_rad), f.shape[0], f.shape[1])
        if g is None:
            continue  # ray never enters the grid
        rows, cols, seg_lengths, forward = g
        n = len(rows)
        n_seg = len(seg_lengths)
        # radon[i] = chord_i * f at crossing i, exactly as the vendored routine builds it.
        radon = seg_lengths * f[rows[:n_seg], cols[:n_seg]]
        indices = range(n) if forward else range(n - 1, -1, -1)
        shielding = 0.0
        for i in indices:
            local = I0 * np.exp(-shielding)          # I_p, before this pixel attenuates anything
            seg = i if forward else i - 1            # chord traversed on leaving crossing i
            if 0 <= seg < n_seg:
                # Deposit into the pixel that OWNS this chord, which is rows[seg] -- not rows[i].
                # eq:xd_dose_state is Q_{k+1,p} = Q_{k,p} + c_q I_p delta_p: the deposit pixel and
                # the chord's owner are the same symbol p, in one line. They coincide on forward
                # rays (seg == i) and differ by one when travel runs against the vendored
                # ascending-(x, y) crossing order, where depositing into rows[i] made every pixel
                # shield ITSELF -- the chord added below belonged to the pixel written next.
                # Forward rays are bit-identical to before; only antiparallel rays move.
                dst = i if forward else seg
                dQ[rows[dst], cols[dst]] += c_q * local * seg_lengths[seg]
                I_sum[rows[dst], cols[dst]] += local
                shielding += radon[seg]
            # the last pixel in travel order has no chord in the vendored convention -> no dose
    return dQ, I_sum


def peak_optical_depth(theta, image_res: int, n_angles: int = 12) -> float:
    """Largest line integral ``sum_p f_p*delta_p`` over a full fan at ``n_angles`` angles.

    The sample's opacity, and the one number that decides whether the model is in a sensible
    regime at all.  ``f`` is an attenuation coefficient, i.e. a reciprocal *length*, so its
    numerical size is meaningless without the pixel pitch: this app measures geometry in pixels
    (``x_range = [-w/2, w/2]`` over ``w`` pixels, so a chord through one pixel is ~1), where the
    manuscript's reference script used a ``[-1, 1]`` box on a 64 grid (chord ~0.03).  A phantom
    with O(1) values therefore has a peak optical depth near 34 here against ~1.1 there, and
    ``exp(-34)`` means the beam is entirely absorbed within the first couple of pixels: all dose
    lands in a two-pixel entry rim, nothing downstream is ever measured, and every diagnostic the
    manuscript quotes is off by an order of magnitude.  Real tomography sits at ``mu*L`` of order
    one, which is what :func:`scale_to_optical_depth` restores.
    """
    theta = np.asarray(theta, dtype=float)
    best = 0.0
    for a in range(n_angles):
        angle = np.pi * a / n_angles
        for r in bundle_r_values(0.0, 0, int(image_res)):
            best = max(best, ray_line_integral(theta, r, angle))
    return float(best)


def scale_to_optical_depth(theta, target: float, image_res: int, n_angles: int = 12):
    """Rescale ``theta`` so its peak optical depth is ``target``; see :func:`peak_optical_depth`.

    Self-normalising, so the same ``target`` means the same physics at any grid resolution.
    """
    depth = peak_optical_depth(theta, image_res, n_angles)
    if depth <= 0.0:
        return np.asarray(theta, dtype=float).copy()
    return np.asarray(theta, dtype=float) * (float(target) / depth)


def degradation_dose_response(image, r, theta, I0, alpha, beta):
    """numpy port of ``util.HelperTools.degradation_Dose_Response`` (numpy branch).

    Degrades the image along the ray ``x·cosθ + y·sinθ = r`` using the dose-response model
    ``pixel·exp(-α·I_local - β·I_local²)`` with ``I_local = I0·exp(-Σ radon)``, where Σ runs in
    the beam **travel direction** ``(-sinθ, cosθ)`` — so the entry pixel sees full ``I0`` and 0°
    (bottom-up) differs from 180° (top-down). The ray path comes from the cached
    :func:`senDOE.helpers.rays.ray_geometry` (which wraps the vendored intersection routine).
    Returns the image unchanged if the ray misses the grid, mirroring the backend's |r| clamp.
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
        seg = i if forward else i - 1  # segment crossed to reach the next pixel in travel order
        valid = 0 <= seg < len(radon)
        # Degrade the pixel that OWNS the chord about to be traversed. On forward rays seg == i,
        # so this is unchanged (including the tail at i = n-1, which owns no chord and is still
        # degraded, preserving this function's long-standing convention). On antiparallel rays
        # the owner is rows[seg] = rows[i-1], and writing rows[i] instead made every pixel shield
        # itself -- the chord added below belonged to the pixel written on the NEXT iteration.
        # The i = 0 tail is skipped there because rows[0] has already been written as the owner
        # of segment 0, and rewriting it with its own chord included is exactly the defect.
        if forward:
            dst = i
        elif valid:
            dst = seg
        else:
            continue
        local = I0 * np.exp(-dose)
        out[rows[dst], cols[dst]] = values[dst] * np.exp(-alpha * local - beta * local**2)
        if valid:
            dose += radon[seg]
    return out


# --- a check against the SPEC, not against a sibling implementation ---------------------
# Run it with `python3 -m senDOE.helpers.dose`.

def reference_local_intensity(f, r, angle_rad, I0, c_q):
    """eq:xd_local_intensity and eq:xd_dose_state, written from the spec alone.

    The spec says the ray "crosses pixels p_1, p_2, ... IN TRAVERSAL ORDER with chord lengths
    delta_{p_i}" and sums over ``m < i``: upstream material shields downstream material, and a
    pixel never shields itself.  So: walk the SEGMENTS in travel order; the pixel owning chord
    ``s`` is ``rows[s]``; deposit there; then let it shield everything after it.

    **How this resolves delta_p for a bundle, which the spec leaves implicit.**  The note writes
    ``delta_p`` as though a pixel has one chord, and under a single ray it does.  Under a bundle
    it does not: eq:xd_local_intensity "sums the contributions" of simultaneous rays and a pixel
    is crossed ~1.17 times per projection on a 64 grid, so the chord is really per
    ``(ray, pixel)`` pair while the notation carries no ray index.  This resolves it as:
    ``I_p`` is the sum over rays of the per-ray intensity and carries NO chord factor, while the
    dose increment is the sum over rays of ``c_q * I_ray * delta_(ray,p)``.  That is the same
    reading :func:`accumulate_dose` takes, so the two can differ only in the deposit index.

    This exists because agreeing with a sibling implementation to 1e-16 proves only that two
    implementations share a convention -- including a wrong one.  The manuscript records exactly
    that failure mode for its own ordering test ("easy to write so that it passes vacuously").
    Structural invariants (mass balance, the ``c_cp = 0`` collapse, the ``I0 = 0`` identity)
    cannot catch a misplaced deposit, because they are self-consistency properties of the
    composition and none of them asks which pixel the dose landed in.  This one does.
    """
    f = np.asarray(f, dtype=float)
    res = f.shape[0]
    dQ = np.zeros_like(f)
    I_sum = np.zeros_like(f)
    g = ray_geometry(float(r), float(angle_rad), res, res)
    if g is None:
        return dQ, I_sum
    rows, cols, seg, forward = g
    n_seg = len(seg)
    shield = 0.0
    for s in (range(n_seg) if forward else range(n_seg - 1, -1, -1)):
        local = I0 * np.exp(-shield)
        pix = (int(rows[s]), int(cols[s]))
        dQ[pix] += c_q * local * seg[s]
        I_sum[pix] += local
        shield += seg[s] * f[pix]
    return dQ, I_sum


def check_photon_balance(image_res: int = 12, verbose: bool = True, strict: bool = True):
    """Compare :func:`accumulate_dose` against :func:`reference_local_intensity`.

    Reports forward-ordered and antiparallel rays separately, because that is where they differ.
    Both families are asserted.  They did not always agree: antiparallel rays were out by a
    full ``I0`` until the deposit index was fixed, because each pixel was shielded by its own
    chord.  This is the check that found it.

    **What it does NOT certify, and this is the honest limit of it.**  The reference calls
    :func:`senDOE.helpers.rays.ray_geometry` for its crossing list, so it is independent of
    ``accumulate_dose`` only in the *walk* -- the travel order and the deposit index.  It shares
    the underlying chord/pixel attribution and is therefore blind to any error in it.  There is
    one, in the chord attribution on positive-slope rays.  The general lesson applies here one
    level down, which is worth saying plainly rather than leaving for someone to discover: a
    reference is only independent along the axes on which it does not reuse the
    thing it checks.  (The chord-attribution check lives with the archived prototypes.)
    """

    rng = np.random.default_rng(7)
    res = int(image_res)
    f = rng.random((res, res)) * 0.4          # non-uniform: a symmetric object hides the defect
    worst = {"forward": 0.0, "antiparallel": 0.0}
    count = {"forward": 0, "antiparallel": 0}
    for ang_deg in np.arange(0.0, 360.0, 15.0):
        ang = float(np.deg2rad(ang_deg))
        for r in bundle_r_values(0.0, 0, res):
            g = ray_geometry(float(r), ang, res, res)
            if g is None:
                continue
            key = "forward" if g[3] else "antiparallel"
            _dq_r, I_ref = reference_local_intensity(f, r, ang, 1.0, 1.0)
            _dq_c, I_code = accumulate_dose(f, [r], ang, 1.0, 1.0)
            worst[key] = max(worst[key], float(np.abs(I_ref - I_code).max()))
            count[key] += 1
    if verbose:
        print("eq:xd_local_intensity -- accumulate_dose against a reference from the spec alone")
        for k in ("forward", "antiparallel"):
            print("    %-13s rays: max |I_p(code) - I_p(spec)| = %.3e   over %d rays"
                  % (k, worst[k], count[k]))
        if worst["antiparallel"] > 1e-12:
            print("    ^ antiparallel rays disagree. On those the deposit lands on rows[i] while")
            print("      the chord just added to the shielding belongs to rows[i-1], so each pixel")
            print("      is shielded by its own chord. degradation_dose_response has the same")
            print("      split, so the 2D live picture and the 3D simulator share it.")
    assert worst["forward"] < 1e-12, "forward rays disagree with the spec -- that is a new bug"
    if strict:
        assert worst["antiparallel"] < 1e-12, (
            "antiparallel rays disagree with eq:xd_local_intensity by %.3e -- the deposit index "
            "regressed: eq:xd_dose_state puts the deposit pixel and the chord owner at the same "
            "symbol p, so deposit into rows[s], not rows[i]" % worst["antiparallel"])
    return worst


if __name__ == "__main__":
    check_photon_balance()
