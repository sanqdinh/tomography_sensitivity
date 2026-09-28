"""Shape statistics of a 2D attenuation field: how compact it is, and how that changed.

Pure numpy. Every radius is in pixels.
"""

import numpy as np


def radius_of_gyration(f) -> float:
    """``sqrt(sum f r^2 / sum f)`` about the field's own centroid, in pixels.

    The compactness diagnostic: it is what "the sample shrinks" means quantitatively.  At
    ``c_cp = 0`` nothing moves, so it must come out *exactly* unchanged; above zero it must fall.
    """
    f = np.asarray(f, dtype=float)
    nr, nc = f.shape
    yy, xx = np.mgrid[0:nr, 0:nc]
    m = f.sum()
    if m <= 0:
        return float("nan")
    xc, yc = (xx * f).sum() / m, (yy * f).sum() / m
    return float(np.sqrt((f * ((xx - xc) ** 2 + (yy - yc) ** 2)).sum() / m))


def support_radius(f, frac: float = 0.99) -> float:
    """Radius about the field's own centroid containing ``frac`` of the mass, in pixels.

    This is what "the specimen shrinks" actually refers to.  Rg is NOT that statistic: mass
    moving outward in the bulk raises Rg while the rim draining inward lowers it, so Rg mixes
    the two and can come out either way without saying which happened.
    """
    f = np.asarray(f, dtype=float)
    nr, nc = f.shape
    yy, xx = np.mgrid[0:nr, 0:nc]
    m = f.sum()
    if m <= 0:
        return float("nan")
    xc, yc = (xx * f).sum() / m, (yy * f).sum() / m
    r = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2).ravel()
    w = np.clip(f.ravel(), 0.0, None)
    o = np.argsort(r)
    c = np.cumsum(w[o])
    if c[-1] <= 0:
        return float("nan")
    return float(r[o][np.searchsorted(c, frac * c[-1])])


def mass_outside(f, r0: float, centre=None) -> float:
    """Mass beyond radius ``r0``.  Must be zero or falling -- material must not leave the body."""
    f = np.asarray(f, dtype=float)
    nr, nc = f.shape
    yy, xx = np.mgrid[0:nr, 0:nc]
    if centre is None:
        m = f.sum()
        centre = ((xx * f).sum() / m, (yy * f).sum() / m)
    xc, yc = centre
    r = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2)
    return float(f[r > r0].sum())


def semi_axis_ratio(f) -> float:
    """``sqrt(lambda_max / lambda_min)`` of the mass second-moment tensor -- the V7 statistic.

    1.0 is isotropic; an anisotropic design (a narrow wedge of angles) should register above it.
    """
    f = np.asarray(f, dtype=float)
    nr, nc = f.shape
    yy, xx = np.mgrid[0:nr, 0:nc]
    m = f.sum()
    if m <= 0:
        return float("nan")
    xc, yc = (xx * f).sum() / m, (yy * f).sum() / m
    dx_, dy_ = xx - xc, yy - yc
    cxx = (f * dx_ * dx_).sum() / m
    cyy = (f * dy_ * dy_).sum() / m
    cxy = (f * dx_ * dy_).sum() / m
    ev = np.linalg.eigvalsh(np.array([[cxx, cxy], [cxy, cyy]]))
    lo, hi = float(ev[0]), float(ev[1])
    if lo <= 0:
        return float("nan")
    return float(np.sqrt(hi / lo))


def shape_diagnostics(theta, f):
    """``(support_pct, half_pct, flips)`` -- what the damage did to the body's shape.

    Radii are measured about the **fixed initial centroid**, not a moving one, so a body that
    translates does not read as a body that contracted.  ``flips`` counts sign changes in the
    radially banded mass difference: ONE is coherent condensation (mass leaves the outside and
    arrives inside), several means mass is shuffling between neighbours.

    Pure numpy, so a forward-only experiment does not have to import pyomo to measure a shape.
    Kept identical to the forward tab's readout in ``app.py`` so the two cannot disagree.
    """
    theta = np.asarray(theta, dtype=float)
    f = np.asarray(f, dtype=float)
    nr, nc = theta.shape
    yy, xx = np.mgrid[0:nr, 0:nc]
    m0 = float(theta.sum())
    if m0 <= 0.0:
        return float("nan"), float("nan"), 0
    cx0, cy0 = (xx * theta).sum() / m0, (yy * theta).sum() / m0
    rad = np.sqrt((xx - cx0) ** 2 + (yy - cy0) ** 2).ravel()
    order = np.argsort(rad)
    rsort = rad[order]

    def _q(img, frac):
        w = np.clip(np.asarray(img, float).ravel(), 0.0, None)[order]
        c = np.cumsum(w)
        return float(rsort[np.searchsorted(c, frac * c[-1])]) if c[-1] > 0 else float("nan")

    r0, r1 = _q(theta, 0.99), _q(f, 0.99)
    h0, h1 = _q(theta, 0.50), _q(f, 0.50)
    d = (f - theta).reshape(nr, nc)
    r2 = rad.reshape(nr, nc)
    bands = [float(d[(r2 >= lo) & (r2 < lo + 2)].sum()) for lo in range(0, int(0.55 * nr), 2)]
    flips = sum(1 for i in range(len(bands) - 1) if bands[i] * bands[i + 1] < 0)
    sup = (100.0 * (r1 - r0) / r0) if r0 > 0 else float("nan")
    half = (100.0 * (h1 - h0) / h0) if h0 > 0 else float("nan")
    return sup, half, flips
