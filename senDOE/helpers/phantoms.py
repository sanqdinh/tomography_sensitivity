"""Test fields and measurement sequences shared by the damage-model checks.

Pure numpy (``phantom`` also needs scikit-image).
"""

import numpy as np


def demo_sequence(n_steps: int = 12):
    """Evenly spaced full-fan projections -- the sequence the manuscript's checks use."""
    return tuple((180.0 * i / n_steps, 0.0, 0) for i in range(n_steps))


def wedge_sequence(n_steps: int = 12, span_deg: float = 20.0):
    """``n_steps`` projections crammed into a ``span_deg`` wedge -- the V7 design contrast."""
    if n_steps == 1:
        return ((0.0, 0.0, 0),)
    return tuple((span_deg * i / (n_steps - 1), 0.0, 0) for i in range(n_steps))


def disc(image_res: int, frac: float = 0.35, edge: float = 2.0):
    """A centred disc -- semi-axis ratio exactly 1.000000 when undamaged.

    V7 needs this rather than Shepp-Logan: the note's elasticity baseline is "1.0001 against
    1.0000 for a full scan", which is only meaningful against an isotropic starting shape.

    ``edge`` is the interface width in pixels, and it is **not cosmetic**.  The contraction
    mechanism needs the interface resolved over at least two pixels: the compaction potential
    ``Pi = dw * state~ / state_max`` turns over there because the density falls by a factor of
    ~2 across the interface while ``dw`` changes by ~2%, so ``Pi`` peaks just INSIDE the rim and
    material outside that peak drains inward.  On a perfectly sharp interface (``edge = 0``) the
    peak sits at the outermost occupied pixel and the result is rim brightening, not contraction.
    Measured at grid 64, c_cp = 0.8, a = b = 0, change in support radius::

        edge = 0 px   +0.004 %      edge = 2 px   -0.414 %      edge = 4 px   -0.552 %

    That is (S4) of the note doing real work.  ``edge = 0`` is kept reachable so the test can
    exhibit the failure mode rather than merely warn about it.
    """
    yy, xx = np.mgrid[0:image_res, 0:image_res]
    c = (image_res - 1) / 2.0
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    R = image_res * frac
    if edge <= 0.0:
        return (r <= R).astype(float)
    t = np.clip((R - r) / edge + 0.5, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)                 # smoothstep: C1 across the interface


def phantom(image_res: int):
    """The Shepp-Logan phantom resized to ``image_res x image_res``."""
    from skimage.data import shepp_logan_phantom
    from skimage.transform import resize
    return resize(shepp_logan_phantom(), (image_res, image_res)).astype(float)
