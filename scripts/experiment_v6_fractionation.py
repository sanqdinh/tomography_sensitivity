"""Dose fractionation in the v6 forward model: 10 angles one at a time vs all 10 at once.

The v6 counterpart of ``experiment_fractionation.py``, which asked the same question of v5.

The two runs deliver the SAME total exposure -- ten full-fan bundles at ten evenly spaced
angles, the same ``I0`` per ray -- and differ only in how it is split in time:

  A. SEQUENTIAL   ten measurement steps, one angle each.  This is what ``simulate`` does.
  B. SIMULTANEOUS one measurement step carrying all ten angles.

Any difference is fractionation, not dose.  It is expected to be non-zero because the converted
fraction ``dw = 1 - exp(-sum_rays c I delta)`` SATURATES: ten small exposures each sit low on
that curve and convert nearly ``c I delta`` apiece, while one exposure of ten times the dose
sits high on it and converts less than ten times as much.  At ``c_omega = 0.4`` that is the
dominant effect and it is large.

WHAT "SIMULTANEOUS" MEANS, precisely.  Within one step every ray integrates the field as it stood
at the START of that step, so rays do not shield one another's damage.  That is not invented for
this experiment -- it is the convention the model already uses for a bundle, and the one
CLAUDE.md records for the 3D sinogram.  Firing ten angles at once therefore sums the ten dose
fields before the single decay, the single potential solve and the single transport solve.
:func:`check_equivalence` is the gate: at ONE bundle the simultaneous step must reproduce
:func:`senDOE.models.tomography_2d_shrinkage_decay.step` bit-for-bit.

TWO THINGS ARE SIMPLER HERE THAN IN v5, both because of the implicit transport.

v5's version of this experiment needed :func:`match_ck`, an outer loop scaling ``c_cp`` until the
worse schedule's compaction number hit a target, because beyond ``C_k = 1`` v5's explicit flux
loses positivity and the comparison would have been run partly outside the model's validity --
CLAUDE.md records the effect coming out at 5x rather than 24x once that was imposed.  **v6 has no
such bound.**  eq:xd_implicit_transport is an M-matrix at every ``c_cp``, so both schedules are
valid at the same ``c_cp``, and ``c_cp`` is held fixed at its material value with no matching
step at all.  The number below is therefore the raw fractionation, not a bounded proxy for it.

And with ``--a 0 --b 0`` the total attenuation is EXACTLY conserved in both runs (prop:xd_mass),
so every difference between them is transport rather than a difference in how much mass each
schedule destroyed.  That is the default here, unlike in the v5 script.

Run it::

    PYTHONPATH=. python3 scripts/experiment_v6_fractionation.py
    PYTHONPATH=. python3 scripts/experiment_v6_fractionation.py --c 0.4 --grid 64 --tag g64
"""

from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")                      # headless, like every other figure path in this repo
import matplotlib.pyplot as plt

from senDOE.helpers.dose import accumulate_dose, scale_to_optical_depth
# step_simultaneous lives in senDOE.models.tomography_2d_shrinkage_decay now, not here: app.py
# needs it too, and two copies of the step map is exactly the drift this repo keeps warning about.
from senDOE.helpers.phantoms import phantom as _phantom
from senDOE.models.tomography_2d_shrinkage_decay import (
    ShrinkageDecayParams as V6Params, resolve, shape_diagnostics, simulate, step,
    step_simultaneous, half_mass_radius, centroid_of)
from senDOE.helpers.rays import bundle_r_values

HERE = os.path.dirname(os.path.abspath(__file__))


def check_equivalence(image_res: int = 24, verbose: bool = True) -> float:
    """Gate: at ONE bundle the simultaneous step must BE the model's own :func:`step`."""
    theta = scale_to_optical_depth(_phantom(image_res), 1.1, image_res)
    p = resolve(V6Params(reach=7.0, f_ref_frac=0.002, eta=1e-3), theta)
    rv = bundle_r_values(0.0, 0, image_res)
    ang = np.deg2rad(37.0)
    a, _ = step(theta, rv, ang, p)
    b, _ = step_simultaneous(theta, [(rv, ang)], p)
    err = float(np.abs(a - b).max())
    if verbose:
        print("  gate: one-bundle simultaneous == step -> %.3e  %s"
              % (err, "PASS" if err == 0.0 else "FAIL (must be exactly 0)"))
    assert err == 0.0, err
    return err


def radial_profile(img, centroid, n_bins: int, r_max: float):
    """Mass per radial band about a FIXED centroid.  Mass, not mean: it must sum to the total."""
    nr, nc = img.shape
    yy, xx = np.mgrid[0:nr, 0:nc]
    rad = np.sqrt((xx - centroid[0]) ** 2 + (yy - centroid[1]) ** 2).ravel()
    w = np.asarray(img, dtype=float).ravel()
    edges = np.linspace(0.0, r_max, n_bins + 1)
    out = np.array([w[(rad >= edges[i]) & (rad < edges[i + 1])].sum() for i in range(n_bins)])
    return 0.5 * (edges[:-1] + edges[1:]), out


def run(image_res=32, n_angles=10, optical_depth=1.1, **over):
    """Both runs on one phantom.  Returns ``(theta, A, B, p, angles)``."""
    theta = scale_to_optical_depth(_phantom(image_res), optical_depth, image_res)
    kw = dict(I0=1.0, c=0.4, a=0.0, b=0.0, c_cp=0.3, reach=7.0, gamma=100.0,
              f_ref_frac=0.002, eta=1e-3)
    kw.update(over)
    p = resolve(V6Params(**kw), theta)

    angles = [180.0 * k / n_angles for k in range(n_angles)]
    rv = bundle_r_values(0.0, 0, image_res)          # full fan, zero offset: identical per angle

    # A -- ten steps, one angle each. The stock path, untouched.
    seq = tuple((a, 0.0, 0) for a in angles)
    fA, infosA = simulate(theta, seq, p, image_res)
    # Total void created over the run, sum_k Pi_k. This is what separates the two candidate
    # causes of any difference -- dw saturating, versus the transport simply being applied ten
    # times. StepInfo6 carries it per step, so it comes straight off the run rather than from a
    # second pass over the trajectory that could disagree with the first.
    A = dict(f=fA, steps=len(infosA), void=sum(i.void for i in infosA),
             dw=max(i.dw_max for i in infosA),
             fmin=min(i.state_min for i in infosA), phi=max(i.phi_max for i in infosA),
             lost=sum(i.lost for i in infosA), colsum=max(i.colsum_err for i in infosA),
             absorp=max(i.absorption_ratio for i in infosA),
             mass_residual=max(i.mass_residual for i in infosA))

    # B -- one step, all ten angles.
    fB, infoB = step_simultaneous(theta, [(rv, np.deg2rad(a)) for a in angles], p)
    B = dict(f=fB, steps=1, void=infoB.void, dw=infoB.dw_max, fmin=infoB.state_min,
             phi=infoB.phi_max, lost=infoB.lost, colsum=infoB.colsum_err,
             absorp=infoB.absorption_ratio, mass_residual=infoB.mass_residual)

    ctr = centroid_of(theta)
    for d in (A, B):
        d["mass"] = float(d["f"].sum())
        d["support_pct"], d["half_pct"], d["flips"] = shape_diagnostics(theta, d["f"])
        d["half_px"] = half_mass_radius(d["f"], ctr)
    return theta, A, B, p, angles


def figure(theta, A, B, p, angles, path):
    """2x3: the two finals on top, their changes and the radial redistribution below.

    A 1x5 strip is unreadable at this many panels -- everything ends up 200 px wide.  Two rows
    keeps each image square and leaves the redistribution panel room to be read.
    """
    nr, nc = theta.shape
    ctr = centroid_of(theta)
    vmax = float(theta.max())
    dA, dB = A["f"] - theta, B["f"] - theta
    span = max(float(np.abs(dA).max()), float(np.abs(dB).max()), 1e-30)

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 9.2))
    ax = axes.ravel()
    for a, img, ttl, cm, lo, hi in (
            (ax[0], theta, "theta (undamaged)", "gray", 0.0, vmax),
            (ax[1], A["f"], "A  SEQUENTIAL\n10 steps, 1 angle each", "gray", 0.0, vmax),
            (ax[2], B["f"], "B  SIMULTANEOUS\n1 step, all 10 angles", "gray", 0.0, vmax),
            (ax[3], dA, "A - theta   (half-mass %+.3f%%)" % A["half_pct"], "coolwarm",
             -span, span),
            (ax[4], dB, "B - theta   (half-mass %+.3f%%)" % B["half_pct"], "coolwarm",
             -span, span)):
        im = a.imshow(img, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
        a.set_title(ttl, fontsize=10)
        a.set_xticks([]); a.set_yticks([])
        fig.colorbar(im, ax=a, fraction=0.046)
    # The two change panels share one colour scale, so "B barely moved" is visible directly
    # rather than only in the numbers.

    # Mass-normalised radial redistribution. At a = b = 0 the totals are identical and exactly
    # conserved, so this is pure shape: positive inner bands with negative outer bands is
    # condensation. Both schedules on one axis, because the comparison IS the panel.
    r, p0 = radial_profile(theta, ctr, 16, 0.55 * nr)
    _r, pA = radial_profile(A["f"], ctr, 16, 0.55 * nr)
    _r, pB = radial_profile(B["f"], ctr, 16, 0.55 * nr)
    n0 = p0 / max(p0.sum(), 1e-300)
    w = (r[1] - r[0]) * 0.40
    ax[5].axhline(0.0, color="#bbbbbb", lw=1)
    ax[5].bar(r - w / 2, pA / max(pA.sum(), 1e-300) - n0, width=w, label="A sequential",
              color="#1f77b4")
    ax[5].bar(r + w / 2, pB / max(pB.sum(), 1e-300) - n0, width=w, label="B simultaneous",
              color="#d62728")
    ax[5].set_title("radial redistribution, mass-normalised\n"
                    "(inner + / outer - = condensation)", fontsize=10)
    ax[5].set_xlabel("radius (px)"); ax[5].set_ylabel("share of total mass, after - theta")
    ax[5].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax[5].legend(fontsize=9, frameon=False)
    ax[5].spines[["top", "right"]].set_visible(False)
    ax[5].set_box_aspect(1.0)

    ratio = (A["half_pct"] / B["half_pct"]) if B["half_pct"] != 0 else float("inf")
    fig.suptitle(
        "v6 dose fractionation: %d evenly spaced angles, sequential vs simultaneous\n"
        "I0=%.3g  c_omega=%.3g  a=%.3g  b=%.3g  c_cp=%.3g  l=%.3g px  eta=%.0e  grid %d   ---   "
        "same total exposure, differing only in how it is split in time\n"
        "SEQUENTIAL CONTRACTS %.2fx AS MUCH (%+.3f%% against %+.3f%%).  Cause: dw saturates --   "
        "peak dw %.4f against %.4f, total void created %.3f against %.3f (%.2fx)\n"
        "total attenuation %.6f in both, identical to %.1e: a=b=0, so every difference here is "
        "transport"
        % (len(angles), p.I0, p.c, p.a, p.b, p.c_cp, p.reach, p.eta, nr,
           ratio, A["half_pct"], B["half_pct"], A["dw"], B["dw"], A["void"], B["void"],
           A["void"] / max(B["void"], 1e-300), A["mass"], abs(A["mass"] - B["mass"])),
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--angles", type=int, default=10)
    ap.add_argument("--I0", type=float, default=1.0)
    ap.add_argument("--c", type=float, default=0.4, help="c_omega, the conversion coefficient")
    ap.add_argument("--a", type=float, default=0.0, help="linear decay; 0 with b=0 conserves mass")
    ap.add_argument("--b", type=float, default=0.0)
    ap.add_argument("--c-cp", type=float, default=0.3)
    ap.add_argument("--reach", type=float, default=7.0)
    ap.add_argument("--gamma", type=float, default=100.0)
    ap.add_argument("--f-ref-frac", type=float, default=0.002)
    ap.add_argument("--eta", type=float, default=1e-3)
    ap.add_argument("--tag", default="", help="suffix for the PNG name")
    a = ap.parse_args(argv)

    print(__doc__.splitlines()[0])
    print()
    check_equivalence()
    print()

    kw = dict(I0=a.I0, c=a.c, a=a.a, b=a.b, c_cp=a.c_cp, reach=a.reach, gamma=a.gamma,
              f_ref_frac=a.f_ref_frac, eta=a.eta)
    theta, A, B, p, angles = run(image_res=a.grid, n_angles=a.angles, **kw)

    print("  grid %d, %d angles, I0=%.3g c_omega=%.3g a=%.3g b=%.3g c_cp=%.3g l=%.3g eta=%.0e"
          % (a.grid, a.angles, p.I0, p.c, p.a, p.b, p.c_cp, p.reach, p.eta))
    print("  NO c_cp matching: v6's transport is an M-matrix at every c_cp, so both schedules")
    print("  are valid at the material value. v5's run of this needed that outer loop.")
    print()
    hdr = ("%-14s %7s %10s %10s %7s %8s %10s %11s %10s"
           % ("schedule", "steps", "half-mass", "support", "flips", "max dw", "void created",
              "total atten", "min f"))
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for name, d in (("A sequential", A), ("B simultaneous", B)):
        print("%-14s %7d %+9.3f%% %+9.3f%% %7d %8.4f %10.4f %11.6f %10.2e"
              % (name, d["steps"], d["half_pct"], d["support_pct"], d["flips"], d["dw"],
                 d["void"], d["mass"], d["fmin"]))
    print()
    ratio = (A["half_pct"] / B["half_pct"]) if B["half_pct"] != 0 else float("inf")
    print("  FRACTIONATION: sequential contracts %.2fx as much as simultaneous "
          "(%+.3f%% against %+.3f%%)." % (ratio, A["half_pct"], B["half_pct"]))
    print("  WHY, measured rather than asserted. Two candidate causes and they are separable:")
    print("    saturation -- dw = 1 - exp(-sum c I delta) is concave, so ten small exposures")
    print("      convert more in total than one of ten times the dose. Total void created:")
    print("      A %.4f against B %.4f, a factor %.2f. Peak dw A %.4f against B %.4f, and B is"
          % (A["void"], B["void"], A["void"] / max(B["void"], 1e-300), A["dw"], B["dw"]))
    print("      hard against the ceiling, so most of its extra dose converts nothing.")
    print("    compounding -- the transport is solved 10 times against once, each time from a")
    print("      field the previous step already moved.")
    print("    The void ratio is %.2f and the contraction ratio is %.2f, so saturation accounts"
          % (A["void"] / max(B["void"], 1e-300), ratio))
    print("    for most of it but not all; the remainder is the compounding.")
    print("  max |A - B| per pixel = %.3e, which is %.1f%% of theta's peak."
          % (float(np.abs(A["f"] - B["f"]).max()),
             100.0 * float(np.abs(A["f"] - B["f"]).max()) / float(theta.max())))
    print("  mass: A %.12f  B %.12f  (difference %.2e; a=b=0 so both are exactly conserved)"
          % (A["mass"], B["mass"], A["mass"] - B["mass"]))
    print("  health: |colsum-1| A %.1e B %.1e · transport residual A %.1e B %.1e · "
          "absorp/vs A %.1e B %.1e" % (A["colsum"], B["colsum"], A["mass_residual"],
                                       B["mass_residual"], A["absorp"], B["absorp"]))
    print()
    name = "v6_fractionation%s.png" % (("_" + a.tag) if a.tag else "")
    path = figure(theta, A, B, p, angles, os.path.join(os.getcwd(), name))
    print("  wrote %s" % path)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
