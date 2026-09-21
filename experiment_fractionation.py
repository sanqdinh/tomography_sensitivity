"""Dose fractionation in the v5 forward model: 10 angles one at a time vs all 10 at once.

The two runs deliver the SAME total exposure -- ten full-fan bundles at ten evenly spaced
angles, the same ``I0`` per ray -- and differ only in how it is split in time:

  A. SEQUENTIAL   ten measurement steps, one angle each.  This is what ``simulate`` does.
  B. SIMULTANEOUS one measurement step carrying all ten angles.

Any difference between them is fractionation, not dose.  It is expected to be non-zero because
every channel in the step map is nonlinear in the per-step fluence: the converted fraction
``dw = 1 - exp(-c I delta)`` saturates, the decay ``exp(-a I - b I^2)`` compounds, and the
compaction flux is driven by a potential solved once per step.

WHAT "SIMULTANEOUS" MEANS HERE, precisely.  Within one step every ray integrates the field as it
stood at the START of that step, so rays do not shield one another's damage.  That is not an
approximation invented for this experiment -- it is the convention the model already uses for a
bundle, and the one CLAUDE.md records for the 3D sinogram ("rays within one measurement do not
see each other's damage").  Firing ten angles at once therefore just sums the ten dose fields
before the single decay and the single compaction, which is what simultaneous irradiation is.

:func:`check_equivalence` is the gate: with ONE bundle, the simultaneous step must reproduce
:func:`degrade_v5.step` bit-for-bit.  It does (0.0e+00), so the generalisation adds nothing and
removes nothing at K=1, and the whole difference measured below is the fractionation.

Run it::

    python3 experiment_fractionation.py                      # defaults
    python3 experiment_fractionation.py --a 0 --c 0.8 --tag a0_c0.8

``--a 0 --b 0`` is the clean shrinkage case: the decay channel is off, :func:`prop:xd_mass`
makes the total EXACTLY conserved, and every difference between the two runs is then transport
alone rather than transport plus a difference in how much mass each schedule destroyed.
"""

from __future__ import annotations

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")                      # headless, like every other figure path in this repo
import matplotlib.pyplot as plt

from degrade_v2 import accumulate_dose, scale_to_optical_depth
from degrade_v5 import (V5Params, _phantom, compaction_number, compaction_potential,
                        flux_divergence, resolve, shape_diagnostics, simulate, step)

from dose_response import bundle_r_values

HERE = os.path.dirname(os.path.abspath(__file__))


def step_simultaneous(f, bundles, p: V5Params):
    """One step carrying several ``(r_values, angle_rad)`` bundles at once.

    Mirrors :func:`degrade_v5.step` line for line; the only change is that steps 1-2 accumulate
    over every bundle against the SAME ``f`` before step 3 decays it and step 4 solves one
    potential.  Returns ``(f_next, info)`` with ``info`` a plain dict.
    """
    f = np.asarray(f, dtype=float)
    cIdelta = np.zeros_like(f)
    I_p = np.zeros_like(f)
    for r_values, angle_rad in bundles:                                  # 1
        a, b = accumulate_dose(f, r_values, angle_rad, p.I0, p.c)
        cIdelta = cIdelta + a
        I_p = I_p + b
    dw = 1.0 - np.exp(-cIdelta)                                          # 2
    ft = f * p.decay_factor(I_p)                                         # 3
    lost = float(f.sum() - ft.sum())

    phi, _Pi, _sigma = compaction_potential(ft, dw, p)                   # 4a
    div, (Fh, Fv) = flux_divergence(ft, phi, p.c_cp, p.flux, p.beta, p.eps_h)   # 4b
    f_next = ft - div                                                    # 5

    info = dict(mass=float(f_next.sum()), lost=lost, dw_max=float(dw.max()),
                I_max=float(I_p.max()), state_min=float(f_next.min()),
                compaction=compaction_number(ft, phi, p.c_cp),
                flux_sum=float(Fh.sum() + Fv.sum()), phi_max=float(phi.max()))
    return f_next, info


def check_equivalence(image_res: int = 24, verbose: bool = True) -> float:
    """Gate: at ONE bundle the simultaneous step must BE :func:`degrade_v5.step`."""
    theta = scale_to_optical_depth(_phantom(image_res), 1.1, image_res)
    p = resolve(V5Params(reach=7.0, f_ref_frac=0.002), theta)
    rv = bundle_r_values(0.0, 0, image_res)
    ang = np.deg2rad(37.0)
    a, _ = step(theta, rv, ang, p)
    b, _ = step_simultaneous(theta, [(rv, ang)], p)
    err = float(np.abs(a - b).max())
    if verbose:
        print("  gate: one-bundle simultaneous == degrade_v5.step -> %.3e  %s"
              % (err, "PASS" if err == 0.0 else "FAIL (must be exactly 0)"))
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
    """Both runs on one phantom.  Returns ``(theta, resultA, resultB, params)``."""
    theta = scale_to_optical_depth(_phantom(image_res), optical_depth, image_res)
    kw = dict(I0=1.0, c=0.1, a=0.05, b=0.0, c_cp=0.3, reach=7.0, gamma=100.0,
              f_ref_frac=0.002, beta=1000.0)
    kw.update(over)
    p = resolve(V5Params(**kw), theta)

    angles = [180.0 * k / n_angles for k in range(n_angles)]
    rv = bundle_r_values(0.0, 0, image_res)          # full fan, zero offset: identical per angle

    # A -- ten steps, one angle each. The stock path, untouched.
    seq = tuple((a, 0.0, 0) for a in angles)
    fA, infosA = simulate(theta, seq, p, image_res)
    A = dict(f=fA, steps=len(infosA),
             ck=max(i.compaction for i in infosA), dw=max(i.dw_max for i in infosA),
             fmin=min(i.state_min for i in infosA), phi=max(i.phi_max for i in infosA),
             lost=sum(i.lost for i in infosA))

    # B -- one step, all ten angles.
    fB, infoB = step_simultaneous(theta, [(rv, np.deg2rad(a)) for a in angles], p)
    B = dict(f=fB, steps=1, ck=infoB["compaction"], dw=infoB["dw_max"],
             fmin=infoB["state_min"], phi=infoB["phi_max"], lost=infoB["lost"])

    for d in (A, B):
        d["mass"] = float(d["f"].sum())
        d["support_pct"], d["half_pct"], d["flips"] = shape_diagnostics(theta, d["f"])
    return theta, A, B, p, angles


def match_ck(target, image_res, n_angles, kw, tol=2e-3, max_iter=25, verbose=True):
    """Scale ``c_cp`` until the WORSE of the two schedules hits ``target`` for max C_k.

    One ``c_cp`` for both runs, not one each: it is a material property, and the experiment
    varies the schedule, not the material.  Matching per-run would confound the comparison with
    a different compaction amplitude, which is exactly what
    :func:`degrade_v5.match_compaction_number` warns against.

    The simultaneous run binds, and there C_k is EXACTLY linear in ``c_cp`` -- within one step
    ``phi`` is solved from ``ft`` and ``dw`` before the flux, so it does not see ``c_cp`` at all.
    Sequentially the feedback through ``f`` makes it only nearly linear, so this iterates the
    fixed point ``c <- c * target / C_k(c)`` rather than assuming one shot is enough.
    """
    c = float(kw.get("c_cp", 0.3))
    for it in range(1, max_iter + 1):
        k = dict(kw); k["c_cp"] = c
        _t, A, B, _p, _a = run(image_res=image_res, n_angles=n_angles, **k)
        worst = max(A["ck"], B["ck"])
        if verbose:
            print("    match c_cp=%.5f -> C_k seq %.4f / sim %.4f (worst %.4f)"
                  % (c, A["ck"], B["ck"], worst))
        if abs(worst - target) <= tol:
            return c
        c *= target / max(worst, 1e-12)
    return c


def _figure(theta, res, title, path, vmax, span, prof_ref, dlim):
    """Four panels.  Colour scales are passed in so the two figures are directly comparable."""
    f = res["f"]
    nr, nc = theta.shape
    yy, xx = np.mgrid[0:nr, 0:nc]
    m0 = float(theta.sum())
    cen = ((xx * theta).sum() / m0, (yy * theta).sum() / m0)   # FIXED initial centroid
    r, prof = radial_profile(f, cen, 16, 0.55 * nr)
    r0, prof0 = prof_ref

    fig, ax = plt.subplots(1, 4, figsize=(17.5, 4.4))
    im = ax[0].imshow(theta, cmap="gray", vmin=0.0, vmax=vmax, interpolation="nearest")
    ax[0].set_title("theta (undamaged)", fontsize=10); fig.colorbar(im, ax=ax[0], fraction=0.046)
    im = ax[1].imshow(f, cmap="gray", vmin=0.0, vmax=vmax, interpolation="nearest")
    ax[1].set_title("f after exposure", fontsize=10); fig.colorbar(im, ax=ax[1], fraction=0.046)
    im = ax[2].imshow(f - theta, cmap="coolwarm", vmin=-span, vmax=span, interpolation="nearest")
    ax[2].set_title("change  f - theta", fontsize=10); fig.colorbar(im, ax=ax[2], fraction=0.046)
    for a in ax[:3]:
        a.set_xticks([]); a.set_yticks([])

    # NORMALISED by total mass, and shown as a difference. At a = 0.05 the decay removes ~39% of
    # the mass uniformly, which swamps the transport in an absolute profile and paints the whole
    # change panel one colour. Dividing by the total removes the decay and leaves the SHAPE
    # change, which is what the compaction does and what this experiment is about. Positive
    # inner bands with negative outer bands is condensation.
    d = prof / max(prof.sum(), 1e-300) - prof0 / max(prof0.sum(), 1e-300)
    ax[3].axhline(0.0, color="#bbbbbb", lw=1)
    ax[3].bar(r, d, width=(r[1] - r[0]) * 0.85,
              color=["#1f77b4" if v >= 0 else "#d62728" for v in d])
    ax[3].set_title("radial redistribution, mass-normalised\n"
                    "(inner + / outer - = condensation)", fontsize=10)
    ax[3].set_xlabel("radius (px)"); ax[3].set_ylabel("share of total mass, after - theta")
    ax[3].set_ylim(-dlim, dlim)          # SHARED across both figures, so they compare by eye
    ax[3].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax[3].spines[["top", "right"]].set_visible(False)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--angles", type=int, default=10)
    ap.add_argument("--I0", type=float, default=1.0)
    ap.add_argument("--c", type=float, default=0.1, help="c_omega, the conversion coefficient")
    ap.add_argument("--a", type=float, default=0.05, help="linear decay; 0 with b=0 conserves mass")
    ap.add_argument("--b", type=float, default=0.0)
    ap.add_argument("--c-cp", type=float, default=0.3)
    ap.add_argument("--reach", type=float, default=7.0)
    ap.add_argument("--gamma", type=float, default=100.0)
    ap.add_argument("--f-ref-frac", type=float, default=0.002)
    ap.add_argument("--match-ck", type=float, default=None, metavar="TARGET",
                    help="solve for the c_cp that puts the WORSE schedule's max C_k at TARGET, "
                         "and use that one c_cp for both runs. Use to stay inside the donor-cell "
                         "positivity bound (C_k <= 1), which the simultaneous schedule breaches "
                         "at c_cp = 0.3 with c_omega = 0.8.")
    ap.add_argument("--tag", default="", help="suffix for the PNG names, so runs do not overwrite")
    a = ap.parse_args(argv)

    print(__doc__.splitlines()[0])
    print()
    check_equivalence()
    print()

    kw = dict(I0=a.I0, c=a.c, a=a.a, b=a.b, c_cp=a.c_cp, reach=a.reach, gamma=a.gamma,
              f_ref_frac=a.f_ref_frac)
    if a.match_ck is not None:
        print("  matching c_cp so the worse schedule's max C_k = %.3f" % a.match_ck)
        kw["c_cp"] = match_ck(a.match_ck, a.grid, a.angles, kw)
        print("    -> c_cp = %.5f (was %.5f), applied to BOTH runs\n" % (kw["c_cp"], a.c_cp))
    theta, A, B, p, angles = run(image_res=a.grid, n_angles=a.angles, **kw)
    sfx = ("_" + a.tag) if a.tag else ""
    nr, _ = theta.shape
    yy, xx = np.mgrid[0:nr, 0:nr]
    m0 = float(theta.sum())
    cen = ((xx * theta).sum() / m0, (yy * theta).sum() / m0)
    prof_ref = radial_profile(theta, cen, 16, 0.55 * nr)

    vmax = float(theta.max())
    span = max(float(np.abs(A["f"] - theta).max()), float(np.abs(B["f"] - theta).max()))

    def _nd(f):
        _r, pr = radial_profile(f, cen, 16, 0.55 * nr)
        return pr / max(pr.sum(), 1e-300) - prof_ref[1] / max(prof_ref[1].sum(), 1e-300)
    dlim = 1.05 * max(float(np.abs(_nd(A["f"])).max()), float(np.abs(_nd(B["f"])).max()))

    hdr = ("grid %d, %d evenly spaced angles, full fan | I0=%g c=%g a=%g b=%g c_cp=%g "
           "reach=%g gamma=%g f_ref_frac=%g"
           % (nr, len(angles), p.I0, p.c, p.a, p.b, p.c_cp, p.reach, p.gamma, p.f_ref_frac))
    pA = _figure(theta, A, "A. SEQUENTIAL - %d steps, one angle each\n%s" % (A["steps"], hdr),
                 os.path.join(HERE, "v5_fractionation_sequential%s.png" % sfx),
                 vmax, span, prof_ref, dlim)
    pB = _figure(theta, B, "B. SIMULTANEOUS - 1 step carrying all %d angles\n%s"
                 % (len(angles), hdr),
                 os.path.join(HERE, "v5_fractionation_simultaneous%s.png" % sfx),
                 vmax, span, prof_ref, dlim)

    print("  %-26s %14s %14s %14s" % ("", "A sequential", "B simultaneous", "B - A"))
    rows = [("measurement steps", "steps", "%d", 0),
            ("total mass", "mass", "%.5f", 1),
            ("mass lost to decay", "lost", "%.5f", 1),
            ("support radius 99% (%)", "support_pct", "%+.3f", 1),
            ("half-mass radius (%)", "half_pct", "%+.3f", 1),
            ("radial sign changes", "flips", "%d", 0),
            ("max C_k (bound <= 1)", "ck", "%.4f", 1),
            ("max dw", "dw", "%.4f", 1),
            ("max phi", "phi", "%.4f", 1),
            ("min f", "fmin", "%.3e", 1)]
    for lab, key, fmt, diff in rows:
        a, b = A[key], B[key]
        d = (fmt % (b - a)) if diff else ("%+d" % (b - a))
        print("  %-26s %14s %14s %14s" % (lab, fmt % a, fmt % b, d))

    dif = B["f"] - A["f"]
    print()
    if p.a == 0.0 and p.b == 0.0:
        m0 = float(theta.sum())
        print("  a = b = 0, so prop:xd_mass makes the total EXACTLY conserved and every")
        print("  difference below is TRANSPORT alone:")
        print("    mass drift  sequential %+.3e   simultaneous %+.3e   (relative to %.5f)"
              % (A["mass"] - m0, B["mass"] - m0, m0))
    if max(A["ck"], B["ck"]) > 1.0:
        print("  WARNING C_k > 1: the donor-cell positivity bound no longer holds "
              "(sequential %.3f, simultaneous %.3f). Check min f." % (A["ck"], B["ck"]))
    print("  final fields differ by %.4e max abs = %.2f%% of peak theta"
          % (np.abs(dif).max(), 100.0 * np.abs(dif).max() / vmax))
    print("  mass A -> B: %.5f -> %.5f (%+.3f%%)"
          % (A["mass"], B["mass"], 100.0 * (B["mass"] - A["mass"]) / A["mass"]))
    print()
    print("  wrote %s" % pA)
    print("  wrote %s" % pB)
    return 0


if __name__ == "__main__":
    sys.exit(main())
