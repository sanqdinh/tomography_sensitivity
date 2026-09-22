"""The two v6 limits subsec:assumptions asks for and does not report.

Both are "Numerical checks" items that :func:`degrade_v6.check_invariants` deliberately leaves
out, because each is a *sweep* rather than an assertion and neither belongs in a Docker build.

1. **The conduction-threshold plateau.**  subsec:system says of ``f_ref``: "The rule is that
   ``f_ref`` be at most a fifth of the smallest interior value the phantom carries, and
   subsec:assumptions reports the plateau below which the answer stops moving."  subsec:assumptions
   reports no such plateau.  It is also not the right rule.  eq:xd_potential_solve's diagonal is
   ``varsigma + gamma(1 - sigma_p)`` with ``1 - sigma_p = exp(-ft_p/f_ref)`` inside the specimen,
   so what has to be small is not ``f_ref/f_interior`` on its own but

       gamma exp(-f_interior / f_ref) / varsigma,      varsigma = (Delta/l)^2

   A fifth leaves ``gamma e^{-5}``, which at any ordinary ``gamma`` and ``l`` is orders of
   magnitude above ``varsigma``: the vacuum penalty then sets the scale *inside* the body, the
   potential is crushed by that ratio, and the specimen does not move.  Nothing raises -- the
   operator is still a perfectly good M-matrix, the bound of eq:xd_potential_bound still holds
   with room to spare, and the answer is simply zero.  This sweep locates the plateau on a disc,
   where "the smallest interior value" is unambiguous and the spec's rule can be read literally.

2. **The small-reach limit approaches the pointwise driver.**  subsec:system: "As ``l -> 0``,
   ``phi_k ~ (l/Delta)^2 Pi_k``, so the closure approaches a pointwise form with effective
   amplitude ``c_cp (l/Delta)^2``.  A nonzero pointwise limit therefore requires corresponding
   rescaling of ``c_cp``."  This sweeps ``l`` down with ``c_cp (l/Delta)^2`` held fixed and
   measures the convergence, both of the driver and of the field it produces.

Run it::

    python3 experiment_v6_limits.py                 # both sweeps, disc, grid 64
    python3 experiment_v6_limits.py --phantom       # the same on Shepp-Logan
    python3 experiment_v6_limits.py --only reach

Nothing here is asserted: these are measurements to quote, and the numbers move with the
geometry.  The assertions live in :func:`degrade_v6.check_invariants`.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from degrade_v6 import (V6Params, simulate, resolve, compaction_potential, implicit_transport,
                        half_mass_radius, centroid_of, scale_to_optical_depth,
                        _disc, _phantom, _demo_sequence)

F_REF_FRACS = (0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 5e-4, 2e-4)
REACH_PX = (16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125)


def _theta(image_res: int, use_phantom: bool, edge: float = 2.0, depth: float = 1.1):
    """The disc is the default because its interior is ONE value, so "a fifth of the smallest
    interior value" is a number rather than a judgement call.

    ``edge`` defaults to a resolved (smoothstep) interface rather than a hard one.  The flux
    reads FACE DIFFERENCES of the driver, so a one-pixel jump in ``Pi`` keeps the two drivers'
    gradients apart long after their values have converged: on a hard disc sweep 2's rel(field)
    sits at 1.1-1.8 until l drops below a quarter pixel, while on a smoothstep disc it falls
    monotonically with rel(driver).  Same lesson as degrade_v3's ``edge``, which is not
    cosmetic either.  Pass ``--edge 0`` to see the sharp-interface behaviour.
    """
    base = _phantom(image_res) if use_phantom else _disc(image_res, edge=float(edge))
    return scale_to_optical_depth(base, depth, image_res)


def _interior_value(theta) -> float:
    """The smallest value the specimen carries, in subsec:system's sense: the lowest occupied
    level, not the lowest pixel (which is vacuum).  Taken as the 10th percentile of the pixels
    above 5% of peak, so a soft rim does not stand in for the bulk."""
    t = np.asarray(theta, dtype=float)
    occ = t[t > 0.05 * float(t.max())]
    return float(np.percentile(occ, 10.0)) if occ.size else float("nan")


def sweep_f_ref(image_res: int = 64, n_steps: int = 12, use_phantom: bool = False,
                edge: float = 2.0, **kw):
    """Sweep the conduction threshold and find where the answer stops moving."""
    theta = _theta(image_res, use_phantom, edge)
    ctr = centroid_of(theta)
    r0 = half_mass_radius(theta, ctr)
    f_int = _interior_value(theta)
    base = resolve(V6Params(**kw), theta)
    vs = base.varsigma()

    print("SWEEP 1: the conduction threshold f_ref")
    print("  %s, grid %d, %d steps, l = %.3g px (varsigma = %.3e), gamma = %.3g, c_cp = %.3g"
          % ("Shepp-Logan" if use_phantom else "disc(edge=%g)" % edge, image_res, n_steps,
             base.reach, vs, base.gamma, base.c_cp))
    print("  f_max = %.4g, smallest interior value = %.4g" % (base.f_max, f_int))
    print("  The condition is the LAST column << 1. The spec's rule is the first row.")
    print()
    print("  %-10s %-11s %12s %12s %9s   %s"
          % ("f_ref/fmax", "f_ref/f_int", "predicted", "measured", "dR_half", ""))
    print("  %-10s %-11s %12s %12s %9s"
          % ("", "", "gam*e^-r/vs", "absorp/vs", ""))
    rows = []
    for frac in F_REF_FRACS:
        p = replace(base, f_ref_frac=frac)
        f_ref = frac * base.f_max
        predicted = base.gamma * np.exp(-f_int / f_ref) / vs
        try:
            f, infos = simulate(theta, _demo_sequence(n_steps), p, image_res)
            dR = 100.0 * (half_mass_radius(f, ctr) - r0) / r0
            ratio = max(i.absorption_ratio for i in infos)
        except Exception as exc:                       # the potential's guards raise
            print("  %-10.4g %-11.3g  solve refused: %s" % (frac, f_ref / f_int, exc))
            continue
        tag = "  <- the spec's rule, and it does not move" if frac == 0.2 else ""
        print("  %-10.4g %-11.3g %12.3e %12.3e %+8.3f%%   %s"
              % (frac, f_ref / f_int, predicted, ratio, dR, tag))
        rows.append((frac, ratio, dR))

    moving = [r for r in rows if abs(r[2]) > 1e-9]
    if len(moving) >= 2:
        ref = moving[-1][2]
        plateau = [r for r in moving if abs(r[2] - ref) <= 0.02 * abs(ref)]
        if plateau:
            print()
            print("  PLATEAU: the answer is within 2%% of its small-f_ref value from "
                  "f_ref/f_max = %.4g downward (dR = %+.3f%%), where absorp/vs = %.2e."
                  % (plateau[0][0], plateau[0][2], plateau[0][1]))
    print()
    return rows


def sweep_reach(image_res: int = 64, n_steps: int = 12, use_phantom: bool = False,
                c_eff: float = 5.0, edge: float = 2.0, **kw):
    """Sweep the reach down with ``c_cp (l/Delta)^2`` fixed; the driver must approach ``Pi``."""
    theta = _theta(image_res, use_phantom, edge)
    ctr = centroid_of(theta)
    r0 = half_mass_radius(theta, ctr)
    kw.pop("c_cp", None)
    base = V6Params(**kw)

    print("SWEEP 2: the small-reach limit approaches the pointwise driver")
    print("  %s, grid %d, gamma = %.3g, f_ref/fmax = %.3g, effective amplitude "
          "c_cp*(l/D)^2 = %.3g" % ("Shepp-Logan" if use_phantom else "disc(edge=%g)" % edge,
                                   image_res, base.gamma, base.f_ref_frac, c_eff))
    print("  rel(driver) = ||phi - (l/D)^2 Pi||_inf / ||(l/D)^2 Pi||_inf.")
    print("  rel(grad)   = the same on FACE DIFFERENCES, which is what the flux actually")
    print("                reads, and is therefore what predicts rel(field).")
    print("  rel(field)  = ||f_phi - f_Pi||_inf / max(||f_phi - ft||_inf, ||f_Pi - ft||_inf),")
    print("                one step of eq:xd_implicit_transport from the same ft.")
    print("  eta is scaled as 1/c_cp so the resting rate c_cp*eta*log2 is HELD FIXED: rescaling")
    print("  c_cp alone would change the smoothing at the same time as the reach, and at l =")
    print("  0.125 px it drives the resting diffusion to 0.22 per face, which converges the two")
    print("  runs for the wrong reason.")
    print()
    print("  %-9s %-11s %-10s %12s %11s %12s"
          % ("l (px)", "c_cp", "eta", "rel(driver)", "rel(grad)", "rel(field)"))
    rows = []
    seq = _demo_sequence(n_steps)
    from degrade_v2 import accumulate_dose
    from dose_response import bundle_r_values
    ang, off, nb = seq[0]
    for ell in REACH_PX:
        c_cp = c_eff * (1.0 / ell) ** 2
        eta = base.eta * (base.c_cp / c_cp)             # hold c_cp*eta, hence the resting rate
        p = resolve(replace(base, reach=ell, c_cp=c_cp, eta=eta), theta)
        scale = 1.0 / p.varsigma()                      # (l/Delta)^2
        dq, I_p = accumulate_dose(theta, bundle_r_values(float(off), int(nb), image_res),
                                  float(np.deg2rad(float(ang))), p.I0, p.c)
        ft = theta * p.decay_factor(I_p)
        phi, Pi, _s = compaction_potential(ft, 1.0 - np.exp(-dq), p)
        ref = scale * Pi
        rel_drv = float(np.abs(phi - ref).max()) / max(float(np.abs(ref).max()), 1e-300)
        gmax = lambda a: max(float(np.abs(np.diff(a, axis=0)).max()),
                             float(np.abs(np.diff(a, axis=1)).max()))
        rel_grd = gmax(phi - ref) / max(gmax(ref), 1e-300)
        fa, _ = implicit_transport(ft, phi, p.c_cp, p.eta)
        fb, _ = implicit_transport(ft, ref, p.c_cp, p.eta)
        den = max(float(np.abs(fa - ft).max()), float(np.abs(fb - ft).max()), 1e-300)
        rel_fld = float(np.abs(fa - fb).max()) / den
        print("  %-9.4g %-11.4g %-10.3g %12.3e %11.3e %12.3e"
              % (ell, p.c_cp, p.eta, rel_drv, rel_grd, rel_fld))
        rows.append((ell, rel_drv, rel_grd, rel_fld))
    print()
    return rows


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--n-steps", type=int, default=12)
    ap.add_argument("--phantom", action="store_true",
                    help="Shepp-Logan instead of the hard disc (the disc is the default because "
                         "its 'smallest interior value' is unambiguous)")
    ap.add_argument("--only", choices=("f_ref", "reach"), default=None)
    ap.add_argument("--edge", type=float, default=2.0,
                    help="disc interface width in px. 0 is a hard edge, where the flux sees "
                         "a one-pixel jump in Pi and sweep 2's field convergence stalls "
                         "long after the driver has converged.")
    ap.add_argument("--I0", type=float, default=1.0)
    ap.add_argument("--c", type=float, default=0.1)
    ap.add_argument("--a", type=float, default=0.0, help="0 with b=0 conserves mass exactly")
    ap.add_argument("--b", type=float, default=0.0)
    ap.add_argument("--c-cp", type=float, default=0.3)
    ap.add_argument("--reach", type=float, default=7.0)
    ap.add_argument("--gamma", type=float, default=100.0)
    ap.add_argument("--f-ref-frac", type=float, default=0.002)
    ap.add_argument("--eta", type=float, default=1e-3)
    ap.add_argument("--c-eff", type=float, default=5.0,
                    help="sweep 2 holds c_cp*(l/Delta)^2 at this value")
    a = ap.parse_args(argv)

    print(__doc__.splitlines()[0])
    print()
    kw = dict(I0=a.I0, c=a.c, a=a.a, b=a.b, c_cp=a.c_cp, reach=a.reach, gamma=a.gamma,
              f_ref_frac=a.f_ref_frac, eta=a.eta)
    if a.only in (None, "f_ref"):
        sweep_f_ref(a.grid, a.n_steps, a.phantom, edge=a.edge, **kw)
    if a.only in (None, "reach"):
        sweep_reach(a.grid, a.n_steps, a.phantom, c_eff=a.c_eff, edge=a.edge, **kw)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
