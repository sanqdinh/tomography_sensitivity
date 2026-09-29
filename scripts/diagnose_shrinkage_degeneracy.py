"""Read-only rank diagnostics for the shrinkage-decay Pyomo reconstruction model."""

from __future__ import annotations

import argparse

import numpy as np
from scipy.linalg import svd

from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP

from senDOE.helpers.dose import scale_to_optical_depth
from senDOE.helpers.phantoms import demo_sequence as _demo_sequence, phantom as _phantom
from senDOE.models.tomography_2d_shrinkage_decay import (ShrinkageDecayParams, resolve,
                                                      select_eta, simulate, simulate_simultaneous)
from senDOE.models.tomography_pyomo_2d_shrinkage_decay import (
    add_estimation_objective, build_shrinkage_decay_model, initialize_from_numpy)
from senDOE.helpers.solvers import solve_with_fallback


def _component(name: str) -> str:
    return name.split("[", 1)[0]


def diagnose(image_res: int, n_steps: int, simultaneous: bool, eta_arg: float | None,
             continuation: bool, solve: bool) -> None:
    theta = scale_to_optical_depth(_phantom(image_res), 1.1, image_res)
    seq = _demo_sequence(n_steps)
    p0 = ShrinkageDecayParams(I0=0.0, c_cp=0.0) if continuation else ShrinkageDecayParams()
    if eta_arg is None:
        eta, eta_info = select_eta(theta, seq, p0, image_res, simultaneous=simultaneous)
    else:
        eta, eta_info = eta_arg, {}
    p = resolve(ShrinkageDecayParams(I0=p0.I0, c_cp=p0.c_cp, eta=eta), theta)
    run = simulate_simultaneous if simultaneous else simulate
    _, _, y = run(theta, seq, p, image_res, record_observations=True)

    m = build_shrinkage_decay_model(theta, seq, p, image_res, f_bounds=(0.0, None),
                       potential=not continuation, simultaneous=simultaneous)
    add_estimation_objective(m, y, 0.001, float(np.abs(theta).max()))
    residual = initialize_from_numpy(m, theta)
    if solve:
        result, solver = solve_with_fallback(m, linear_solver="ma97", max_iter=1000, tee=False)
        print("  solve=%s (%s)" % (result.solver.termination_condition, solver))
    nlp = PyomoNLP(m)
    jac = nlp.evaluate_jacobian().toarray()
    cons = nlp.get_pyomo_constraints()
    vars_ = nlp.get_pyomo_variables()

    # Full dense SVD is intentional: this script is for small diagnostic instances where an
    # actual numerical rank, and the singular vectors identifying bad blocks, are more useful
    # than a structural sparsity rank.
    u, singular, vh = svd(jac, full_matrices=False, check_finite=True)
    scale = singular[0] if singular.size else 1.0
    rel = singular / scale
    eps_rank = np.finfo(float).eps * max(jac.shape)
    rank = int(np.count_nonzero(rel > eps_rank))

    print("shrinkage-decay reconstruction Jacobian")
    print("  grid=%d steps=%d schedule=%s model=%s eta=%.6g forward_ratio=%s" % (
        image_res, n_steps, "simultaneous" if simultaneous else "sequential",
        "continuation" if continuation else "full", eta,
        "%.3g" % eta_info.get("ratio", float("nan"))))
    print("  variables=%d constraints=%d expected_free_theta=%d init_residual=%.3e" % (
        jac.shape[1], jac.shape[0], image_res * image_res, residual))
    print("  rank=%d row_deficiency=%d sigma_max=%.3e sigma_min=%.3e condition=%.3e" % (
        rank, jac.shape[0] - rank, singular[0], singular[-1],
        singular[0] / max(singular[-1], np.finfo(float).tiny)))
    print("  ten smallest relative singular values:",
          " ".join("%.3e" % x for x in rel[-10:]))

    # Attribute the weakest left/right singular directions to constraint and variable blocks.
    for label, vec, objects in (("left/constraints", u[:, -1], cons),
                                ("right/variables", vh[-1], vars_)):
        energy = {}
        for value, obj in zip(vec, objects):
            key = _component(obj.name)
            energy[key] = energy.get(key, 0.0) + float(value * value)
        top = sorted(energy.items(), key=lambda item: item[1], reverse=True)[:8]
        print("  weakest %s:" % label,
              ", ".join("%s %.1f%%" % (name, 100.0 * value) for name, value in top))

    # c_sp_pair has one derivative for each directed rate. A tiny member means that row has
    # numerically forgotten one rate even though its infinity norm remains large. The relaxed
    # -eta floor keeps the physical zero-rate limit away from an active bound.
    pair_ratios = []
    pair_zeros = 0
    for i, con in enumerate(cons):
        if con.parent_component().name != "c_sp_pair":
            continue
        values = np.abs(jac[i])
        values = values[values != 0.0]
        if len(values) < 2:
            pair_zeros += 1
        else:
            pair_ratios.append(float(np.exp(min(np.log(values.max()) - np.log(values.min()),
                                                np.log(np.finfo(float).max)))))
    print("  c_sp_pair: rows=%d missing-one-derivative=%d max derivative ratio=%s" % (
        len(pair_ratios) + pair_zeros, pair_zeros,
        "%.3e" % max(pair_ratios) if pair_ratios else "n/a"))

    if m.has_potential:
        sig = np.array([m.sig[q, k].value for q in m.PIX for k in m.TM])
        sig_ub = next(iter(m.sig.values())).ub
        print("  sig: exactly_one=%d within_1e-12_of_one=%d upper_bound=%.12g min_slack=%.3e" % (
            int(np.count_nonzero(sig == 1.0)), int(np.count_nonzero(np.abs(sig - 1.0) <= 1e-12)),
            sig_ub, float(sig_ub - sig.max())))

    # Equality-only rank misses LICQ failures created by an active variable bound. Add unit rows
    # for bounds at several numerical activity tolerances and identify which variable blocks make
    # the augmented Jacobian rank-deficient.
    x = np.asarray(nlp.get_primals(), dtype=float)
    for active_tol in (0.0, 1e-14, 1e-10, 1e-8):
        active = []
        for col, (value, var) in enumerate(zip(x, vars_)):
            if var.lb is not None and value - float(var.lb) <= active_tol:
                active.append((col, var, "lb"))
            if var.ub is not None and float(var.ub) - value <= active_tol:
                active.append((col, var, "ub"))
        if active:
            bounds = np.zeros((len(active), jac.shape[1]))
            bounds[np.arange(len(active)), [item[0] for item in active]] = 1.0
            augmented = np.vstack((jac, bounds))
            aug_sv = svd(augmented, compute_uv=False, check_finite=True)
            aug_rank = int(np.count_nonzero(
                aug_sv / aug_sv[0] > np.finfo(float).eps * max(augmented.shape)))
            by_block = {}
            for _col, var, side in active:
                key = "%s.%s" % (_component(var.name), side)
                by_block[key] = by_block.get(key, 0) + 1
            print("  active bounds tol=%.0e: %d, augmented row deficiency=%d (%s)" % (
                active_tol, len(active), augmented.shape[0] - aug_rank,
                ", ".join("%s=%d" % item for item in sorted(by_block.items()))))

    # Count theta variables that no observation can influence at first order through the full
    # equality system. For a full-row-rank J, solve J_state dx_state = -J_theta one column at a
    # time implicitly via a null-space basis. The right null space must have exactly npix columns;
    # its projection onto theta must be nonsingular for theta to parameterise all feasible motion.
    _, _, vh_full = svd(jac, full_matrices=True, check_finite=True)
    null = vh_full[rank:].T
    theta_idx = [i for i, var in enumerate(vars_) if var.parent_component().name == "f"
                 and var.index()[1] == 0]
    theta_projection = null[theta_idx, :]
    theta_sv = svd(theta_projection, compute_uv=False, check_finite=True)
    print("  feasible-nullity=%d theta-projection rank=%d/%d min_sv=%.3e" % (
        null.shape[1], int(np.count_nonzero(theta_sv > np.finfo(float).eps * max(theta_projection.shape)
                                           * theta_sv[0])), len(theta_idx), theta_sv[-1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-res", type=int, default=6)
    parser.add_argument("--n-steps", type=int, default=5)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--eta", type=float)
    parser.add_argument("--continuation", action="store_true")
    parser.add_argument("--solve", action="store_true")
    args = parser.parse_args()
    diagnose(args.image_res, args.n_steps, not args.sequential, args.eta, args.continuation,
             args.solve)
