"""Row / column / objective-gradient scaling report for a Pyomo NLP, via PyNumero.

Model-agnostic on purpose: it collects and names, and asserts nothing. The per-model gate
supplies the judgement (see ``check_scaling`` in
:mod:`senDOE.models.tomography_pyomo_2d_shrinkage_decay`).
"""

import numpy as np


def scaling_report(m, *, tiny: float = 1e-30, extra=None) -> dict:
    """Row / column / objective-gradient inf-norms of a Pyomo NLP **at its current point**.

    Everything IPOPT's own ``gradient-based`` scaling looks at, plus the two things it does not:
    the per-row *minimum* over structurally present entries (so a row that is numerically
    rank-deficient is visible), and the count of entries below ``tiny``.

    Returns ``{}`` with ``skipped`` set when PyNumero's ASL extension is unavailable, so a caller
    can gate on it rather than crash.

    ``extra(nlp, J, cons, varz) -> dict`` is merged in, which is how a model-specific gate adds
    its own rows without building a second ``PyomoNLP``.

    CAVEAT, and it belongs in every reading of the output: this is ONE point. Row medians and the
    geometry-driven spreads were invariant over 60 iterations of a real solve, but ``c_sig``'s
    within-row spread swung 200 decades along the same run. A start-point report is a tripwire,
    not a certificate.
    """
    out = {"ok": True, "failures": []}
    try:
        from pyomo.contrib.pynumero.asl import AmplInterface
        if not AmplInterface.available():
            return {"ok": True, "skipped": "pynumero ASL (libpynumero_ASL.so) unavailable"}
        from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
    except Exception as exc:
        return {"ok": True, "skipped": "pynumero unavailable: %s" % str(exc)[:120]}

    # PyomoNLP REQUIRES an objective -- a model straight out of build_*_model has none and raises
    # NotImplementedError. The caller must have added one.
    nlp = PyomoNLP(m)
    J = nlp.evaluate_jacobian().tocsr()
    g = np.asarray(nlp.evaluate_grad_objective(), dtype=float)
    cons, varz = nlp.get_pyomo_constraints(), nlp.get_pyomo_variables()

    row_max = np.abs(J).max(axis=1).toarray().ravel()
    # Per-row min over STRUCTURALLY PRESENT entries. sparse .min() returns 0 for any row with an
    # implicit zero, which is every row here, so it has to come off the indptr slices. Explicit
    # zeros are kept deliberately: an exactly-zero derivative that is structurally present is
    # precisely the rank-deficiency this exists to catch.
    row_min = np.zeros_like(row_max)
    row_tiny = np.zeros(J.shape[0], dtype=int)
    for i in range(J.shape[0]):
        d = np.abs(J.data[J.indptr[i]:J.indptr[i + 1]])
        row_min[i] = d.min() if d.size else 0.0
        row_tiny[i] = int((d < tiny).sum())
    col_max = np.abs(J).max(axis=0).toarray().ravel()

    blocks, cols = {}, {}
    for i, c in enumerate(cons):
        blocks.setdefault(c.parent_component().name, []).append(i)
    for i, v in enumerate(varz):
        cols.setdefault(v.parent_component().name, []).append(i)

    def _stat(idx, hi, lo=None, tn=None):
        h = hi[idx]
        d = {"rows": len(idx), "min": float(h.min()), "med": float(np.median(h)),
             "max": float(h.max())}
        if lo is not None:
            nz = lo[idx] > 0
            d["spread"] = float((h[nz] / lo[idx][nz]).max()) if nz.any() else float("inf")
            d["tiny"] = int(tn[idx].sum())
        return d

    out.update(
        n_vars=int(nlp.n_primals()), n_cons=int(nlp.n_constraints()), nnz=int(J.nnz),
        blocks={k: _stat(np.array(v), row_max, row_min, row_tiny) for k, v in blocks.items()},
        cols={k: _stat(np.array(v), col_max) for k, v in cols.items()},
        row_med=float(np.median(row_max)), row_min=float(row_max.min()),
        row_max=float(row_max.max()),
        frac_row_one=float(np.mean(row_max == 1.0)),
        empty_rows=int(sum(1 for i in range(J.shape[0])
                           if J.indptr[i + 1] == J.indptr[i])),
        zero_rows=int((row_max == 0.0).sum()),
        dead_cols=int(((col_max == 0.0) & (np.abs(g) == 0.0)).sum()),
        grad_inf=float(np.abs(g).max()), grad_nnz=int((g != 0.0).sum()),
    )
    out["row_min_block"] = min(out["blocks"], key=lambda k: out["blocks"][k]["min"])
    out["row_max_block"] = max(out["blocks"], key=lambda k: out["blocks"][k]["max"])
    out["grad_over_row_med"] = out["grad_inf"] / max(out["row_med"], 1e-300)
    out["grad_over_row_max"] = out["grad_inf"] / max(out["row_max"], 1e-300)
    # IPOPT's gradient-based objective factor: min(1, 100/||grad f||). It CAPS, never lifts, so a
    # value of exactly 1 means IPOPT will not rescale this objective at all.
    out["ipopt_df"] = float(min(1.0, 100.0 / max(out["grad_inf"], 1e-300)))
    if extra is not None:
        out.update(extra(nlp, J, cons, varz))
    return out
