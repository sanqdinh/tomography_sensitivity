# Vendored `senDOE` snapshot

The `senDOE/` package in this repo is a **vendored snapshot** — a minimal subset of the
research package copied verbatim so this app is self-contained and deployable without the
original repo on `PYTHONPATH`.

- **Source repo:** `sDOE_senNLP` (sibling checkout `~/sandbox/sDOE_senNLP`)
- **Source commit:** `1036434a151f1530d495dfd7bbe20b7002f31309`
  (`2026-06-10` — "tuning the degradation")
- **Source path:** `sDOE_senNLP/senDOE/`

## Files vendored (exact import-graph subset for `Example2_simple_uq.py`)

```
senDOE/__init__.py
senDOE/models/__init__.py
senDOE/models/tomography_pyomo_pixel_intersection.py
senDOE/helpers/__init__.py
senDOE/helpers/geometry.py
senDOE/helpers/statistics.py
senDOE/sensitivity/__init__.py
senDOE/sensitivity/pyomo_sensitivity.py
```

Deliberately **not** vendored (not in the import chain): `senDOE/tests/`, `doe.py`,
`mpc.py`, the other `models/tomography*.py` variants, and `helpers/transformers.py`.

## Updating the snapshot

To refresh against a newer commit of the source repo, re-copy the same 8 files verbatim and
update the commit hash above. Do not edit the vendored files in place — keep them identical
to the source so provenance stays clean.
