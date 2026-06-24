# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Interactive Streamlit app for tomographic reconstruction + sensitivity-based uncertainty
quantification (UQ). It started as the research example
`sDOE_senNLP/examples/Example2/Example2_simple_uq.py` and now lets the user define the
projection geometry interactively. `README.md` is a high-level project intro (no ops — setup
lives in `Dockerfile`/`fly.toml`); this file is for working *in* the code.

## Architecture (three layers, read top-down)

1. **`app.py`** — Streamlit UI and the only entrypoint. **Two modes share one page**, laid out
   as picture (left) | dials + buttons (mid) | read-only sequence table (right):
   - **Live dose-response simulator** (runs entirely in the browser, no solver). The picture is a
     *derived view* of the table, never the solve output: the cumulative dose-response degradation
     `pixel·exp(-α·I_local − β·I_local²)` of the first `view_k` measurements (a small numpy helper,
     `_degraded_image`, reusing the vendored geometry — not a backend change). The measurement
     sequence lives in `st.session_state["beam_table"]` and is built up **only** by the **➕ Take
     measurement** / **Reset** buttons (it's a read-only `st.dataframe`, no longer an editable
     `data_editor`). **Previous/Next Measurement** scrub `view_k` over `0..N` to replay history:
     blue dashes = the viewed measurement, red dashes = the live next-measurement preview.
   - **Reconstruct** (the heavy solve; the old "Run") — converts the table to `BeamStep`s with the
     **same** `_table_to_seq` mapping the live image uses (so the solve matches the picture), then
     calls `run_simple_uq` with the live `I0/α/β/tv_weight`. Runs **only on the button press**.
   Streamlit re-executes the whole script on every widget interaction, so results are stashed in
   `st.session_state["results"]` to survive reruns without re-solving. IPOPT output streams live to
   a `log_callback` that renders a rolling tail (last 8000 chars). `IMAGE_RES = 30` is now a fixed
   module constant (was a sidebar slider); the sidebar is gone.

   - **`live_sim_component/index.html`** — a **no-build static Streamlit component** (raw
     `postMessage` bridge, no npm/React) declared via `components.declare_component(path=…)`. It
     owns the **Angle / Offset / # Beams sliders and the SVG beam overlay**, redrawing the red
     preview dashes **client-side while dragging** — Python only supplies the PNG background
     (`_live_background_uri`, a 1:1 `image_res×image_res` data-URI upscaled `pixelated` to match
     `interpolation="nearest"`) and the committed (blue) bundle. This exists because `st.slider`
     only reports on release (one server round-trip per drag = no live overlay); the component
     reports the values back on release so `Take measurement`/`Reconstruct` read them from
     `session_state`. Its geometry JS mirrors `_bundle_r_values` and must stay in sync. The old
     matplotlib version (`_live_figure`) is now **dead code** left in `app.py`.

2. **`tomography_uq.py`** — `run_simple_uq(params: UQParams, log_callback=None) -> UQResults`,
   the parameterized re-implementation of Example2. Contract it deliberately upholds: takes
   a dataclass instead of module-level constants, **never calls `plt.show()`** (returns
   matplotlib `Figure`s), renders headless (Agg), and can stream the solver log. It imports
   the physics from `senDOE` but does **not** import or modify the original example. Running
   it directly (`python3 tomography_uq.py`) is a headless smoke test with the defaults.

3. **`senDOE/`** — **vendored snapshot** of the research package (`sDOE_senNLP`), copied
   verbatim so the app is self-contained. **Do not edit these files in place** — that breaks
   provenance. To refresh, re-copy the exact 8-file import subset and bump the commit hash in
   `SENDOE_VENDOR.md`. The pieces actually used:
   - `models/tomography_pyomo_pixel_intersection.py` — builds the Pyomo model
     (`create_sample_model`, `add_beam_constraints_pyomo` — accepts arbitrary `(r, theta)`
     measurement lists — sinogram-RMSE / total-variation objective expressions). `image[ix, iy,
     time]` is the decision variable; the `time==0` slice is fixed to the phantom for the forward
     solve and freed for the inverse solve.
   - `helpers/geometry.py` — Radon transform via line / pixel-grid intersection.
   - `helpers/statistics.py` — `d_optimality` and related design criteria (log-det via
     Cholesky / eigendecomposition / stochastic-Lanczos).
   - `sensitivity/pyomo_sensitivity.py` — `extract_sensitivity_matrix` (k_aug or sipopt
     backend) for the dx/dp Jacobian.

### Pipeline inside `run_simple_uq`
User-defined geometry (`params.beam_steps`: a list of `BeamStep(angle_deg, offset, n_beams)`) →
forward IPOPT solve (simulate a sinogram from a Shepp-Logan phantom) → inverse IPOPT solve
(reconstruct `image[:, :, 0]` via sinogram RMSE + total-variation) → **k_aug** extracts
`d(image0)/d(sinogram)`; posterior covariance `= J·(σ²·I)·Jᵀ`, yielding the per-pixel
log-covariance map and the scalar **D-optimality** → a beam/measurement view of the geometry.

Geometry notes:
- Each beam step is one time index; `n_horizon = len(beam_steps) + 1` (reproduces the old
  `n_angle = n_horizon − 1`). The default `beam_steps` (9 evenly-spaced angles, full fan) are
  byte-identical to the previous `linspace` geometry, so default results are unchanged.
- Per-step rays: `r = offset + (arange(n_beams) − (n_beams−1)/2)`, with `n_beams=0` ⇒ full
  `image_res` fan. Rays with `abs(r) > image_res/2 − 0.5` are dropped — this guards the vendored
  `geometry.py` empty-intersection `IndexError`; **do not** remove that clamp.
- The classical FBP/SART baselines were removed (per-step offsets make the detector grid
  irregular, which `iradon`/`extract_sinogram_value` assume is uniform). `extract_sinogram_value`
  is now unused (left in the vendored file).
- k_aug's param list is the **full** `sinogram_data` product (`r_union × angle_union × time`),
  not just the real measurements — pre-existing behavior; the covariance self-sizes. Arbitrary
  *fractional* offsets enlarge `r_union` and thus k_aug cost (`app.py` warns); integer / 0.5-grid
  offsets reuse the shared detector grid.

## Native solver dependency (the main gotcha)

The two NLP solves and the sensitivity step need **compiled** binaries that are *not*
pip-installable: **IPOPT** (with HSL) and **k_aug** / **dot_sens**. They come from IDAES via
`idaes get-extensions`. Consequences for development:

- **You generally cannot run or test the pipeline without these binaries on `PATH`** (or
  `IPOPT_EXECUTABLE` set). The reliable path is Docker — `docker build` runs fail-fast smoke
  tests (a real IPOPT solve, `import senDOE`), so a green build means the solvers work.
- `senDOE/sensitivity/pyomo_sensitivity.py` prepends `~/.idaes/bin` to `PATH` on import so
  k_aug/dot_sens resolve. In Docker, `IDAES_DATA=/opt/idaes` installs them user-independently
  and they are symlinked into `/usr/local/bin`. `_resolve_ipopt()` searches
  `IPOPT_EXECUTABLE` → `/usr/local/bin/ipopt` → `PATH` → bare `ipopt`.
- Linear-solver fallback order is `ma27 → ma57 → mumps` (`_FALLBACK_LINEAR_SOLVERS`). The
  fallback only switches when a solver **fails to run** (e.g. missing from the build), not on
  a non-optimal termination. **`ma86` is intentionally excluded** — it is not in the IDAES
  IPOPT build used for deployment (`requirements.txt`'s comment about HSL `ma86` notwithstanding).
- `print_info_string` from Example2 is deliberately omitted — it makes the IDAES IPOPT
  3.13.2 build exit abnormally.

## Commands

Build + run (recommended):
```
docker build -t tomo-uq:local .
docker run --rm -p 8501:8501 tomo-uq:local      # http://localhost:8501
```
Headless pipeline smoke test (no browser):
```
docker run --rm tomo-uq:local python3 -c \
  "from tomography_uq import run_simple_uq, UQParams; r=run_simple_uq(UQParams()); print(r.d_optimality, r.forward_solver_status)"
```
Without Docker (needs IPOPT + k_aug on `PATH`):
```
pip install -r requirements.txt
idaes get-extensions        # installs ipopt/k_aug/dot_sens under ~/.idaes/bin
streamlit run app.py
python3 tomography_uq.py    # headless smoke test with defaults
```
Deploy is Fly.io scale-to-zero (`fly deploy`); see `README.md`.

There is **no automated test suite** — the vendored snapshot deliberately excludes
`senDOE/tests/`. The smoke tests above plus the Dockerfile build-time checks
(`import senDOE`, an end-to-end IPOPT solve) are the verification path.

## Cost / sizing

A full run at the default size (`image_res` 30, 9 beam steps) takes **minutes** — two IPOPT
solves plus a k_aug extraction. Cost scales ~ `N² · (#beam steps)`, and fractional beam offsets
further inflate the k_aug parameter count (see geometry notes); pushing any of these up can OOM
the default 2 GB Fly VM.
