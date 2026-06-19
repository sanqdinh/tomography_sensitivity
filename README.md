# tomography_sensitivity

Interactive Streamlit web app for tomographic reconstruction + sensitivity-based
uncertainty quantification. It wraps the research example
`sDOE_senNLP/examples/Example2/Example2_simple_uq.py` so the parameters become live
controls and the results render in the browser.

## What it does

On each run it:

1. **Forward solve** — an IPOPT NLP simulates measurements (a sinogram) from a Shepp-Logan
   phantom.
2. **Inverse solve** — a second IPOPT NLP reconstructs the initial image (sinogram RMSE +
   total-variation regularization).
3. **Classical baselines** — FBP (`iradon`) and SART reconstructions for comparison.
4. **Uncertainty quantification** — `k_aug` extracts the sensitivity Jacobian
   d(image)/d(sinogram); the posterior covariance is `J · (σ²·I) · Jᵀ`, yielding a per-pixel
   log-covariance map and the scalar **D-optimality** criterion.

It produces six figures (phantom, NLP reconstruction, covariance map, merged sinogram, FBP,
SART) and the D-optimality value.

> ⏱️ A full run at the default size (30×30, 9 angles) takes minutes — two IPOPT solves plus a
> k_aug sensitivity extraction. Keep the browser tab open while it runs.

## Native dependencies (important)

This app needs **compiled** solver binaries that are *not* pip-installable on their own:

- **IPOPT** (with HSL `ma86`) for the two NLP solves
- **k_aug** for the sensitivity extraction

Both come from IDAES via `idaes get-extensions` (the `Dockerfile` does this automatically).
Running outside Docker requires those binaries on `PATH` (or `IPOPT_EXECUTABLE` set). The
linear solver is configurable and falls back `ma86 → ma57 → ma27 → mumps` if one isn't
available.

The `senDOE/` package here is a **vendored snapshot** of the research code — see
[`SENDOE_VENDOR.md`](./SENDOE_VENDOR.md).

## Run locally with Docker (recommended)

```bash
docker build -t tomo-uq:local .
docker run --rm -p 8501:8501 tomo-uq:local
# open http://localhost:8501
```

The build runs fail-fast smoke tests (IPOPT `--version`, `import senDOE`, and a tiny IPOPT
solve), so if `docker build` succeeds the native solvers work in the image.

### Headless pipeline smoke test

```bash
docker run --rm tomo-uq:local \
  python3 -c "from tomography_uq import run_simple_uq, UQParams; \
r=run_simple_uq(UQParams()); print(r.d_optimality, r.forward_solver_status)"
```

## Run locally without Docker

Requires Python 3.10+, the packages in `requirements.txt`, and IPOPT + k_aug on `PATH`
(e.g. via `idaes get-extensions`).

```bash
pip install -r requirements.txt
idaes get-extensions          # installs ipopt/k_aug/dot_sens under ~/.idaes/bin
streamlit run app.py
```

## Deploy to Fly.io (on-demand / scale-to-zero)

`fly.toml` is configured for `auto_stop_machines` / `auto_start_machines` with
`min_machines_running = 0` — the machine boots on the first request and stops after idle, so
there is no always-on cost. (Requires a Fly account; deploy with your own auth.)

```bash
fly auth login
fly launch --no-deploy        # detects the Dockerfile; keep the provided fly.toml
fly deploy                     # builds remotely if you have no local Docker
fly open                       # open the deployed URL
fly logs                       # watch cold-start / health checks / OOM
```

After idle, `fly status` should show the machine **stopped**; the next request cold-starts
it within the health-check grace period.

## Layout

```
app.py              Streamlit UI (entrypoint)
tomography_uq.py    run_simple_uq(params, log_callback) — the parameterized pipeline
senDOE/             vendored snapshot of the research package (see SENDOE_VENDOR.md)
Dockerfile          ubuntu:22.04 + idaes get-extensions + fail-fast smoke tests
fly.toml            Fly.io scale-to-zero config
requirements.txt    pinned Python dependencies
.streamlit/         Streamlit server config
```
