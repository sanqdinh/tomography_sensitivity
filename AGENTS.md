# Repository Guidelines

## Project Structure & Module Organization

`app.py` is the Streamlit entry point. Core Python code lives in `senDOE/`: numerical utilities are in `helpers/`, reconstruction models in `models/`, and Pyomo sensitivity integration in `sensitivity/`. Browser components are static assets under `frontend/<component>/`. Put reproducible experiments and diagnostics in `scripts/`; `archives/` contains historical prototypes and should not be treated as production code. Deployment configuration is in `Dockerfile`, `fly.toml`, `.streamlit/`, and `.github/workflows/`.

## Build, Test, and Development Commands

- `pip install -r requirements.txt` installs the pinned Python stack.
- `streamlit run app.py` starts the local UI on port 8501.
- `docker build -t tomo-uq:local .` creates the recommended environment, installs IPOPT/k_aug/dot_sens, and runs build-time smoke checks.
- `docker run --rm -p 8501:8501 tomo-uq:local` serves the containerized app.
- `python3 -m senDOE.models.tomography_3d` runs a fast, solver-free 3D check.
- `python3 -m senDOE.models.tomography_2d_shrinkage_decay` runs the model invariant checks.
- `python3 -m senDOE.models.tomography_pyomo_2d_shrinkage_decay --image-res 16 --n-steps 2` compares the Pyomo and NumPy formulations. Add `--solve` only when IPOPT is available.

Run scripts from the repository root with `PYTHONPATH=.`, for example `PYTHONPATH=. python3 scripts/experiment_shrinkage_limits.py`.

## Coding Style & Naming Conventions

Follow existing Python style: four-space indentation, descriptive docstrings, type hints on public interfaces, and grouped standard-library/third-party/local imports. Use `snake_case` for modules, functions, and variables; `PascalCase` for parameter dataclasses. Keep numerical checks close to the model they validate. No formatter or linter is configured, so preserve nearby formatting and compile changed modules with `python3 -m py_compile <file>`.

## Testing Guidelines

There is no standalone automated test suite or coverage threshold. Treat the module checks above and a successful Docker build as the verification path. For numerical changes, run the smallest relevant grid first and report tolerances or residuals. Changes touching native solver orchestration require the Docker checks because pip alone does not provide IPOPT, k_aug, or dot_sens.

## Commit & Pull Request Guidelines

Recent commits use concise, imperative subjects such as `Raise v6 reconstruction defaults`. Keep commits focused and explain numerical rationale in the body when applicable. Pull requests should summarize behavior changes, list commands run, link related issues, and include screenshots for UI changes. Call out solver, performance, or deployment impacts explicitly.
