# syntax=docker/dockerfile:1.7
# Base = the exact platform the IDAES prebuilt solver binaries target
# (idaes get-extensions resolves ubuntu2204-x86_64 here). IPOPT/k_aug link only against
# standard Ubuntu 22.04 shared libs and statically embed HSL (incl. ma86), so this is the
# lowest-risk base for the native binaries.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    IDAES_DATA=/opt/idaes \
    PATH=/opt/idaes/bin:/usr/local/bin:/usr/bin:/bin

# 1) Runtime shared libs the IDAES ipopt/k_aug binaries need (verified via objdump -p),
#    plus python and the tools `idaes get-extensions` uses to download.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        libgfortran5 \
        liblapack3 \
        libblas3 \
        libstdc++6 \
        libgomp1 \
        libgcc-s1 \
        libquadmath0 \
        libbz2-1.0 \
        git \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) Python dependencies. senDOE is fetched from a private Git repository; the BuildKit secret
# is read only by git's askpass helper and is removed before the layer is committed.
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN --mount=type=secret,id=sendoe_read_token \
    set -eu; \
    askpass=/tmp/sendoe-git-askpass; \
    printf '%s\n' '#!/bin/sh' \
        'case "$1" in' \
        '*Username*) echo x-access-token ;;' \
        '*Password*) cat /run/secrets/sendoe_read_token ;;' \
        '*) exit 1 ;;' \
        'esac' > "$askpass"; \
    chmod 700 "$askpass"; \
    GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0 \
        python3 -m pip install --no-cache-dir -r requirements.txt; \
    rm -f "$askpass"

# 3) Install the native solver binaries to a FIXED, user-independent location.
#    IDAES_DATA (set above) makes get-extensions install to /opt/idaes/bin regardless of
#    $HOME, so the binaries resolve for any runtime user (avoids the /root/.idaes pitfall).
RUN mkdir -p /opt/idaes \
    && idaes get-extensions --verbose \
    && test -x /opt/idaes/bin/ipopt \
    && test -x /opt/idaes/bin/k_aug \
    && test -x /opt/idaes/bin/dot_sens

# 4) Satisfy any hardcoded /usr/local/bin/ipopt and give a clean default on PATH.
RUN ln -sf /opt/idaes/bin/ipopt    /usr/local/bin/ipopt \
    && ln -sf /opt/idaes/bin/k_aug    /usr/local/bin/k_aug \
    && ln -sf /opt/idaes/bin/dot_sens /usr/local/bin/dot_sens

# 5) Fail-fast: confirm the binaries actually run in this image (ABI check).
RUN /opt/idaes/bin/ipopt --version \
    && ( /opt/idaes/bin/k_aug --help >/dev/null 2>&1 || true ) \
    # ma97 is not a preference, it is the difference between 0.2 s and 16 s per iteration on the
    # v6 estimation NLP (99.6% of which is KKT factorisation). If a future `idaes get-extensions`
    # ships a build without it, the app would silently fall back to ma27 and every reconstruction
    # would get ~73x slower with nothing in the logs saying why. Fail the build instead.
    && python3 -c "\
import pyomo.environ as pyo; \
m=pyo.ConcreteModel(); m.x=pyo.Var(initialize=1.0); \
m.c=pyo.Constraint(expr=m.x>=2.0); m.o=pyo.Objective(expr=(m.x-3.0)**2); \
s=pyo.SolverFactory('ipopt', executable='/opt/idaes/bin/ipopt'); \
s.options['linear_solver']='ma97'; r=s.solve(m); \
assert abs(pyo.value(m.x)-3.0) < 1e-6, pyo.value(m.x); \
print('IPOPT linear solver ma97 present and solving:', r.solver.termination_condition)"

# 6) Application code. senDOE is installed from its pinned Git revision in requirements.txt.
COPY .streamlit/ ./.streamlit/
COPY frontend/ ./frontend/
# archives/ contains historical prototypes and is deliberately not copied: nothing imports it,
# and the image must build without it.
COPY app.py ./

# 6b) Materialize plotly.min.js for the 3D Volume component from THIS image's plotly, so the
#     browser-side bundle always matches the figure JSON the app emits. app.py does the same copy
#     at import time; doing it here means the runtime copy is a no-op and cannot fail on a
#     read-only /app (which would silently drop the view to server-side rendering).
RUN python3 -c "\
import os, shutil, plotly; \
src=os.path.join(os.path.dirname(plotly.__file__),'package_data','plotly.min.js'); \
shutil.copyfile(src, '/app/frontend/volume_sim_component/plotly.min.js'); \
print('plotly.min.js staged: %.1f MB' % (os.path.getsize(src)/1e6))"

# 7) Build-time make-or-break checks: installed package imports, and IPOPT solves end-to-end.
RUN python3 -c "import senDOE; print('senDOE import OK')" \
    && python3 -c "import plotly.graph_objects as go; go.Volume(); print('plotly OK')" \
    && test -s /app/frontend/volume_sim_component/plotly.min.js \
    && test -s /app/frontend/volume_sim_component/index.html \
    && python3 -c "from senDOE.models.tomography_3d import shepp_logan_3d, degrade_volume; \
v=shepp_logan_3d(16,4); d=degrade_volume(v,((0.0,0.0,0),),5.0,0.3,0.01,16); \
assert v.shape==(16,16,4) and d.sum()<v.sum(); print('3D degradation sim OK')" \
    && python3 -c "from senDOE.helpers.dose import check_photon_balance; \
w=check_photon_balance(verbose=False); \
print('accumulate_dose == spec reference: forward %.1e, antiparallel %.1e' % (w['forward'], w['antiparallel']))" \
    && python3 -c "import senDOE.models.tomography_2d_shrinkage_decay, senDOE.models.tomography_pyomo_2d_shrinkage_decay; \
print('senDOE shrinkage-decay models present and importable')" \
    && python3 -c "import senDOE.models.tomography_2d_shrinkage_decay as sd; \
r=sd.check_invariants(image_res=48, n_steps=4, verbose=False); \
print('shrinkage-decay invariants OK: mass drift %.1e, collapse %.1e, |colsum-1| %.1e, ' \
      'I0=0 leak %.1e (REPORTED: softplus is eta*log2 at rest, not 0)' \
      % (r['mass_drift'], r['collapse'], r['colsum_err'], r['I0_leak']))" \
    && python3 -c "import senDOE.models.tomography_pyomo_2d_shrinkage_decay as sdp; \
sdp.check_softplus_lifting(verbose=False); \
rq=sdp.check_forward(image_res=16, n_steps=2, verbose=False, simultaneous=False); \
rs=sdp.check_forward(image_res=16, n_steps=2, verbose=False, simultaneous=True); \
assert max(rq, rs) < 1e-10, (rq, rs); \
print('shrinkage-decay Pyomo model == numpy model: residual %.1e sequential / %.1e ' \
      'simultaneous (softplus lifting exact)' % (rq, rs))" \
    && python3 -c "from pyomo.contrib.pynumero.asl import AmplInterface; \
assert AmplInterface.available(), 'libpynumero_ASL.so missing -- check_scaling would silently skip'; \
print('pynumero ASL OK')" \
    && python3 -c "import senDOE.models.tomography_pyomo_2d_shrinkage_decay as sdp; \
r=sdp.check_scaling(image_res=16, n_steps=2, simultaneous=False, verbose=False); \
s=sdp.check_scaling(image_res=16, n_steps=2, simultaneous=True,  verbose=False); \
assert r['ok'] and s['ok'], (r['failures'], s['failures']); \
print('shrinkage-decay NLP scaling OK: median row %.3g, min %.3g (%s), max %.3g (%s), |grad f| %.3g' \
      % (s['row_med'], s['row_min'], s['row_min_block'], s['row_max'], s['row_max_block'], s['grad_inf']))" \
    && python3 -m py_compile app.py && echo 'app.py compiles' \
    && python3 -c "import pyomo.environ as pyo; \
m=pyo.ConcreteModel(); m.x=pyo.Var(initialize=1.0); \
m.c=pyo.Constraint(expr=m.x>=2.0); m.o=pyo.Objective(expr=(m.x-3.0)**2); \
s=pyo.SolverFactory('ipopt', executable='/opt/idaes/bin/ipopt'); \
r=s.solve(m); print('ipopt termination:', r.solver.termination_condition); \
assert abs(pyo.value(m.x)-3.0)<1e-6, pyo.value(m.x); print('IPOPT solve OK')"

EXPOSE 8501

# Single-tenant public demo: running as root is acceptable; IDAES_DATA + global PATH make
# the solver layout user-independent anyway.
CMD ["python3", "-m", "streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
