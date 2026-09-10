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
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) Python dependencies (all wheels; no compilation).
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir -r requirements.txt

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
    && ( /opt/idaes/bin/k_aug --help >/dev/null 2>&1 || true )

# 6) Application code + vendored senDOE (senDOE/ at /app => importable as top-level `senDOE`).
COPY senDOE/ ./senDOE/
COPY .streamlit/ ./.streamlit/
COPY live_sim_component/ ./live_sim_component/
COPY tomography_uq.py tomography_3d.py dose_response.py app.py ./

# 7) Build-time make-or-break checks: vendored package imports, and IPOPT solves end-to-end.
RUN python3 -c "import senDOE; print('senDOE import OK')" \
    && python3 -c "import plotly.graph_objects as go; go.Volume(); print('plotly OK')" \
    && python3 -c "from tomography_3d import shepp_logan_3d, degrade_volume; \
v=shepp_logan_3d(16,4); d=degrade_volume(v,((0.0,0.0,0),),5.0,0.3,0.01,16); \
assert v.shape==(16,16,4) and d.sum()<v.sum(); print('3D degradation sim OK')" \
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
