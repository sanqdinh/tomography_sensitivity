"""Streamlit UI for the tomographic uncertainty-quantification pipeline.

Wraps :func:`tomography_uq.run_simple_uq` in an interactive page: tune the parameters in
the sidebar, click **Run**, watch the IPOPT solver log stream live, then inspect the six
figures and the D-optimality metric.

The heavy solve (two IPOPT solves + a k_aug sensitivity extraction, minutes at the default
size) runs ONLY on the button press; Streamlit reruns the whole script on every widget
interaction, so results are stashed in ``st.session_state`` to survive reruns without
re-solving.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import streamlit as st

from tomography_uq import UQParams, run_simple_uq

st.set_page_config(page_title="Tomography Sensitivity UQ", layout="wide")

st.title("Tomographic reconstruction + sensitivity-based UQ")
st.caption(
    "Forward IPOPT solve simulates measurements → inverse solve reconstructs the phantom → "
    "k_aug extracts d(image)/d(sinogram) → posterior covariance & D-optimality. "
    "Based on `Example2_simple_uq.py`."
)

with st.sidebar:
    st.header("Run parameters")
    image_res = st.slider(
        "Image resolution (N×N)", min_value=16, max_value=48, value=30, step=2,
        help="Pixels per side. Cost grows ~ N² · n_horizon; the default (30) matches Example2.",
    )
    n_horizon = st.slider(
        "n_horizon", min_value=3, max_value=16, value=10,
        help="Number of time steps. Projection angles = n_horizon − 1 (default 9).",
    )
    angle_start = st.number_input("Angle start (deg)", value=0.0, step=10.0)
    angle_stop = st.number_input("Angle stop (deg)", value=180.0, step=10.0)
    st.divider()
    I0 = st.number_input("I0 (beam intensity)", value=0.0, step=0.5,
                         help="0 → no dose degradation (Example2 default).")
    alpha = st.number_input("alpha (dose response)", value=0.3, step=0.05, format="%.3f")
    beta = st.number_input("beta (Rose response)", value=0.01, step=0.01, format="%.4f")
    st.divider()
    tv_weight = st.number_input("TV regularization weight", value=0.1, step=0.05,
                                format="%.4f")
    noise_scale = st.number_input("Measurement noise covariance (× I)", value=10.0,
                                  step=1.0)
    st.divider()
    max_iter = st.number_input("IPOPT max_iter", value=1000, step=100, min_value=1)
    linear_solver = st.selectbox("Linear solver", ["ma27", "ma57", "mumps"],
                                 index=0,
                                 help="Falls back to the next option if one isn't available.")

    run_clicked = st.button("Run", type="primary", use_container_width=True)
    st.caption("⏱️ A full run (res 30, 9 angles) takes minutes. Keep this tab open.")

# --- heavy solve: only on button press -------------------------------------------------
if run_clicked:
    params = UQParams(
        image_res=int(image_res),
        n_horizon=int(n_horizon),
        I0=float(I0),
        alpha=float(alpha),
        beta=float(beta),
        tv_weight=float(tv_weight),
        noise_cov_scale=float(noise_scale),
        ipopt_max_iter=int(max_iter),
        linear_solver=str(linear_solver),
        angle_start=float(angle_start),
        angle_stop=float(angle_stop),
    )

    st.subheader("Solver log")
    log_box = st.empty()
    log_lines: list[str] = []

    def log_callback(chunk: str) -> None:
        log_lines.append(chunk)
        # Show a rolling tail so very long IPOPT logs stay responsive in the browser.
        log_box.code("".join(log_lines)[-8000:], language="text")

    with st.spinner("Solving forward + inverse NLP and extracting k_aug sensitivity…"):
        try:
            results = run_simple_uq(params, log_callback=log_callback)
            st.session_state["results"] = results
        except Exception as exc:  # surface failures instead of a blank page
            st.session_state.pop("results", None)
            st.error(f"Run failed: {exc}")
            st.exception(exc)

# --- render persisted results (survives reruns without re-solving) ---------------------
results = st.session_state.get("results")
if results is not None:
    st.success(
        f"**D-optimality = {results.d_optimality:.4f}**  ·  "
        f"forward: {results.forward_solver_status} ({results.forward_linear_solver})  ·  "
        f"inverse: {results.inverse_solver_status} ({results.inverse_linear_solver})  ·  "
        f"{results.n_free_image0} free pixels, "
        f"{results.n_sinogram_measurements} measurements"
    )

    row1 = st.columns(3)
    row1[0].pyplot(results.fig_phantom, use_container_width=True)
    row1[0].caption("Original phantom")
    row1[1].pyplot(results.fig_nlp, use_container_width=True)
    row1[1].caption("Reconstruction (NLP)")
    row1[2].pyplot(results.fig_covariance, use_container_width=True)
    row1[2].caption("Posterior covariance (log10 diagonal)")

    row2 = st.columns(3)
    row2[0].pyplot(results.fig_sinogram, use_container_width=True)
    row2[0].caption("Sinogram (merged over time)")
    row2[1].pyplot(results.fig_fbp, use_container_width=True)
    row2[1].caption("Reconstruction (FBP)")
    row2[2].pyplot(results.fig_sart, use_container_width=True)
    row2[2].caption("Reconstruction (SART)")
else:
    st.info("Set parameters in the sidebar and click **Run** to start a solve.")
