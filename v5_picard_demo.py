"""Picard outer loop: freeze sigma, solve, re-evaluate, repeat to a fixed point.

The inner problem's c_phi block is LINEAR in phi with constant coefficients, so it contributes
nothing to the Lagrangian Hessian and the sigma-phi cross term that made the projected Hessian
indefinite is gone by construction. The fixed point is the exact-sigma solution, so freezing
costs nothing at convergence -- it is only the error of stopping at one outer iteration.
"""
import sys, time, io, re; sys.path.insert(0,"/home/sdinh/sandbox/tomography_sensitivity")
import numpy as np, pyomo.environ as pyo
import degrade_v5_uq as U5
from degrade_v5 import V5Params, simulate, resolve, material_indicator
from degrade_v3 import _phantom
from degrade_v2 import scale_to_optical_depth
from degrade_v2_uq import _make_solver

res, K = 32, 4
th = scale_to_optical_depth(_phantom(res), 1.1, res)
sc = float(th.max())
seq = tuple((180.0*k/K, 0., 0) for k in range(K))
p = resolve(V5Params(I0=1., c=0.1, a=0.05, b=0., c_cp=0.3, reach=7.0, gamma=100.,
                     f_ref_frac=0.002), th)
_f, _i, y = simulate(th, seq, p, res, record_observations=True)
npix = res*res

def sigma_from(theta_est):
    """sigma evaluated along the trajectory of a given theta estimate."""
    tr = U5.numpy_trajectory(theta_est, seq, p, res)
    return tr["sig"], tr

print("PICARD LOOP, grid %d, K=%d, ma97/metis. Stopping rule: converge or v5 is not tractable."%(res,K))
print("v4 comparator: 45.6 s, 139 iters, 2 regularised, optimal")
print("v5 sigma-free:  no convergence in 200 iters, 201 regularised\n")
theta_est = np.full_like(th, float(th.mean()))
prev = None
t_all = time.time()
for outer in range(1, 9):
    sig, tr = sigma_from(theta_est)
    m = U5.build_v5_model(th, seq, p, res, f_bounds=(0., 1.5*sc), sigma_fixed=sig)
    U5.pin_model(m, tr, fix=False)
    U5.add_estimation_objective(m, y, 0.001, sc)
    s = _make_solver("ma97", 400, 1e-8, {"ma97_order":"metis"})
    t = time.time()
    try:
        s.solve(m, tee=False, logfile="/tmp/pic.log")
    except Exception as e:
        print("  outer %d: solver exception %s"%(outer, str(e)[:70])); break
    dt = time.time()-t
    log = open("/tmp/pic.log").read()
    it = (re.findall(r"Number of Iterations\.*:\s*(\S+)", log) or ["?"])[-1]
    ex = (re.findall(r"EXIT: (.*)", log) or ["?"])[0][:34]
    reg = sum(1 for l in log.splitlines() if re.match(r"^\s*\d+\s", l) and "  -   " not in l)
    new = np.array([pyo.value(m.f[q,0]) for q in m.PIX]).reshape(res,res)
    ds = "-" if prev is None else "%.3e"%(np.abs(new-prev).max()/max(np.abs(new).max(),1e-300))
    err = 100*np.linalg.norm(new-th)/np.linalg.norm(th)
    print("  outer %d: %6.1f s  iters %-4s  regularised %-4d  d(theta) %-10s  L2 %.2f%%  %s"
          %(outer, dt, it, reg, ds, err, ex))
    sys.stdout.flush()
    if prev is not None and np.abs(new-prev).max()/max(np.abs(new).max(),1e-300) < 1e-4:
        print("\n  FIXED POINT reached at outer %d, total %.1f s"%(outer, time.time()-t_all)); break
    prev, theta_est = new, new
    del m
else:
    print("\n  no fixed point in 8 outer iterations, total %.1f s"%(time.time()-t_all))
