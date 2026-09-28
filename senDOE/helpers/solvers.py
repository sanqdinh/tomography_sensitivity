"""Running IPOPT: locating the binary, walking the linear-solver chain, and reading its log.

:func:`solve_with_fallback` is the entry point. It distinguishes three outcomes -- the binary
did not run, IPOPT rejected the model, IPOPT ran and failed numerically -- because each needs a
different response, and collapsing them sends the reader after a binary that is present.
"""

import os
import re
import shutil

import pyomo.environ as pyo


def resolve_ipopt() -> str:
    """Locate the IPOPT executable: env override -> common path -> PATH -> bare name."""
    exe = os.environ.get("IPOPT_EXECUTABLE")
    if exe and os.path.exists(exe):
        return exe
    if os.path.exists("/usr/local/bin/ipopt"):
        return "/usr/local/bin/ipopt"
    which = shutil.which("ipopt")
    if which:
        return which
    return "ipopt"  # let Pyomo resolve it from PATH


# ma97 first: on the shrinkage-decay estimation NLP it is 73-149x faster per iteration than
# ma27 (the cost is 99.6% KKT factorisation, and ma27 has no nested-dissection ordering). mumps
# is REMOVED -- it is absent from the IPOPT 3.14 build /usr/local/bin/ipopt resolves to, where
# it aborts with OPTION_INVALID rather than failing cleanly, so it was never a usable last
# resort. ma57 and ma27 are both KEPT and both still reachable: solve_with_fallback's own
# docstring records a measured case where ma97 fails in restoration at iteration 828 while ma57
# reaches optimal in 308 and ma27 in 1479, so neither is dead weight.
_FALLBACK_LINEAR_SOLVERS = ["ma97", "ma57", "ma27"]


def _make_solver(linear_solver: str, max_iter: int, tol: float = 1e-8, options=None):
    s = pyo.SolverFactory("ipopt", executable=resolve_ipopt())
    s.options["max_iter"] = int(max_iter)
    s.options["linear_solver"] = linear_solver
    s.options["tol"] = float(tol)
    for k, v in (options or {}).items():
        s.options[k] = v
    return s


def solve_with_fallback(model, *, linear_solver="ma27", max_iter=3000, tol=1e-8, tee=False,
                        log_callback=None, options=None):
    """Solve with ``linear_solver``, then the rest of ``ma97 -> ma57 -> ma27``.  Three outcomes.

    - the binary **failed to run** (missing from the build): try the next one, and if none runs,
      say so.
    - IPOPT ran and rejected the **model** (``can't evaluate sqrt'(0)``, an invalid number, too
      few degrees of freedom): raise at once.  Every other linear solver fails identically, and
      walking the chain reports a missing binary that is sitting right there.
    - IPOPT ran and failed **numerically** (restoration failure, step computation error): try the
      next one, because they genuinely differ -- measured at grid 12 / K=3 on a damage-model
      estimation NLP, ma97 fails in restoration at iteration 828 while ma57 reaches optimal in
      308 and ma27 in 1479.  But if they all fail, the message must say that they RAN and failed,
      not that none was usable.
    """
    order = [linear_solver] + [s for s in _FALLBACK_LINEAR_SOLVERS if s != linear_solver]
    last, numerical = None, []
    for name in order:
        try:
            res = _solve_streaming(_make_solver(name, max_iter, tol, options), model, tee,
                                   log_callback)
            return res, name
        except Exception as exc:
            text = str(exc)
            if _is_model_error(text):
                raise RuntimeError(_curate(text)) from exc
            if _is_numerical_failure(text):
                numerical.append(name)
                last = exc
                if log_callback:
                    log_callback("\n[%s ran and failed numerically; trying the next linear "
                                 "solver, which on this model can differ]\n" % name)
                continue
            last = exc
            if log_callback:
                log_callback("\n[linear solver %r unavailable: %s]\n" % (name, text[:200]))
    if numerical:
        raise RuntimeError(
            "every linear solver tried (%s) RAN and failed numerically; %s. This is a property "
            "of the model at this point, not a missing binary. %s"
            % (", ".join(order), ", ".join(numerical) + " reached a numerical failure",
               _curate(str(last)))) from last
    raise RuntimeError("no usable IPOPT linear solver (tried %s): %s" % (order, last))


def _is_numerical_failure(text: str) -> bool:
    """Did IPOPT run and fail on the numbers, rather than on the model or the binary?

    Distinct from :func:`_is_model_error` because the response differs: a model IPOPT rejects
    will be rejected identically by every linear solver, whereas a numerical failure will not.
    """
    return ("Restoration Failed" in text or "restoration phase failed" in text
            or "Error in step computation" in text)


def _is_model_error(text: str) -> bool:
    """Did IPOPT start and object to the model, rather than fail to start?

    Any of these means the binary ran: retrying on another linear solver fails identically and
    reports a missing binary that is sitting right there.
    """
    return ("can't evaluate" in text or "Error evaluating" in text
            or "Invalid number" in text or "Ipopt " in text
            or "too few degrees of freedom" in text)


def _curate(text: str) -> str:
    if "sqrt'(0)" in text:
        return ("IPOPT could not differentiate a square root at zero: \"can't evaluate "
                "sqrt'(0)\". Some smoothed |.| in the model has a zero argument and a zero "
                "smoothing term there, so it is not differentiable at that point. Give the "
                "smoothing a floor that does not vanish, or switch the term off.")
    if "Restoration Failed" in text or "restoration phase failed" in text:
        it = (re.findall(r"Number of Iterations\.*:\s*(\S+)", text) or ["?"])[-1]
        return ("IPOPT terminated in RESTORATION FAILURE after %s iterations. This is a result, "
                "not a missing solver: the binary ran, could not restore feasibility, and gave "
                "up. SWITCHING LINEAR SOLVER OFTEN DOES HELP here -- measured on a "
                "damage-model estimation NLP at grid 12/K=3, ma97 fails this way at iteration "
                "828 while ma57 "
                "reaches optimal in 308 and ma27 in 1479. Also worth checking before blaming the "
                "model: the objective scale. IPOPT's gradient-based scaling only caps large "
                "gradients (min(1, 100/||g||)) and never lifts a small one, so an objective "
                "whose gradient is orders below the constraint rows is left invisible -- one "
                "measured case had 3.2e-05 against rows up to 5.1e+03, and obj_scaling_factor "
                ">= 1e3 converted that same failure into an optimal solve. Read the last "
                "iteration line too: a large "
                "lg(rg) with ||d|| = 0 means the step computation degenerated."
                % it)
    first = [ln for ln in text.splitlines() if "valuat" in ln]
    return "IPOPT rejected the model: %s" % (first[0].strip() if first else text[:300])


def _solve_streaming(solver, model, tee, log_callback):
    """``solver.solve(tee=True)`` with the subprocess log forwarded to ``log_callback``.

    The log is ALWAYS captured, even with no ``log_callback``.  Pyomo's ``tee=False`` buries the
    solver's stdout in a temp file it then deletes, and raises only "Solver (ipopt) did not exit
    normally" -- so a model IPOPT explicitly rejected ("can't evaluate sqrt'(0)") became
    indistinguishable from a missing binary, and :func:`solve_with_fallback` walked the whole
    linear-solver chain and blamed the linear solver.  Capturing it means the reason survives
    into the exception.
    """
    try:
        from pyomo.common.tee import capture_output
    except Exception:
        return solver.solve(model, tee=tee)

    buf = []

    # A FAILING log_callback MUST NOT BREAK THE SOLVE OR EAT THE LOG.
    #
    # Pyomo forwards the solver's stdout on its own reader thread ("Thread-N (_mergedReader)").
    # If the callback raises there -- which a Streamlit callback does, with NoSessionContext,
    # because a background thread has no ScriptRunContext -- the exception propagates out of
    # write(), and Pyomo's TeeStream responds by printing
    #     Error writing to output stream <_W @ 0x...>: NoSessionContext:
    #     Is this a writeable TextIOBase object?
    #     The following was left in the output buffer: '  21  4.7560765e-02 ...'
    # per chunk. So the terminal fills with noise AND the IPOPT iteration lines are DROPPED
    # rather than displayed. Swallowing the callback's exception fixes both: `buf` is appended
    # before the callback is tried, so the log is complete regardless.
    #
    # Latched off after the first failure: a callback that raises once raises 3000 times, and
    # the exception handling is not free.
    broken = [False]

    class _W:
        def write(self, chunk):
            if chunk:
                buf.append(chunk)
                if log_callback is not None and not broken[0]:
                    try:
                        log_callback(chunk)
                    except Exception:
                        broken[0] = True     # keep buffering; stop forwarding
            return len(chunk)

        def flush(self):
            pass

        def writable(self):
            return True

        def isatty(self):
            return False

    try:
        with capture_output(_W()):
            return solver.solve(model, tee=True)
    except Exception as exc:
        tail = "".join(buf)[-4000:]
        raise RuntimeError("%s\n--- solver log ---\n%s" % (exc, tail)) from exc


def reg_fraction(log: str):
    """``(regularised, total)`` IPOPT iterations.

    Column 6 of an iteration line is ``lg(rg)``; ``"-"`` means no Hessian regularisation was
    applied on that iteration.  Counting any other way is how the bogus "201 of 200" figure
    arose -- a substring test that also matched the header and the restoration lines.
    """
    n = r = 0
    for ln in log.splitlines():
        f = ln.split()
        if len(f) < 10 or not re.fullmatch(r"\d+r?", f[0]):
            continue
        n += 1
        if f[6] != "-":
            r += 1
    return r, n
