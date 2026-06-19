import os
import numpy as np
import scipy.sparse
from pyomo.environ import Var, Constraint, Objective, SolverFactory, value, ComponentUID
from pyomo.contrib.sensitivity_toolbox.sens import sensitivity_calculation, SensitivityInterface
from pyomo.contrib.sensitivity_toolbox.k_aug import K_augInterface, InTempDir

# k_aug / dot_sens live under ~/.idaes/bin (IDAES install)
_IDAES_BIN = os.path.expanduser("~/.idaes/bin")
if _IDAES_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + _IDAES_BIN


def extract_sensitivity_matrix(model, var_list, param_list, mode="k_aug",
                               return_type="dense"):
    """Extract the sensitivity matrix dx/dp from a Pyomo model.

    Computes the Jacobian of var_list with respect to param_list via the
    Implicit Function Theorem on the KKT system at the solved point.

    Parameters
    ----------
    model : pyomo ConcreteModel
        For k_aug mode, the model must be pre-solved with IPOPT.
        For sipopt mode, the model may be unsolved or pre-solved.
    var_list : list of Pyomo Var data objects
        Variables to differentiate (rows of the output matrix).
    param_list : list of Pyomo Param or fixed Var data objects
        Parameters to differentiate with respect to (columns of the
        output matrix). May be mutable Params or fixed Vars.
    mode : str, optional
        "k_aug" (default) or "sipopt".
    return_type : str, optional
        "dense" (default) returns a numpy.ndarray.
        "sparse" returns a scipy.sparse.csr_matrix.

    Returns
    -------
    dvar_dparam : numpy.ndarray or scipy.sparse.csr_matrix
        Shape (len(var_list), len(param_list)).
        Entry [i, j] = d(var_list[i]) / d(param_list[j]).
    """
    if return_type not in ("dense", "sparse"):
        raise ValueError(
            f"Invalid return_type '{return_type}'. Must be 'dense' or 'sparse'."
        )

    if mode == "k_aug":
        return _extract_sensitivity_kaug(model, var_list, param_list,
                                         return_type)
    elif mode == "sipopt":
        return _extract_sensitivity_sipopt(model, var_list, param_list,
                                           return_type)
    else:
        raise ValueError(
            f"Invalid mode '{mode}'. Must be 'k_aug' or 'sipopt'."
        )


def _extract_sensitivity_kaug(model, var_list, param_list, return_type):
    """k_aug backend: replicate get_dsdp logic with square_problem support."""
    theta_names = [p.name for p in param_list]

    # Fix param vars on the model (as get_dsdp does)
    param_components = []
    for name in theta_names:
        comp = model.find_component(name)
        if comp is None:
            raise RuntimeError("Cannot find component %s on model" % name)
        if comp.ctype is Var:
            comp.fix()
        param_components.append(comp)

    # Create SensitivityInterface (clones model) and setup
    sens = SensitivityInterface(model, clone_model=True)
    m = sens.model_instance
    sens.setup_sensitivity(param_components)

    # Detect square system: count free vars vs active equality constraints
    n_free = sum(
        1 for v in m.component_data_objects(Var, active=True) if not v.fixed
    )
    n_eq = sum(
        1 for c in m.component_data_objects(Constraint, active=True) if c.equality
    )
    is_square = n_free == n_eq

    # Add dummy objective if no active objective exists (k_aug expects one)
    has_obj = any(
        True for _ in m.component_data_objects(Objective, active=True)
    )
    if not has_obj:
        m._kaug_dummy_obj = Objective(expr=0)

    # Create K_augInterface and set square_problem option if needed
    k_aug_interface = K_augInterface()
    if is_square:
        k_aug_interface.set_k_aug_options(square_problem="")

    # Run k_aug
    k_aug_interface.k_aug(m, tee=False)

    # Write NL file with symbolic labels to get col names
    nl_data = {}
    with InTempDir():
        base_fname = "col_row"
        nl_file = ".".join((base_fname, "nl"))
        col_file = ".".join((base_fname, "col"))
        m.write(nl_file, io_options={"symbolic_solver_labels": True})
        for fname in [nl_file, col_file]:
            with open(fname, "r") as fp:
                nl_data[fname] = fp.read()

    # Parse k_aug output (identical to get_dsdp)
    dsdp_raw = np.fromstring(k_aug_interface.data["dsdp_in_.in"], sep="\n\t")
    col = nl_data[col_file].strip("\n").split("\n")

    dsdp_raw = dsdp_raw.reshape(
        (len(theta_names), int(len(dsdp_raw) / len(theta_names)))
    )
    dsdp_raw = dsdp_raw[: len(theta_names), : len(col)]

    # Filter out sensitivity toolbox internal components and negate
    block_name = sens.get_default_block_name()
    col = [c for c in col if block_name not in c]
    dsdp = np.zeros((len(theta_names), len(col)))
    for i in range(len(theta_names)):
        for j in range(len(col)):
            dsdp[i, j] = -dsdp_raw[i, j]

    # Build column name -> index map
    col_idx = {name: i for i, name in enumerate(col)}

    # dsdp[k, j] = d(var_j) / d(param_k)
    # We need result[i, k] = d(var_list[i]) / d(param_list[k])
    var_cols = [col_idx[var.name] for var in var_list]
    dvar_dparam = scipy.sparse.csr_matrix(dsdp)[:, var_cols].T

    if return_type == "sparse":
        return scipy.sparse.csr_matrix(dvar_dparam)
    return np.asarray(dvar_dparam.todense())


def _extract_sensitivity_sipopt(model, var_list, param_list, return_type):
    """sIPOPT backend: one sensitivity_calculation call per parameter."""
    n_vars = len(var_list)
    n_params = len(param_list)
    dvar_dparam = np.zeros((n_vars, n_params))

    dp = 1e-3

    for j, param in enumerate(param_list):
        p_val = value(param)

        # Clone and perturb non-fixed variable values so ipopt_sens takes
        # >0 iterations (avoids skipping Hessian eval on pre-solved models).
        m_work = model.clone()
        for v in m_work.component_data_objects(Var, active=True):
            if not v.fixed:
                v.set_value(value(v) * 0.9 + 0.01)

        param_on_work = ComponentUID(param, context=model).find_component_on(m_work)

        m_sens = sensitivity_calculation(
            "sipopt",
            m_work,
            paramList=[param_on_work],
            perturbList=[p_val + dp],
            cloneModel=True,
            tee=False,
        )

        for i, var in enumerate(var_list):
            var_on_sens = ComponentUID(var, context=model).find_component_on(m_sens)
            x_nom = value(var_on_sens)
            x_pert = m_sens.sens_sol_state_1[var_on_sens]
            dvar_dparam[i, j] = (x_pert - x_nom) / dp

    if return_type == "sparse":
        return scipy.sparse.csr_matrix(dvar_dparam)
    return dvar_dparam
