import pyomo.environ as pyo
import numpy as np
from skimage.transform import resize
from sympy.matrices.expressions.matexpr import MatrixSymbol
import sympy as sp
from senDOE.helpers.geometry import get_line_abc_from_r_theta, line_grid_intersections


def create_sample_model(n_horizon: int, image_res):
    """
    Create a Pyomo ConcreteModel named 'sample' with variable image[i,j,h]
    of size (100, 100, n_horizon).
    All entries with horizon index == 0 are fixed to 0. Others are free.
    """
    if n_horizon < 1:
        raise ValueError("n_horizon must be at least 1")

    m = pyo.ConcreteModel(name="sample")

    # index sets: 0 .. 99 for spatial dims, 0 .. n_horizon-1 for horizon
    m.ix = pyo.Set(initialize=range(image_res))  # rows index 0..99
    m.iy = pyo.Set(initialize=range(image_res))  # cols index 0..99
    m.time = pyo.Set(initialize=range(n_horizon))  # horizon index 0..n_horizon-1

    # 2D variable x time
    m.image = pyo.Var(
        m.ix, m.iy, m.time, domain=pyo.Reals, initialize=1, bounds=(1e-7, 1.1)
    )
    # NOTE: the pre-refactor model used an `m.local_intensity` pyo.Var here
    # that was constrained (via `m.intensity_constraint`) to equal the
    # per-pixel cumulative-radon expression, and the dynamic constraint then
    # read the Var. The current code inlines that expression into
    # `m.local_intensity_np` (a NumPy ndarray of Pyomo expressions) and the
    # dynamic constraint reads `local_intensity_np` directly. The Var + its
    # empty constraint container were dead weight (1800 phantom free vars
    # per model) and have been deleted. The commented-out blocks below
    # (intensity_constraint / dynamic_constraint Var-based variants)
    # document the pre-refactor pattern; reviving the Var would require
    # both populating intensity_constraint AND switching the
    # local_intensity_np references in the dynamic constraint to use the Var.
    m.local_intensity_np = np.ndarray((image_res, image_res, n_horizon), dtype=object)

    # fix slice at horizon 0 to value 0 and arrange into an np array object
    image = np.ndarray(n_horizon, dtype=object)

    for k in m.time:
        image[k] = np.empty((image_res, image_res), dtype=object)
        for i in m.ix:
            for j in m.iy:
                if k == 0:
                    m.image[i, j, 0].fix(1.0)
                image[k][i, j] = m.image[i, j, k]

    m.image_array = image
    m.measurement_list = []

    m.angle_set = pyo.Set(initialize=[])
    m.r_distance_set = pyo.Set(initialize=[])
    m.sinogram = pyo.Var(
        m.r_distance_set, m.angle_set, m.time, domain=pyo.Reals, initialize=0
    )
    m.sinogram_constraint = pyo.Constraint(m.r_distance_set, m.angle_set, m.time)
    m.dynamic_constraint = pyo.Constraint(m.ix, m.iy, m.time)
    return m


def load_image_to_sample(model, image, time=0):
    model_width = model.ix.last() + 1
    model_height = model.iy.last() + 1

    # check if the image is the same size as the model
    if image.shape != (model_height, model_width):
        image_input = resize(image, (model_height, model_width))
    else:
        image_input = image

    for i in model.ix:
        for j in model.iy:
            if time == 0:
                model.image[i, j, time].fix(max(image_input[i, j], 1e-7))
            else:
                model.image[i, j, time].set_value(max(image_input[i, j], 1e-7))


def degradation_local_intensity_expression(
    image,
    r_distance_set,
    angle_set,
    I0=10,
    alpha_Dose_Response=0.3,
    beta_Rose_Response=0.01,
    mode="sympy",
    verbose=False,
):
    if not isinstance(image, np.ndarray):
        raise TypeError(
            "image must be a NumPy ndarray. Please repack into NP array with dtype = object"
        )

    if image.dtype != object:
        raise TypeError(
            "Incorrect dtype. image must be a NumPy array with dtype=object"
        )

    if mode == "sympy":
        _exp = sp.exp
    elif mode == "pyomo":
        _exp = pyo.exp
    else:
        print("Unknown mode, reverse to numpy expression")
        _exp = np.exp
    height, width = image.shape
    extent = [-width / 2, width / 2, -height / 2, height / 2]
    a_seg, b_seg, c_seg = get_line_abc_from_r_theta(r_distance_set, angle_set)
    segment_intersection, image_intersection, radon, intersection_length = (
        line_grid_intersections(
            a_seg,
            b_seg,
            c_seg,
            image,
            x_range=[extent[0], extent[1]],
            y_range=[extent[2], extent[3]],
        )
    )

    if isinstance(image, MatrixSymbol):
        height, width = image.shape

        local_intensity = [[0 for _ in range(width)] for _ in range(height)]
        for i in range(len(segment_intersection)):
            i_x = int(image_intersection[i, 0])
            i_y = int(image_intersection[i, 1])
            local_intensity[i_x, i_y] = I0 * _exp(-sum(radon[j] for j in range(i)))
    else:
        local_intensity = np.zeros(image.shape, dtype=object)
        for i in range(len(segment_intersection)):
            i_x = int(image_intersection[i, 0])
            i_y = int(image_intersection[i, 1])
            local_intensity[i_x, i_y] = I0 * _exp(-sum(radon[j] for j in range(i)))
            if verbose:
                print(i_x, i_y, local_intensity[i_x, i_y])

    sinogram = sum(radon)
    return local_intensity, sinogram


def add_beam_constraints_pyomo(
    model,
    measurement_set,
    injection_time=0,
    I0=10,
    alpha_Dose_Response=0.3,
    beta_Rose_Response=0.01,
    image_res=100,
):

    image_pyomo = model.image_array
    local_intensity_new = np.zeros((image_res, image_res))
    radon = {}
    for measurement in measurement_set:
        add_intensity, radon_tempt = degradation_local_intensity_expression(
            image_pyomo[injection_time],
            measurement["r"],
            measurement["theta"],
            I0=I0,
            alpha_Dose_Response=alpha_Dose_Response,
            beta_Rose_Response=beta_Rose_Response,
            mode="pyomo",
            verbose=False,
        )
        radon[
            measurement["r"],
            measurement["theta"],
            injection_time,
        ] = radon_tempt
        local_intensity_new = local_intensity_new + add_intensity

    model.local_intensity_new = local_intensity_new

    # %% Intensity constraints
    # def _local_intensity_rule(model, i, j):
    #     if model.local_intensity_new[i, j] is None:
    #         return model.local_intensity[i, j, injection_time] == 0
    #     return (
    #         model.local_intensity[i, j, injection_time]
    #         == model.local_intensity_new[i, j]
    #     )
    #
    # model.local_intensity_constraint = pyo.Constraint(
    #     range(image_res), range(image_res), rule=_local_intensity_rule
    # )

    # for i in model.ix:
    #     for j in model.iy:
    #         if model.local_intensity_new[i, j] is None:
    #             model.intensity_constraint[i, j, injection_time] = (
    #                 model.local_intensity[i, j, injection_time] == 0
    #             )
    #         else:
    #             model.intensity_constraint[i, j, injection_time] = (
    #                 model.local_intensity[i, j, injection_time]
    #                 == model.local_intensity_new[i, j]
    #             )

    for i in model.ix:
        for j in model.iy:
            if model.local_intensity_new[i, j] is None:
                model.local_intensity_np[i, j, injection_time] = 0
            else:
                model.local_intensity_np[i, j, injection_time] = (
                    model.local_intensity_new[i, j]
                )

    # %% Dynamic constraints
    # def _local_dynamic_rule(model, i, j):
    #     if injection_time == model.time.last():
    #         return pyo.Constraint.Skip
    #
    #     return model.image[i, j, injection_time + 1] == model.image[
    #         i, j, injection_time
    #     ] * pyo.exp(
    #         -alpha_Dose_Response * model.local_intensity[i, j, injection_time]
    #         - beta_Rose_Response * model.local_intensity[i, j, injection_time] ** 2
    #     )
    #
    # model.local_dynamic_constraint = pyo.Constraint(
    #     range(image_res), range(image_res), rule=_local_dynamic_rule
    # )
    dynamic_constraint_scale = 1e-3
    for i in model.ix:
        for j in model.iy:
            if injection_time < model.time.last():
                model.dynamic_constraint[
                    i, j, injection_time
                ] = dynamic_constraint_scale * model.image[
                    i, j, injection_time + 1
                ] == dynamic_constraint_scale * model.image[
                    i, j, injection_time
                ] * pyo.exp(
                    -alpha_Dose_Response
                    * model.local_intensity_np[i, j, injection_time]
                    - beta_Rose_Response
                    * model.local_intensity_np[i, j, injection_time] ** 2
                )

                # model.dynamic_constraint[i, j, injection_time] = pyo.log(
                #     model.image[i, j, injection_time + 1]
                # ) == pyo.log(model.image[i, j, injection_time]) + (
                #     -alpha_Dose_Response * model.local_intensity[i, j, injection_time]
                #     - beta_Rose_Response
                #     * model.local_intensity[i, j, injection_time] ** 2
                # )

    # Sinogram constraints
    measurement_list_new = [
        [measurement["r"], measurement["theta"], injection_time]
        for measurement in measurement_set
    ]

    model.r_distance_set.update([measurement["r"] for measurement in measurement_set])
    model.angle_set.update([measurement["theta"] for measurement in measurement_set])

    for measurement in measurement_list_new:
        if measurement not in model.measurement_list:
            model.sinogram[measurement] = 0
            model.sinogram[measurement].free()
            model.sinogram_constraint[measurement] = (
                radon[tuple(measurement)] == model.sinogram[measurement]
            )

    return model


def extract_sinogram_value(model, time=0):
    r_set = np.sort([r for r in model.r_distance_set])
    angle_set = np.sort([angle for angle in model.angle_set])
    sinogram = np.zeros((len(r_set), len(angle_set)))

    for r in r_set:
        for angle in angle_set:
            # return i_r location of r in r_set
            i_r = np.where(r_set == r)[0][0]
            i_theta = np.where(angle_set == angle)[0][0]
            sinogram[i_r, i_theta] = model.sinogram[r, angle, time].value
    return sinogram


def update_sinogram_rmse_expression(
    model, measurement_set, sinogram_data, singram_weight=None
):
    if singram_weight is None:
        singram_weight = np.ones(len(sinogram_data))
        singram_weight = singram_weight.flatten()
    else:
        singram_weight = np.diag(singram_weight)
    model.rmse_sinogram_expression = 0

    n_measurements = len(measurement_set)
    model.sinogram_data = pyo.Var(
        model.r_distance_set,
        model.angle_set,
        model.time,
        domain=pyo.Reals,
        initialize=0,
    )
    for i_meas, measurement in enumerate(measurement_set):
        id = [measurement["r"], measurement["theta"], measurement["time"]]
        # print(id)
        model.sinogram_data[id].fix(sinogram_data[i_meas])
        model.rmse_sinogram_expression += (1 / singram_weight[i_meas]) * (
            model.sinogram[id] - model.sinogram_data[id]
        ) ** 2
    return model


def update_image_rmse_expression(model, image_data, image_weight=None):
    if image_weight is None:
        image_weight = np.ones(image_data.shape)
        image_weight = np.diag(image_weight.flatten())
    else:
        image_weight = np.diag(
            image_weight
        )  # Extract diagonal elements to form a diagonal matrix
        image_weight = np.diag(image_weight)  # Ensure it's a diagonal matrix

    image_weight_inv = np.linalg.inv(image_weight)
    image_data_vector = []
    image_varables_vector = []
    for i in model.ix:
        for j in model.iy:
            image_data_vector.append(image_data[i, j])
            image_varables_vector.append(model.image[i, j, 0])

    image_data_vector = np.array(image_data_vector)
    image_varables_vector = np.array(image_varables_vector, dtype=object)

    model.rmse_image_expression = (
        (image_varables_vector - image_data_vector).T
        @ image_weight_inv
        @ (image_varables_vector - image_data_vector)
    )

    return model


def update_image_TV_expression(model, time=0, mode="isotropic"):
    _sqrt = pyo.sqrt
    image = model.image_array[time]
    diff0 = image[1:, :] - image[:-1, :]
    diff1 = image[:, 1:] - image[:, :-1]
    n_image = image.shape[0]
    if mode == "anisotropic":
        tv = 0
        i1, i2 = diff0.shape
        eps = 1e-4
        for i in range(i1):
            for j in range(i2):
                tv += _sqrt(diff0[i, j] ** 2 + eps)

        i1, i2 = diff1.shape
        for i in range(i1):
            for j in range(i2):
                tv += _sqrt(diff1[i, j] ** 2 + eps)
        model.tv_expression = tv
    elif mode == "isotropic":
        tv = 0
        eps = 1e-4
        for i in range(n_image):
            for j in range(n_image):
                if i < n_image - 1:
                    tv0 = diff0[i, j] ** 2
                else:
                    tv0 = 0

                if j < n_image - 1:
                    tv1 = diff1[i, j] ** 2
                else:
                    tv1 = 0

                tv += _sqrt(tv0 + tv1 + eps)
        model.tv_expression = tv
    return model


def build_augmented_nlp(
    image_res,
    base_angles_deg,
    hypothetical_angles_deg,
    r_interval_set,
    sinogram_data_real,
    measurement_set_real,
    predicted_sinograms,
    predicted_measurement_sets,
    I0=0,
    alpha_Dose_Response=0.3,
    beta_Rose_Response=0.01,
    obj_TV_weight=0.1,
    obj_sinogram_weight=1.0,
    solver=None,
    ipopt_tee=False,
):
    """
    Build the augmented NLP (Eq. 252-257) with N hypothetical data pairs.

    Creates a Pyomo model with n_horizon = 1 + N, adds beam constraints
    for real measurements at injection_time=0 and hypothetical measurements
    at injection_time=1..N, sets up the inverse objective with both real
    and predicted sinogram data.

    Parameters
    ----------
    image_res : int
        Image resolution (N x N grid).
    base_angles_deg : list of float
        Angles (degrees) for real measurements at injection_time=0.
    hypothetical_angles_deg : list of list of float
        hypothetical_angles_deg[j] = angles (degrees) for step j+1.
    r_interval_set : ndarray
        Detector positions in image units.
    sinogram_data_real : list of float
        Observed sinogram values from real measurements.
    measurement_set_real : list of dict
        Measurement descriptors for real observations (r, theta, time=0).
    predicted_sinograms : list of list of float
        predicted_sinograms[j] = sinogram values for hypothetical step j+1.
    predicted_measurement_sets : list of list of dict
        predicted_measurement_sets[j] = measurement descriptors for step j+1.
    I0 : float
        Beam intensity for degradation model.
    alpha_Dose_Response, beta_Rose_Response : float
        Degradation model parameters.
    obj_TV_weight, obj_sinogram_weight : float
        Objective weights for TV regularization and sinogram RMSE.
    solver : Pyomo SolverFactory or None
        If provided, the model is solved before returning.
    ipopt_tee : bool
        Whether to print IPOPT output.

    Returns
    -------
    result : dict
        'model'              : Pyomo ConcreteModel (augmented NLP)
        'all_measurement_set': list of dict (real + hypothetical)
        'all_sinogram_data'  : list of float (real + hypothetical)
    """
    import pyomo.environ as pyo

    N = len(hypothetical_angles_deg)
    n_horizon = 1 + N

    # 1. Create model with extended time horizon
    model = create_sample_model(n_horizon=n_horizon, image_res=image_res)
    dummy_image = 0.01 * np.ones((image_res, image_res))
    load_image_to_sample(model, dummy_image)

    # 2. Add beam constraints for real measurements at injection_time=0
    for degree in base_angles_deg:
        meas_k = [
            {"r": float(r), "theta": float(degree * np.pi / 180), "time": 0}
            for r in r_interval_set
        ]
        model = add_beam_constraints_pyomo(
            model,
            meas_k,
            injection_time=0,
            I0=I0,
            alpha_Dose_Response=alpha_Dose_Response,
            beta_Rose_Response=beta_Rose_Response,
            image_res=image_res,
        )

    # 3. Add beam constraints for hypothetical measurements at injection_time=j+1
    for j in range(N):
        for degree in hypothetical_angles_deg[j]:
            meas_j = [
                {"r": float(r), "theta": float(degree * np.pi / 180), "time": j + 1}
                for r in r_interval_set
            ]
            model = add_beam_constraints_pyomo(
                model,
                meas_j,
                injection_time=j + 1,
                I0=I0,
                alpha_Dose_Response=alpha_Dose_Response,
                beta_Rose_Response=beta_Rose_Response,
                image_res=image_res,
            )

    # 4. Combine measurement sets and sinogram data
    all_measurement_set = list(measurement_set_real)
    all_sinogram_data = list(sinogram_data_real)
    for j in range(N):
        all_measurement_set.extend(predicted_measurement_sets[j])
        all_sinogram_data.extend(predicted_sinograms[j])

    # 5. Set up inverse objective
    model = update_sinogram_rmse_expression(
        model,
        all_measurement_set,
        all_sinogram_data,
    )
    model = update_image_TV_expression(model, 0)
    model.image[:, :, 0].free()
    for t in model.time:
        model.image[:, :, t].set_value(0.01)
    W_image = 1 - hamming_window(image_res, two_d=True)
    model = update_image_weigth(model=model, weight=W_image, time=0)
    model.obj = pyo.Objective(
        expr=obj_sinogram_weight * model.rmse_sinogram_expression
        + obj_TV_weight * model.tv_expression,
    )

    # 6. Solve if solver provided
    if solver is not None:
        solver.solve(model, tee=ipopt_tee)

    return {
        "model": model,
        "all_measurement_set": all_measurement_set,
        "all_sinogram_data": all_sinogram_data,
    }


def hamming_window(N: int, two_d: bool = False, hamming_beta=0.46164) -> np.ndarray:
    """
    Generate a Hamming window.

    Parameters
    ----------
    N : int
        Number of points in the window. Must be a non negative integer.
    two_d : bool, optional
        If True, return a 2D separable Hamming window (outer product of 1D window
        with itself). Default is False.

    Returns
    -------
    np.ndarray
        1D array of shape (N,) when two_d is False.
        2D array of shape (N, N) when two_d is True.

    Notes
    -----
    The Hamming window is defined as
        w[n] = 0.54 - 0.46 * cos(2 * pi * n / (N - 1)),  for n = 0, ..., N-1
    For N = 0 an empty array is returned.
    For N = 1 the window is [1.0].
    """
    if not isinstance(N, int):
        raise TypeError("N must be an integer")
    if N < 0:
        raise ValueError("N must be non negative")

    if N == 0:
        w1 = np.empty((0,), dtype=np.float64)
    elif N == 1:
        w1 = np.array([1.0], dtype=np.float64)
    else:
        n = np.arange(N, dtype=np.float64)
        w1 = (1 - hamming_beta) - hamming_beta * np.cos(2.0 * np.pi * n / (N - 1))
        w1 = w1.astype(np.float64, copy=False)

    if two_d:
        # separable 2D window via outer product
        return np.outer(w1, w1)
    return w1


def update_image_weigth(model, time=0, weight=None):
    # Check if W_image has the same size as the image
    image = model.image_array[time]
    if weight.shape != image.shape:
        raise ValueError("W_image must have the same shape as the image")

    ix, iy = image.shape
    image_weigthed = 0
    for i in range(ix):
        for j in range(iy):
            image_weigthed += weight[i, j] * image[i, j] ** 2

    model.weighted_image_expression = image_weigthed
    return model
