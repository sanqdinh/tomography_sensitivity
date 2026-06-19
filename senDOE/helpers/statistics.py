"""
Statistics utilities for sequential design of experiments.

Provides numerically stable computation of D-optimality criteria
for large covariance matrices.
"""

import numpy as np


def log_determinant(Sigma, method='auto', n_probes=30, n_lanczos=50, seed=None):
    """
    Compute or estimate log(det(Sigma)) for a symmetric positive
    (semi-)definite matrix.

    Parameters
    ----------
    Sigma : ndarray, shape (p, p)
        Covariance matrix (symmetric positive semi-definite).
    method : {'auto', 'cholesky', 'eigenvalue', 'stochastic'}
        - 'cholesky': Exact via Cholesky decomposition. O(p^3/3).
          Fastest for well-conditioned SPD matrices. Falls back to
          'eigenvalue' if the matrix is not positive definite.
        - 'eigenvalue': Exact via symmetric eigendecomposition. O(p^3).
          Handles rank-deficient matrices by summing log of positive
          eigenvalues only.
        - 'stochastic': Stochastic Lanczos Quadrature estimate.
          O(n_probes * n_lanczos * p^2). Useful when p is very large.
        - 'auto': Uses 'cholesky' for p <= 5000, 'stochastic' otherwise.
    n_probes : int
        Number of random probe vectors (stochastic method only).
    n_lanczos : int
        Number of Lanczos iterations per probe (stochastic method only).
        Capped at p internally.
    seed : int or None
        Random seed for reproducibility (stochastic method only).

    Returns
    -------
    logdet : float
        log(det(Sigma)), or sum of log of positive eigenvalues if
        rank-deficient.
    """
    p = Sigma.shape[0]
    if Sigma.shape != (p, p):
        raise ValueError(f"Expected square matrix, got shape {Sigma.shape}")

    if method == 'auto':
        method = 'cholesky' if p <= 5000 else 'stochastic'

    if method == 'cholesky':
        return _logdet_cholesky(Sigma)
    elif method == 'eigenvalue':
        return _logdet_eigenvalue(Sigma)
    elif method == 'stochastic':
        return _logdet_stochastic(Sigma, n_probes, n_lanczos, seed)
    else:
        raise ValueError(f"Unknown method '{method}'")


def d_optimality(Sigma, **kwargs):
    """
    D-optimality criterion: (1/p) * log(det(Sigma)).

    For experimental design, smaller values indicate tighter parameter
    uncertainty (better designs). This is the log of the geometric mean
    of the eigenvalues.

    Parameters
    ----------
    Sigma : ndarray, shape (p, p)
        Covariance matrix.
    **kwargs
        Passed to log_determinant (method, n_probes, n_lanczos, seed).

    Returns
    -------
    d_opt : float
    """
    p = Sigma.shape[0]
    return log_determinant(Sigma, **kwargs) / p


def a_optimality(Sigma):
    """
    A-optimality criterion: trace(Sigma).

    The trace of the covariance matrix equals the sum of the eigenvalues,
    which is the sum of the parameter variances. Smaller values indicate
    tighter overall parameter uncertainty (better designs).

    Parameters
    ----------
    Sigma : ndarray, shape (p, p)
        Covariance matrix (symmetric positive semi-definite).

    Returns
    -------
    a_opt : float
    """
    p = Sigma.shape[0]
    if Sigma.shape != (p, p):
        raise ValueError(f"Expected square matrix, got shape {Sigma.shape}")
    return np.trace(Sigma)


# ── Covariance update utilities ──────────────────────────────────────


def predicted_observation_covariance(C_u, Sigma_k, Sigma_eps):
    """
    Predicted observation covariance (Eq. 16).

    S_{k+1}(u) = C_u @ Sigma_k @ C_u^T + Sigma_eps

    Accounts for both state estimation uncertainty (propagated through the
    observation Jacobian) and measurement noise.

    Parameters
    ----------
    C_u : ndarray, shape (n_y, n_f)
        Observation Jacobian at the current estimate for candidate action u.
    Sigma_k : ndarray, shape (n_f, n_f)
        Current posterior covariance of the state estimate.
    Sigma_eps : ndarray, shape (n_y, n_y)
        Measurement noise covariance.

    Returns
    -------
    S : ndarray, shape (n_y, n_y)
        Predicted observation covariance (symmetric positive definite).
    """
    n_y, n_f = C_u.shape
    if Sigma_k.shape != (n_f, n_f):
        raise ValueError(
            f"Sigma_k shape {Sigma_k.shape} inconsistent with C_u columns {n_f}"
        )
    if Sigma_eps.shape != (n_y, n_y):
        raise ValueError(
            f"Sigma_eps shape {Sigma_eps.shape} inconsistent with C_u rows {n_y}"
        )
    S = C_u @ Sigma_k @ C_u.T + Sigma_eps
    S = 0.5 * (S + S.T)
    return S


def covariance_update(Sigma_k, C_u, Sigma_eps):
    """
    Kalman-style covariance update (Eq. 23).

    Sigma_{k+1|k}(u) = Sigma_k - Sigma_k C_u^T S^{-1} C_u Sigma_k

    where S = C_u Sigma_k C_u^T + Sigma_eps (Eq. 16).

    Uses scipy.linalg.solve (never forms inv(S) explicitly) for
    numerical stability.

    Parameters
    ----------
    Sigma_k : ndarray, shape (n_f, n_f)
        Current posterior covariance.
    C_u : ndarray, shape (n_y, n_f)
        Observation Jacobian for candidate action u.
    Sigma_eps : ndarray, shape (n_y, n_y)
        Measurement noise covariance.

    Returns
    -------
    Sigma_post : ndarray, shape (n_f, n_f)
        Updated (posterior) covariance (symmetric positive semi-definite).
    """
    from scipy.linalg import solve

    S = predicted_observation_covariance(C_u, Sigma_k, Sigma_eps)
    # Solve S @ X = C_u @ Sigma_k for X, shape (n_y, n_f)
    X = solve(S, C_u @ Sigma_k, assume_a='pos')
    # Sigma_post = Sigma_k - Sigma_k @ C_u.T @ X
    Sigma_post = Sigma_k - (Sigma_k @ C_u.T) @ X
    Sigma_post = 0.5 * (Sigma_post + Sigma_post.T)
    return Sigma_post


# ── Information-space utilities ─────────────────────────────────────


def fisher_information_matrix(C_u, Sigma_eps):
    """
    Fisher information matrix for a single measurement action (Eq. 33).

    F(u) = C_u^T @ Sigma_eps^{-1} @ C_u

    Measures how much information a measurement at action u provides
    about the parameters. Enters the information dynamics additively.

    Parameters
    ----------
    C_u : ndarray, shape (n_y, n_f)
        Observation Jacobian at the current estimate for candidate action u.
    Sigma_eps : ndarray, shape (n_y, n_y)
        Measurement noise covariance (symmetric positive definite).

    Returns
    -------
    F : ndarray, shape (n_f, n_f)
        Fisher information matrix (symmetric positive semi-definite).
    """
    from scipy.linalg import solve

    n_y, n_f = C_u.shape
    if Sigma_eps.shape != (n_y, n_y):
        raise ValueError(
            f"Sigma_eps shape {Sigma_eps.shape} inconsistent with C_u rows {n_y}"
        )
    # Compute Sigma_eps^{-1} @ C_u via solve (avoids explicit inverse)
    Sinv_C = solve(Sigma_eps, C_u, assume_a='pos')  # (n_y, n_f)
    F = C_u.T @ Sinv_C  # (n_f, n_f)
    F = 0.5 * (F + F.T)
    return F


def information_matrix_update(Omega, F_u):
    """
    Single-step information matrix update (Eq. 34).

    Omega_{k+1} = Omega_k + F(u)

    The information matrix grows additively with each measurement,
    reflecting that information accumulates linearly.

    Parameters
    ----------
    Omega : ndarray, shape (n_f, n_f)
        Current information matrix (Sigma^{-1}).
    F_u : ndarray, shape (n_f, n_f)
        Fisher information matrix for the measurement action.

    Returns
    -------
    Omega_new : ndarray, shape (n_f, n_f)
        Updated information matrix (symmetric).
    """
    n_f = Omega.shape[0]
    if Omega.shape != (n_f, n_f):
        raise ValueError(f"Expected square Omega, got shape {Omega.shape}")
    if F_u.shape != (n_f, n_f):
        raise ValueError(
            f"F_u shape {F_u.shape} inconsistent with Omega size {n_f}"
        )
    Omega_new = Omega + F_u
    Omega_new = 0.5 * (Omega_new + Omega_new.T)
    return Omega_new


def information_matrix_trajectory(Omega_0, F_list):
    """
    Apply a sequence of Fisher information matrices (Eqs. 34, 39).

    Returns the full trajectory [Omega_0, Omega_1, ..., Omega_J] where
    Omega_j = Omega_0 + sum_{i=1}^{j} F_i (telescoping form).

    Parameters
    ----------
    Omega_0 : ndarray, shape (n_f, n_f)
        Initial information matrix (typically Sigma_k^{-1}).
    F_list : list of ndarray, each shape (n_f, n_f)
        Sequence of Fisher information matrices [F_1, F_2, ..., F_J].

    Returns
    -------
    trajectory : list of ndarray
        [Omega_0, Omega_1, ..., Omega_J], length len(F_list) + 1.
    """
    trajectory = [Omega_0.copy()]
    Omega = Omega_0.copy()
    for F_u in F_list:
        Omega = information_matrix_update(Omega, F_u)
        trajectory.append(Omega)
    return trajectory


def d_optimality_information(Omega, **kwargs):
    """
    D-optimality criterion in information space (Eq. 42).

    phi_D = -(1/n) * log det(Omega)

    Equivalent to d_optimality(Sigma) when Omega = Sigma^{-1}, since
    -(1/n) log det(Omega) = (1/n) log det(Sigma).

    Parameters
    ----------
    Omega : ndarray, shape (n_f, n_f)
        Information matrix (symmetric positive definite).
    **kwargs
        Passed to log_determinant (method, n_probes, n_lanczos, seed).

    Returns
    -------
    phi_D : float
        D-optimality value in information space (smaller = more information).
    """
    n = Omega.shape[0]
    return -log_determinant(Omega, **kwargs) / n


def a_optimality_information(Omega):
    """
    A-optimality criterion in information space (Eq. 43).

    phi_A = tr(Omega^{-1})

    Equivalent to a_optimality(Sigma) when Omega = Sigma^{-1}, since
    tr(Omega^{-1}) = tr(Sigma).

    Parameters
    ----------
    Omega : ndarray, shape (n_f, n_f)
        Information matrix (symmetric positive definite).

    Returns
    -------
    phi_A : float
        A-optimality value in information space (smaller = more information).
    """
    n = Omega.shape[0]
    if Omega.shape != (n, n):
        raise ValueError(f"Expected square matrix, got shape {Omega.shape}")
    Sigma = np.linalg.inv(Omega)
    return np.trace(Sigma)


def select_best_action_information(Omega_k, candidates, observation_jacobian_fn,
                                   Sigma_eps, criterion='d_optimality'):
    """
    Screen candidate actions in information space.

    For each candidate u, computes F(u) = C_u^T Sigma_eps^{-1} C_u and
    evaluates the design criterion on Omega_k + F(u).

    Parameters
    ----------
    Omega_k : ndarray, shape (n_f, n_f)
        Current information matrix (Sigma_k^{-1}).
    candidates : array-like
        Iterable of candidate actions.
    observation_jacobian_fn : callable
        Function that takes a candidate action and returns C_u,
        shape (n_y, n_f).
    Sigma_eps : ndarray, shape (n_y, n_y)
        Measurement noise covariance.
    criterion : {'d_optimality', 'a_optimality'} or callable
        Design criterion to minimize. If a string, uses the corresponding
        information-space function.

    Returns
    -------
    best_action : object
        The candidate action with the lowest criterion value.
    best_criterion : float
        The criterion value at the best action.
    criterion_values : ndarray, shape (n_candidates,)
        Criterion values for all candidates.
    """
    if criterion == 'd_optimality':
        criterion_fn = d_optimality_information
    elif criterion == 'a_optimality':
        criterion_fn = a_optimality_information
    elif callable(criterion):
        criterion_fn = criterion
    else:
        raise ValueError(f"Unknown criterion '{criterion}'")

    candidates = list(candidates)
    criterion_values = np.empty(len(candidates))

    for i, u in enumerate(candidates):
        C_u = observation_jacobian_fn(u)
        F_u = fisher_information_matrix(C_u, Sigma_eps)
        Omega_new = information_matrix_update(Omega_k, F_u)
        criterion_values[i] = criterion_fn(Omega_new)

    best_idx = np.argmin(criterion_values)
    return candidates[best_idx], criterion_values[best_idx], criterion_values


def select_best_action(Sigma_k, candidates, observation_jacobian_fn, Sigma_eps,
                       criterion='d_optimality'):
    """
    Screen candidate actions and select the one minimizing the design criterion.

    For each candidate u, computes the Kalman-updated posterior covariance
    and evaluates the design criterion. Returns the candidate with the
    minimum criterion value.

    Parameters
    ----------
    Sigma_k : ndarray, shape (n_f, n_f)
        Current posterior covariance.
    candidates : array-like
        Iterable of candidate actions (e.g., angles in radians).
    observation_jacobian_fn : callable
        Function that takes a candidate action and returns C_u,
        shape (n_y, n_f).
    Sigma_eps : ndarray, shape (n_y, n_y)
        Measurement noise covariance.
    criterion : {'d_optimality', 'a_optimality'} or callable
        Design criterion to minimize. If a string, uses the corresponding
        function from this module.

    Returns
    -------
    best_action : object
        The candidate action with the lowest criterion value.
    best_criterion : float
        The criterion value at the best action.
    criterion_values : ndarray, shape (n_candidates,)
        Criterion values for all candidates.
    """
    if criterion == 'd_optimality':
        criterion_fn = d_optimality
    elif criterion == 'a_optimality':
        criterion_fn = a_optimality
    elif callable(criterion):
        criterion_fn = criterion
    else:
        raise ValueError(f"Unknown criterion '{criterion}'")

    candidates = list(candidates)
    criterion_values = np.empty(len(candidates))

    for i, u in enumerate(candidates):
        C_u = observation_jacobian_fn(u)
        Sigma_post = covariance_update(Sigma_k, C_u, Sigma_eps)
        criterion_values[i] = criterion_fn(Sigma_post)

    best_idx = np.argmin(criterion_values)
    return candidates[best_idx], criterion_values[best_idx], criterion_values


# ── Multi-step covariance utilities ────────────────────────────────────


def correlated_predicted_observation_covariance(G, Sigma_k, Sigma_eps, N):
    """
    Correlated predicted observation covariance (Eq. 281).

    S_pred = G @ Sigma_k @ G^T + I_N x Sigma_eps

    where G = [G_1; ...; G_N] is the stacked observation Jacobian and
    the Kronecker product I_N x Sigma_eps adds independent measurement
    noise at each future step.

    Parameters
    ----------
    G : ndarray, shape (N*n_y, n_f)
        Stacked observation Jacobians from compute_stacked_observation_jacobians.
    Sigma_k : ndarray, shape (n_f, n_f)
        Current posterior covariance.
    Sigma_eps : ndarray, shape (n_y, n_y)
        Per-step measurement noise covariance.
    N : int
        Number of future steps.

    Returns
    -------
    S_pred : ndarray, shape (N*n_y, N*n_y)
        Correlated predicted observation covariance (symmetric, PD).
    """
    noise_block = np.kron(np.eye(N), Sigma_eps)
    S_pred = G @ Sigma_k @ G.T + noise_block
    S_pred = 0.5 * (S_pred + S_pred.T)
    return S_pred


def multi_step_posterior_covariance(J_k_plus_N, Sigma_eps, S_pred, k_plus_1):
    """
    Multi-step posterior covariance (Eq. 290).

    Sigma_{k+N|k} = J_k^{+N} @ blkdiag(I_{k+1} x Sigma_eps, S_pred) @ (J_k^{+N})^T

    Parameters
    ----------
    J_k_plus_N : ndarray, shape (n_f, (k+1+N)*n_y)
        Multi-step sensitivity matrix from augmented NLP.
    Sigma_eps : ndarray, shape (n_y, n_y)
        Per-step measurement noise covariance.
    S_pred : ndarray, shape (N*n_y, N*n_y)
        Correlated predicted observation covariance from Step 4.4.
    k_plus_1 : int
        Number of real observation blocks (k+1).

    Returns
    -------
    Sigma_post : ndarray, shape (n_f, n_f)
        Multi-step posterior covariance (symmetric, PSD).
    """
    from scipy.linalg import block_diag

    real_block = np.kron(np.eye(k_plus_1), Sigma_eps)
    noise_cov = block_diag(real_block, S_pred)
    Sigma_post = J_k_plus_N @ noise_cov @ J_k_plus_N.T
    Sigma_post = 0.5 * (Sigma_post + Sigma_post.T)
    return Sigma_post


# ── Exact methods ──────────────────────────────────────────────────────


def _logdet_cholesky(Sigma):
    """Exact log-determinant via Cholesky. O(p^3/3).

    Falls back to eigenvalue method if Cholesky fails (matrix not SPD).
    """
    try:
        L = np.linalg.cholesky(Sigma)
        return 2.0 * np.sum(np.log(np.diag(L)))
    except np.linalg.LinAlgError:
        return _logdet_eigenvalue(Sigma)


def _logdet_eigenvalue(Sigma):
    """Exact log-determinant via symmetric eigendecomposition. O(p^3).

    Handles rank-deficient matrices by summing only over eigenvalues
    above a relative tolerance.
    """
    eigvals = np.linalg.eigvalsh(Sigma)
    tol = np.max(np.abs(eigvals)) * Sigma.shape[0] * np.finfo(Sigma.dtype).eps
    pos = eigvals[eigvals > tol]
    if len(pos) == 0:
        return -np.inf
    return np.sum(np.log(pos))


# ── Stochastic method ─────────────────────────────────────────────────


def _lanczos(Sigma, q1, m):
    """
    m-step Lanczos iteration with full reorthogonalization.

    Given symmetric matrix Sigma and unit starting vector q1, produces
    a tridiagonal matrix T such that the eigenvalues of T approximate
    the extremal eigenvalues of Sigma.

    Parameters
    ----------
    Sigma : ndarray, shape (p, p)
    q1 : ndarray, shape (p,)
        Unit starting vector.
    m : int
        Number of Lanczos steps (capped at p).

    Returns
    -------
    T : ndarray, shape (k, k)
        Symmetric tridiagonal matrix, k <= m.
    """
    p = Sigma.shape[0]
    m = min(m, p)

    Q = np.empty((m, p))
    alpha = np.empty(m)
    beta = np.empty(m)

    Q[0] = q1
    k = m

    for j in range(m):
        v = Sigma @ Q[j]
        if j > 0:
            v -= beta[j - 1] * Q[j - 1]
        alpha[j] = Q[j] @ v
        v -= alpha[j] * Q[j]

        # Full reorthogonalization against all previous vectors.
        # Essential for numerical stability of the log-det estimate.
        coeffs = Q[:j + 1] @ v
        v -= Q[:j + 1].T @ coeffs

        beta_j = np.linalg.norm(v)

        if j < m - 1:
            if beta_j < 1e-12:
                k = j + 1
                break
            beta[j] = beta_j
            Q[j + 1] = v / beta_j

    T = np.diag(alpha[:k])
    if k > 1:
        T += np.diag(beta[:k - 1], 1) + np.diag(beta[:k - 1], -1)
    return T


def _logdet_stochastic(Sigma, n_probes, n_lanczos, seed):
    """
    Stochastic log-determinant via Stochastic Lanczos Quadrature (SLQ).

    Uses Hutchinson's trace estimator combined with Lanczos quadrature:
        log(det(A)) = tr(log(A)) ≈ (1/s) sum_i  z_i^T log(A) z_i

    where z_i are Rademacher random vectors (entries ±1). Each quadratic
    form z^T log(A) z is approximated by projecting onto the Krylov
    subspace via Lanczos.

    Reference: Ubaru, Chen, Saad (2017). "Fast Estimation of tr(f(A))
    via Stochastic Lanczos Quadrature."
    """
    rng = np.random.default_rng(seed)
    p = Sigma.shape[0]
    m = min(n_lanczos, p)

    estimates = np.empty(n_probes)

    for i in range(n_probes):
        # Rademacher probe vector: entries ±1 with equal probability.
        # ||z||^2 = p exactly.
        z = rng.choice([-1.0, 1.0], size=p)
        q1 = z / np.sqrt(p)  # unit vector

        T = _lanczos(Sigma, q1, m)

        # Eigendecompose the small tridiagonal matrix (size k x k).
        eigs, V = np.linalg.eigh(T)

        # Clamp tiny/negative eigenvalues (numerical noise in Lanczos).
        eigs = np.maximum(eigs, 1e-30)

        # z^T log(A) z ≈ ||z||^2 * e_1^T V log(Λ) V^T e_1
        #               = p * sum_j  V[0,j]^2 * log(eigs[j])
        estimates[i] = p * np.dot(V[0] ** 2, np.log(eigs))

    return np.mean(estimates)
