"""Pure LDS sufficient-statistic and HDP objective calculations."""

from __future__ import annotations

import math
import numpy as np
import torch
from scipy.special import gammaln, psi

from .OptimizerRhoOmega import kvec


def stable_cholesky(matrix: torch.Tensor, jitter_scale: float = 1e-9) -> torch.Tensor:
    """Cholesky factor of an SPD matrix with scale-aware numerical jitter."""

    matrix = torch.as_tensor(matrix)
    symmetric = 0.5 * (matrix + matrix.T)
    eye = torch.eye(
        symmetric.shape[0], device=symmetric.device, dtype=symmetric.dtype
    )
    scale = torch.mean(torch.diag(symmetric).abs()).clamp_min(
        torch.finfo(symmetric.dtype).eps
    )
    last_error = None
    for multiplier in (0.0, 1.0, 10.0, 100.0, 1_000.0, 10_000.0):
        try:
            return torch.linalg.cholesky(
                symmetric + multiplier * jitter_scale * scale * eye
            )
        except torch.linalg.LinAlgError as error:
            last_error = error
    raise torch.linalg.LinAlgError(
        "matrix is not positive definite after adaptive jitter"
    ) from last_error


def _logdet_spd(matrix: torch.Tensor) -> torch.Tensor:
    factor = stable_cholesky(matrix)
    return 2.0 * torch.log(torch.diag(factor)).sum()


def gaussian_entropy(covariance: torch.Tensor) -> torch.Tensor:
    """Differential entropy of a multivariate Gaussian."""

    covariance = torch.as_tensor(covariance)
    dimension = covariance.shape[0]
    constant = covariance.new_tensor(1.0 + math.log(2.0 * math.pi))
    return 0.5 * (dimension * constant + _logdet_spd(covariance))


def gaussian_conditional_entropy(
    current_covariance: torch.Tensor,
    previous_covariance: torch.Tensor,
    cross_covariance: torch.Tensor,
) -> torch.Tensor:
    """Return H[q(x_t | x_{t-1})] from two smoothing covariance blocks.

    ``cross_covariance`` has orientation ``Cov(x_t, x_{t-1})``.
    """

    previous_factor = stable_cholesky(previous_covariance)
    solved = torch.cholesky_solve(cross_covariance.T, previous_factor).T
    conditional = current_covariance - solved @ cross_covariance.T
    return gaussian_entropy(0.5 * (conditional + conditional.T))


def gaussian_kl(
    q_mean: torch.Tensor,
    q_covariance: torch.Tensor,
    p_mean: torch.Tensor,
    p_covariance: torch.Tensor,
) -> torch.Tensor:
    """KL[q || p] for multivariate Gaussian distributions."""

    q_mean = torch.as_tensor(q_mean)
    p_mean = torch.as_tensor(p_mean, device=q_mean.device, dtype=q_mean.dtype)
    q_covariance = torch.as_tensor(
        q_covariance, device=q_mean.device, dtype=q_mean.dtype
    )
    p_covariance = torch.as_tensor(
        p_covariance, device=q_mean.device, dtype=q_mean.dtype
    )
    if q_mean.ndim == 1:
        q_mean = q_mean[:, None]
    if p_mean.ndim == 1:
        p_mean = p_mean[:, None]
    factor = stable_cholesky(p_covariance)
    trace = torch.trace(torch.cholesky_solve(q_covariance, factor))
    difference = q_mean - p_mean
    quadratic = torch.sum(difference * torch.cholesky_solve(difference, factor))
    dimension = q_covariance.shape[0]
    return 0.5 * (
        trace
        + quadratic
        - dimension
        + _logdet_spd(p_covariance)
        - _logdet_spd(q_covariance)
    )


def expected_gaussian_log_likelihood(
    observation: torch.Tensor,
    mean: torch.Tensor,
    mean_covariance: torch.Tensor,
    noise_covariance: torch.Tensor,
) -> torch.Tensor:
    """E_q[log N(y | latent, R)] for Gaussian latent observation moments."""

    observation = torch.as_tensor(observation)
    mean = torch.as_tensor(mean, device=observation.device, dtype=observation.dtype)
    mean_covariance = torch.as_tensor(
        mean_covariance, device=observation.device, dtype=observation.dtype
    )
    noise_covariance = torch.as_tensor(
        noise_covariance, device=observation.device, dtype=observation.dtype
    )
    if observation.ndim == 1:
        observation = observation[:, None]
    if mean.ndim == 1:
        mean = mean[:, None]
    residual = observation - mean
    scatter = residual @ residual.T + mean_covariance
    factor = stable_cholesky(noise_covariance)
    expected_quadratic = torch.trace(torch.cholesky_solve(scatter, factor))
    dimension = observation.shape[0]
    return -0.5 * (
        expected_quadratic
        + _logdet_spd(noise_covariance)
        + dimension * observation.new_tensor(math.log(2.0 * math.pi))
    )


def inverse_wishart_expected_logdet(
    degrees_of_freedom: float,
    scale: torch.Tensor,
) -> torch.Tensor:
    """E[log |Sigma|] for IW(nu, Psi)."""

    scale = torch.as_tensor(scale)
    dimension = scale.shape[0]
    nu = torch.as_tensor(
        degrees_of_freedom, device=scale.device, dtype=scale.dtype
    )
    indices = torch.arange(dimension, device=scale.device, dtype=scale.dtype)
    multidigamma = torch.digamma(0.5 * (nu - indices)).sum()
    return _logdet_spd(scale) - dimension * math.log(2.0) - multidigamma


def inverse_wishart_kl(
    q_degrees_of_freedom: float,
    q_scale: torch.Tensor,
    p_degrees_of_freedom: float,
    p_scale: torch.Tensor,
) -> torch.Tensor:
    """KL[IW_q || IW_p] using the normalized inverse-Wishart density."""

    q_scale = torch.as_tensor(q_scale)
    p_scale = torch.as_tensor(
        p_scale, device=q_scale.device, dtype=q_scale.dtype
    )
    dimension = q_scale.shape[0]
    q_nu = torch.as_tensor(
        q_degrees_of_freedom, device=q_scale.device, dtype=q_scale.dtype
    )
    p_nu = torch.as_tensor(
        p_degrees_of_freedom, device=q_scale.device, dtype=q_scale.dtype
    )
    q_logdet = _logdet_spd(q_scale)
    p_logdet = _logdet_spd(p_scale)
    log_two = q_scale.new_tensor(math.log(2.0))
    q_constant = (
        0.5 * q_nu * q_logdet
        - 0.5 * q_nu * dimension * log_two
        - torch.mvlgamma(0.5 * q_nu, dimension)
    )
    p_constant = (
        0.5 * p_nu * p_logdet
        - 0.5 * p_nu * dimension * log_two
        - torch.mvlgamma(0.5 * p_nu, dimension)
    )
    expected_logdet = inverse_wishart_expected_logdet(q_nu, q_scale)
    q_factor = stable_cholesky(q_scale)
    expected_precision_times_difference = q_nu * torch.trace(
        torch.cholesky_solve(q_scale - p_scale, q_factor)
    )
    return (
        q_constant
        - p_constant
        - 0.5 * (q_nu - p_nu) * expected_logdet
        - 0.5 * expected_precision_times_difference
    )


def matrix_normal_inverse_wishart_kl(
    q_mean: torch.Tensor,
    q_column_precision: torch.Tensor,
    q_degrees_of_freedom: float,
    q_scale: torch.Tensor,
    p_mean: torch.Tensor,
    p_column_precision: torch.Tensor,
    p_degrees_of_freedom: float,
    p_scale: torch.Tensor,
) -> torch.Tensor:
    """KL between matrix-normal inverse-Wishart distributions."""

    q_mean = torch.as_tensor(q_mean)
    device, dtype = q_mean.device, q_mean.dtype
    q_precision = torch.as_tensor(q_column_precision, device=device, dtype=dtype)
    p_precision = torch.as_tensor(p_column_precision, device=device, dtype=dtype)
    p_mean = torch.as_tensor(p_mean, device=device, dtype=dtype)
    q_scale = torch.as_tensor(q_scale, device=device, dtype=dtype)
    p_scale = torch.as_tensor(p_scale, device=device, dtype=dtype)
    row_dimension, column_dimension = q_mean.shape

    q_precision_factor = stable_cholesky(q_precision)
    q_column_covariance = torch.cholesky_inverse(q_precision_factor)
    column_trace = torch.trace(p_precision @ q_column_covariance)
    logdet_q_precision = _logdet_spd(q_precision)
    logdet_p_precision = _logdet_spd(p_precision)

    q_scale_factor = stable_cholesky(q_scale)
    expected_precision = float(q_degrees_of_freedom) * torch.cholesky_inverse(
        q_scale_factor
    )
    difference = q_mean - p_mean
    mean_quadratic = torch.trace(
        expected_precision @ difference @ p_precision @ difference.T
    )
    matrix_normal_kl = 0.5 * (
        row_dimension * column_trace
        - row_dimension * column_dimension
        + row_dimension * (logdet_q_precision - logdet_p_precision)
        + mean_quadratic
    )
    return matrix_normal_kl + inverse_wishart_kl(
        q_degrees_of_freedom,
        q_scale,
        p_degrees_of_freedom,
        p_scale,
    )


def expected_residual_scatter(
    current_mean: torch.Tensor,
    previous_mean: torch.Tensor,
    transform: torch.Tensor,
    current_covariance: torch.Tensor,
    previous_covariance: torch.Tensor,
    cross_covariance: torch.Tensor,
) -> torch.Tensor:
    """Return E[(x_t - A x_{t-1})(x_t - A x_{t-1})^T].

    ``cross_covariance`` follows the explicit orientation
    ``Cov(x_t, x_{t-1})`` and is generally not symmetric.
    """

    residual = current_mean - transform @ previous_mean
    scatter = (
        residual @ residual.T
        + current_covariance
        + transform @ previous_covariance @ transform.T
        - cross_covariance @ transform.T
        - transform @ cross_covariance.T
    )
    return 0.5 * (scatter + scatter.T)


def hdp_top_level_term(
    rho: np.ndarray,
    omega: np.ndarray,
    alpha: float,
    start_alpha: float,
    kappa: float,
    gamma: float,
    expected_beta: np.ndarray | None = None,
) -> float:
    """Top-level stick-breaking ELBO term used by the rho/omega optimizer."""

    rho = np.asarray(rho, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    if rho.ndim != 1 or omega.shape != rho.shape:
        raise ValueError("rho and omega must be one-dimensional arrays of equal length")
    if np.any((rho <= 0) | (rho >= 1)) or np.any(omega <= 0):
        raise ValueError("rho must be in (0, 1) and omega must be positive")
    K = rho.size
    eta1 = rho * omega
    eta0 = (1.0 - rho) * omega
    elog_u = psi(eta1) - psi(omega)
    elog_1mu = psi(eta0) - psi(omega)
    c_beta = lambda a1, a0: np.sum(gammaln(a1 + a0) - gammaln(a1) - gammaln(a0))
    beta_normalizers = K * c_beta(np.ones(K), np.full(K, gamma)) - c_beta(eta1, eta0)
    concentration_term = K * K * np.log(alpha) + K * np.log(start_alpha)

    if kappa > 0:
        coef_u = K + 1.0 - eta1
        coef_1mu = K * kvec(K) + 1.0 + gamma - eta0
        if expected_beta is None:
            expected_beta = np.hstack([rho, 1.0])
            expected_beta[1:] *= np.cumprod(1.0 - rho)
            expected_beta = expected_beta[:-1]
        sticky_beta = np.sum(expected_beta) * (np.log(alpha + kappa) - np.log(kappa))
        sticky_constant = K * (np.log(kappa) - np.log(alpha + kappa))
    else:
        coef_u = K + 2.0 - eta1
        coef_1mu = (K + 1.0) * kvec(K) + gamma - eta0
        sticky_beta = 0.0
        sticky_constant = 0.0

    return float(
        concentration_term
        + sticky_constant
        + sticky_beta
        + beta_normalizers
        + np.inner(coef_u, elog_u)
        + np.inner(coef_1mu, elog_1mu)
    )
