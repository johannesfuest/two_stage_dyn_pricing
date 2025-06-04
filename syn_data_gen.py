import statsmodels.api as sm
import numpy as np
import math
import random
import torch
from scipy.stats import uniform_direction, truncnorm
from typing import List, Union

# set seed for reproducibility
#random.seed(42)
#np.random.seed(42)

_TensorLike = Union[float, torch.Tensor]
_SQRT2   = math.sqrt(2.0)
_INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)


def gen_noise(
    n_episodes: int,
    mu_noise: float = 0.05,
    sigma_noise: float = 0.05,
    a_noise: float = 0.02,
    b_noise: float = 0.11,
    seed: int = None,
) -> np.ndarray:
    """
    Generate truncated-normal noise for the synthetic time-series data.

    Parameters
    ----------
    n_episodes : int
        k, the number of episodes (we start counting at 1).
    mu_noise : float, default 0.05
        Mean of the underlying (untruncated) normal.
    sigma_noise : float, default 0.05
        Standard deviation of the underlying normal.
    a_noise : float, default 0.02
        Lower bound (inclusive) of the noise.
    b_noise : float, default 0.11
        Upper bound (inclusive) of the noise.

    Returns
    -------
    np.ndarray
        One-dimensional array of length containing the noise.
    """
    if n_episodes < 2:
        raise ValueError("n_episodes must be at least 2.")

    # total number of samples across all episodes
    total_samples = sum(2 ** (k - 1) for k in range(1, n_episodes + 1))

    # SciPy expects bounds in “standard normal” units
    a_std = (a_noise - mu_noise) / sigma_noise
    b_std = (b_noise - mu_noise) / sigma_noise

    # draw the noise
    noise = truncnorm.rvs(
        a_std,
        b_std,
        loc=mu_noise,
        scale=sigma_noise,
        size=total_samples,
        random_state=seed,  # add a seed here if you need reproducibility
    )
    return noise


def noise_pdf(
    x: float,
    mu_noise: float = 0.05,
    sigma_noise: float = 0.05,
    a_noise: float = 0.02,
    b_noise: float = 0.11,
) -> float:
    """
    Generate the PDF of the truncated normal distribution.

    Parameters
    ----------
    x:  float
        The value at which to evaluate the PDF.
    mu_noise : float, default 0.05
        Mean of the underlying (untruncated) normal.
    sigma_noise : float, default 0.05
        Standard deviation of the underlying normal.
    a_noise : float, default 0.02
        Lower bound (inclusive) of the noise.
    b_noise : float, default 0.11
        Upper bound (inclusive) of the noise.

    Returns
    -------
    float
        The value of the PDF at the given x.
    """
    a_std = (a_noise - mu_noise) / sigma_noise
    b_std = (b_noise - mu_noise) / sigma_noise

    return truncnorm(a_std, b_std, loc=mu_noise, scale=sigma_noise).pdf(x)


def noise_cdf(
    x: float,
    mu_noise: float = 0.05,
    sigma_noise: float = 0.05,
    a_noise: float = 0.02,
    b_noise: float = 0.11,
) -> float:
    """
    Generate the CDF of the truncated normal distribution.
    Parameters
    ----------
    x:  float
        The value at which to evaluate the CDF.
    mu_noise : float, default 0.05
        Mean of the underlying (untruncated) normal.
    sigma_noise : float, default 0.05
        Standard deviation of the underlying normal.
    a_noise : float, default 0.02
        Lower bound (inclusive) of the noise.
    b_noise : float, default 0.11
        Upper bound (inclusive) of the noise.
    Returns
    -------
    function
        CDF of the truncated normal distribution.
    """
    a_std = (a_noise - mu_noise) / sigma_noise
    b_std = (b_noise - mu_noise) / sigma_noise

    return truncnorm(a_std, b_std, loc=mu_noise, scale=sigma_noise).cdf(x)

def _phi(z: torch.Tensor) -> torch.Tensor:
    """Standard–normal pdf  φ(z) = exp(-z²/2)/√(2π)."""
    return torch.exp(-0.5 * z * z) * _INV_SQRT2PI


def _Phi(z: torch.Tensor) -> torch.Tensor:
    """Standard–normal cdf  Φ(z)  (via erf, which is autograd-friendly)."""
    return 0.5 * (1.0 + torch.erf(z / _SQRT2))


def noise_pdf_torch(
    x: _TensorLike,
    mu: _TensorLike,
    sigma: _TensorLike,
    a: _TensorLike,
    b: _TensorLike,
) -> torch.Tensor:
    """
    PDF of N(mu, sigma²) truncated to [a, b].

    Returned tensor broadcasts over all inputs.
    """
    # promote to tensors on same dtype/device
    x, mu, sigma, a, b = map(
        lambda t: torch.as_tensor(t, dtype=torch.float64, device='cpu'),
        (x, mu, sigma, a, b),
    )

    if (sigma <= 0).any():
        raise ValueError("sigma must be positive.")

    # standardised variables
    z      = (x - mu) / sigma
    alpha  = (a - mu) / sigma
    beta   = (b - mu) / sigma
    Z      = _Phi(beta) - _Phi(alpha)        # normalising constant

    # core pdf
    base   = _phi(z) / (sigma * Z)
    # zero outside the support (keeps autograd, avoids NaNs)
    return torch.where((x < a) | (x > b), torch.zeros_like(base), base)


def noise_cdf_torch(
    x: _TensorLike,
    mu: _TensorLike,
    sigma: _TensorLike,
    a: _TensorLike,
    b: _TensorLike,
) -> torch.Tensor:
    """
    CDF of N(mu, sigma²) truncated to [a, b].

    Values below a → 0, above b → 1.
    """
    x, mu, sigma, a, b = map(
        lambda t: torch.as_tensor(t, dtype=torch.float64, device='cpu'),
        (x, mu, sigma, a, b),
    )

    if (sigma <= 0).any():
        raise ValueError("sigma must be positive.")

    z      = (x - mu) / sigma
    alpha  = (a - mu) / sigma
    beta   = (b - mu) / sigma
    Z      = _Phi(beta) - _Phi(alpha)

    cdf_raw = (_Phi(z) - _Phi(alpha)) / Z
    # clamp to [0,1] outside support
    return torch.where(
        x < a, torch.zeros_like(cdf_raw),
        torch.where(x > b, torch.ones_like(cdf_raw), cdf_raw),
    )


def gen_theta_star(
    d: int = 30,
    seed: int = None,
) -> np.ndarray:
    """
    Generate theta star for the time series data (sampled from the unit sphere).
    Args:
        d (int): Dimension of the context. Default is 30.
        seed (int, optional): Random seed for reproducibility.
    Returns:
        np.ndarray: theta star
    """
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(d)
    u /= np.linalg.norm(u)
    return u


def gen_deltas(
    M: int = 50,
    d: int = 30,
    delta_max: float = 2.0,
    seed: int = None,
) -> np.ndarray:
    """
    Generate deltas for the time series data.
    Args:
        M (int): Number of securities. Default is 50.
        d (int): Dimension of the context. Default is 30.
        delta_max (float): Maximum delta. Default is 2.0.
        seed (int, optional): Random seed for reproducibility.
    Returns:
        np.ndarray: delta_js for the synthetic data.
    """
    rng = np.random.default_rng(seed)
    covariance_matrix = 0.2 * np.eye(d) + np.ones((d, d))
    samples = rng.multivariate_normal(
        mean=np.zeros(d),
        cov=covariance_matrix,
        size=M
    )
    # Normalize the samples
    for i, sample in enumerate(samples):
        factor = delta_max / np.linalg.norm(sample)
        samples[i] = sample * factor
    return samples


def gen_theta_stars(
    theta_star: np.ndarray,
    deltas: np.ndarray,
) -> np.ndarray:
    """
    Generate theta star js for the time series data.
    Args:
        theta_star (np.ndarray): theta star base for the time series data.
        deltas (np.ndarray): deltas for the time series data.
    Returns:
        np.ndarray: theta stars for the synthetic data.
    """
    theta_stars = []
    for delta in deltas:
        theta_stars.append(theta_star + delta)
    return np.array(theta_stars)


def gen_x_ts(
    n_rounds: int,
    d: int = 30,
    seed: int = None,
    x_max_vector: Union[float, np.ndarray] = 1.0,
) -> np.ndarray:
    """
    Generate x_t context vectors for each round t, sampled i.i.d. from N(0, I_d),
    then scaled to have specified L2 norms.

    Parameters
    ----------
    n_rounds : int
        Number of time steps.
    d : int
        Context dimension.
    seed : int
        Random seed.
    x_max_vector : float or np.ndarray
        Scalar or array of shape (n_rounds,) giving per-x_t L2 norms.

    Returns
    -------
    np.ndarray
        Shape (n_rounds, d), each row scaled to norm x_max_vector[i].
    """
    rng = np.random.default_rng(seed)
    x_t = rng.normal(0, 1, size=(n_rounds, d))
    x_t = x_t / np.linalg.norm(x_t, axis=1, keepdims=True)  # Normalize to unit norm

    return x_t


def gen_n_remaining_payments(
    M: int = 50,
    seed: int = None,
) -> np.ndarray:
    """
    Generate the number of remaining payments per security, sampled uniformly from [10, 50] (inclusive).

    Parameters
    ----------
    M : int
        Number of securities.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of integers of shape (M,), each in [10, 50].
    """
    rng = np.random.default_rng(seed)
    return rng.integers(10, 51, size=M)

def gen_coupon_rates(
    M: int = 50,
    seed: int = None,
) -> np.ndarray:
    """
    Generate coupon rates for the synthetic data, uniformly distributed in [0.02, 0.1].

    Args:
        M (int): Number of securities. Default is 50.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: Coupon rates for the synthetic data.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(0.02, 0.1, size=M)



def gen_arrivals(
    M: int = 50,
    mode: str = "uniform",     # "uniform", "poly", or "exp"
    alpha: float = 2.0,        # decay rate (0, 1, 2, 3 in paper plots)
    n_episodes: int = 10,       # highest round index k (starts at 1)
    seed: int = None,
) -> List[List[int]]:
    """
    # TODO: what exactly is quadratic decay?
    Draw 2**(k-1) i.i.d. samples from {0, …, M-1} for each round k = 1 … n_episodes.

    Returns
    -------
    arrivals : List[List[int]]
        arrivals[k-2] holds the samples for round k.
    """
    if M < 1:
        raise ValueError("M must be positive")
    mode = mode.lower()
    if mode not in {"uniform", "poly", "exp"}:
        raise ValueError("mode must be 'uniform', 'poly', or 'exp'")

    # --- build probability vector p ---
    if mode == "uniform":
        p = np.full(M, 1.0 / M)
    elif mode == "poly":
        weights = (np.arange(1, M + 1) ** (-alpha)).astype(float)
        p = weights / weights.sum()
    elif mode == "exp":
        weights = np.exp(-alpha * np.arange(M)).astype(float)
        p = weights / weights.sum()
    else:
        raise ValueError("mode must be 'uniform', 'poly', or 'exp'")
    rng = np.random.default_rng(seed)
    arrivals: List[List[int]] = []

    for k in range(1, n_episodes + 1):
        n_samples = 2 ** (k - 1)
        samples = rng.choice(M, size=n_samples, p=p, replace=True)
        arrivals.append(samples.tolist())
    return arrivals