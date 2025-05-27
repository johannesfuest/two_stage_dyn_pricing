import statsmodels.api as sm
import numpy as np
import math
import random
from scipy.stats import uniform_direction, truncnorm
from typing import List, Union

# set seed for reproducibility
random.seed(42)
np.random.seed(42)


def sample_w_sphere(d: int, W: float, n_samples: int = 1,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Sample points uniformly from the surface of a d-dimensional sphere.
    Parameters
    ----------
    d : int
        Dimension of the sphere.
    W : float
        Radius of the sphere.
    n_samples : int, default 1
        Number of samples to draw.
    rng : np.random.Generator, optional
        Random number generator to use. If None, a default RNG is used.
    Returns
    -------
    np.ndarray
        An array of shape (n_samples, d) containing the sampled points.
    """
    if rng is None:
        rng = np.random.default_rng()
    z = rng.standard_normal(size=(n_samples, d))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    z = z.reshape((d, n_samples))  # reshape to (d, n_samples)
    return W * z


def gen_noise(
    n_episodes: int,
    mu_noise: float = 0.05,
    sigma_noise: float = 0.05,
    a_noise: float = 0.02,
    b_noise: float = 0.11,
) -> List[float]:
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
    List[float]
        A list of noise values sampled from the truncated normal distribution.
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
        random_state=None,  # add a seed here if you need reproducibility
    )
    return list(noise)


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


def gen_theta_star(
    d: int = 30,
) -> float:
    """
    Generate theta star for the time series data (sampled from the unit sphere).
    Args:
        d (int): Dimension of the context. Default is 30.
    Returns:
        float: theta star
    """
    u = uniform_direction.rvs(dim=d, size=1)
    u = u.reshape((d, 1)) # reshape to column vector
    assert np.allclose(np.linalg.norm(u, axis=0), 1.0)
    return u


def gen_deltas(
    M: int = 50, # Number of securities (2, 10, 50 in paper plots)
    d: int = 30, # Dimension of the context (30 in paper plots)
    delta_max: float = 2.0, # Maximum delta (0.1, 0.5, 2.0 in paper plots)
) -> np.ndarray:
    """
    Generate deltas for the time series data.
    Args:
        M (int): Number of securities. Default is 50.
        d (int): Dimension of the context. Default is 30.
        delta_max (float): Maximum delta. Default is 2.0.
    Returns:
        np.ndarray: delta_js for the synthetic data.
    """
    covariance_matrix = 0.2*np.eye(d) + np.ones(d)@np.ones(d).T
    samples = np.random.multivariate_normal(
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
) -> List[np.ndarray]:
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
        theta_stars.append(theta_star + delta.reshape(-1, 1))
    return theta_stars


def gen_x_ts(
    M: int = 50, # Number of securities (2, 10, 50 in paper plots)
    d: int = 30, # Dimension of the context (30 in paper plots)
) -> List[np.ndarray]:
    """
    Generate x_t for the synthetic data.
    Args:
        M (int): Number of securities. Default is 50.
        d (int): Dimension of the context. Default is 30.
    Returns:
        List[np.ndarray]: x_ts for the synthetic data.
    """
    x_ts = []
    for i in range(M):
        x_t = np.random.normal(0, 1, (d, 1))
        x_ts.append(x_t)
    return x_ts


def gen_n_remaining_payments(
    M: int = 50, # Number of securities (2, 10, 50 in paper plots)
) -> List[List[float]]:
    """
    Generate n_remaining_payments for the synthetic data.
    Args:
        M (int): Number of securities. Default is 50.
    Returns:
        List[List[float]]: n_remaining_payments for the synthetic data.
    """
    n_remaining_payments = []
    for i in range(M):
        n_remaining_payment = np.random.randint(10, 50)
        n_remaining_payment = list(np.linspace(start=0.5, stop=n_remaining_payment * 0.5, num=n_remaining_payment, endpoint=True))
        n_remaining_payments.append(n_remaining_payment)
    return n_remaining_payments


def gen_coupon_rates(
    M: int = 50, # Number of securities (2, 10, 50 in paper plots)
) -> List[float]:
    """
    # TODO: discuss whether boundaries are inclusive or exclusive
    Generate coupon rates for the synthetic data.
    Args:
        M (int): Number of securities. Default is 50.
    Returns:
        List[float]: coupon rates for the synthetic data, uniformly distributed.
    """
    coupon_rates = []
    for i in range(M):
        coupon_rate = np.random.uniform(0.02, 0.1)
        coupon_rates.append(coupon_rate)
    return coupon_rates


def gen_arrivals(
    M: int = 50,
    mode: str = "uniform",     # "uniform", "poly", or "exp"
    alpha: float = 2.0,        # decay rate (0, 1, 2, 3 in paper plots)
    n_episodes: int = 10       # highest round index k (starts at 1)
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
    rng = np.random.default_rng()
    arrivals: List[List[int]] = []

    for k in range(1, n_episodes + 1):
        n_samples = 2 ** (k - 1)
        samples = rng.choice(M, size=n_samples, p=p, replace=True)
        arrivals.append(samples.tolist())
    return arrivals
            
    