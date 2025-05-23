import statsmodels.api as sm
import numpy as np
import random
from scipy.stats import uniform_direction
from typing import List

# set seed for reproducibility
random.seed(42)
np.random.seed(42)

def gen_noise(
    n_episodes: int,
    mu_noise: float = 0.05,
    sigma_noise: float = 0.05,
    a_noise: float = 0.02,
    b_noise: float = 0.11,
)-> np.ndarray:
    """
    # TODO: discuss whether clipping is what they mean in the paper
    Generate noise for the time series data.
    Args:
        n_episodes (int): k, the number of episodes. Note that we start from 2
        mu_noise (float): Mean of the noise.
        sigma_noise (float): Standard deviation of the noise.
        a_noise (float): Lower bound of the noise.
        b_noise (float): Upper bound of the noise.
    Returns:
        np.ndarray: Noise for the synthetic data.
    """
    episode_sizes = range(2, n_episodes + 1)
    total_samples = 0
    for episode_size in episode_sizes:
        total_samples += 2**(episode_size - 1)
    noise = np.random.normal(mu_noise, sigma_noise, total_samples)
    noise = np.clip(noise, a_noise, b_noise)
    return noise


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
    assert np.allclose(np.linalg.norm(u, axis=1), 1.0)
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
    M: int = 50, # Number of securities (2, 10, 50 in paper plots)
    d: int = 30, # Dimension of the context (30 in paper plots)
) -> np.ndarray:
    """
    Generate x_t for the synthetic data.
    Args:
        M (int): Number of securities. Default is 50.
        d (int): Dimension of the context. Default is 30.
    Returns:
        np.ndarray: x_t for the synthetic data (simple mv normal).
    """
    x_ts = []
    for i in range(M):
        x_t = np.random.normal(0, 1, d)
        x_ts.append(x_t)
    return np.array(x_ts)

def gen_n_remaining_payments(
    M: int = 50, # Number of securities (2, 10, 50 in paper plots)
) -> np.ndarray:
    """
    # TODO: discuss whether boundaries are inclusive or exclusive
    Generate n_remaining_payments for the synthetic data.
    Args:
        M (int): Number of securities. Default is 50.
    Returns:
        np.ndarray: n_remaining_payments for the synthetic data. Uniform 10, 50
    """
    n_remaining_payments = []
    for i in range(M):
        n_remaining_payment = np.random.randint(10, 50)
        n_remaining_payments.append(n_remaining_payment)
    return np.array(n_remaining_payments)


def gen_coupon_rates(
    M: int = 50, # Number of securities (2, 10, 50 in paper plots)
) -> np.ndarray:
    """
    # TODO: discuss whether boundaries are inclusive or exclusive
    Generate coupon rates for the synthetic data.
    Args:
        M (int): Number of securities. Default is 50.
    Returns:
        np.ndarray: coupon rates for the synthetic data, uniformly distributed.
    """
    coupon_rates = []
    for i in range(M):
        coupon_rate = np.random.uniform(0.02, 0.1)
        coupon_rates.append(coupon_rate)
    return np.array(coupon_rates)


def gen_arrivals(
    M: int = 50,
    mode: str = "uniform",     # "uniform", "poly", or "exp"
    alpha: float = 2.0,        # decay rate (0, 1, 2, 3 in paper plots)
    n_episodes: int = 10       # highest round index k (starts at 2)
) -> List[List[int]]:
    """
    # TODO: what exactly is quadratic decay?
    Draw 2**(k-1) i.i.d. samples from {0, …, M-1} for each round k = 2 … n_episodes.

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

    for k in range(2, n_episodes + 1):
        n_samples = 2 ** (k - 1)
        samples = rng.choice(M, size=n_samples, p=p, replace=True)
        arrivals.append(samples.tolist())
    return arrivals
            
    