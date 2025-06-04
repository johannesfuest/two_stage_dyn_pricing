import numpy as np
import random
import hydra
import torch
from omegaconf import DictConfig
import json
from typing import List, Tuple, Dict, Any, Union
from scipy.optimize import minimize
from scipy.stats import uniform_direction
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import hydra
from omegaconf import OmegaConf
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar
from copy import deepcopy
from typing import Tuple
import numpy as np
from scipy.stats import norm
from scipy.stats import truncnorm

from syn_data_gen import (
    gen_theta_stars,
    gen_x_ts,
    gen_n_remaining_payments,
    gen_coupon_rates,
    gen_deltas,
    gen_noise,
    gen_theta_star,
    gen_arrivals,
    noise_pdf,
    noise_cdf,
)

#random.seed(42)
#np.random.seed(42)



def get_synthetic_data(
        cfg: DictConfig,
) -> Dict[str, Any]:
    """
    Generates a synthetic dataset for simulating dynamic pricing in credit markets,
    following the multi-task learning framework with contextual information.

    Args:
        cfg: Configuration object containing fields such as:
            - k: Number of episodes
            - d: Context dimension
            - M: Number of securities
            - delta_max: Maximum deviation from the shared theta
            - mode: Arrival distribution mode ("uniform", "poly", "exp")
            - alpha: Parameter for decay in non-uniform arrival distributions
            - seed: Random seed for reproducibility

    Returns:
        A dictionary containing the following synthetic components:
            - theta_stars: Array of per-security parameter vectors (θ_j*)
            - theta_star: Shared base parameter vector (θ*)
            - x_ts: Context vectors for each round
            - remaining_payments: Number of future payments per security
            - coupon_rates: Coupon rates for each security
            - arrivals: List of security indices arriving in each episode
            - noise: Truncated normal noise for observed yields
            - deltas: Per-security deviations from the shared θ*
    """
    # Generate noise vector using truncated normal distribution
    noise = gen_noise(cfg.k, seed=cfg.seed)

    # Generate shared latent parameter θ* on unit sphere
    theta_star = gen_theta_star(cfg.d, seed=cfg.seed)

    # Generate per-security deviations δ_j, scaled to delta_max
    deltas = gen_deltas(cfg.M, cfg.d, delta_max=cfg.delta_max, seed=cfg.seed)

    # Combine θ* and δ_j to obtain θ_j* for each security
    theta_stars = gen_theta_stars(theta_star, deltas)

    # Total number of rounds across k episodes
    T = sum(2 ** (k - 1) for k in range(1, cfg.k + 1))

    # Generate i.i.d. context vectors x_t for each round
    x_ts = gen_x_ts(T, cfg.d, seed=cfg.seed)

    # Generate number of remaining payments for each security
    remaining_payments = gen_n_remaining_payments(cfg.M, seed=cfg.seed)

    # Generate coupon rates for each security
    coupon_rates = gen_coupon_rates(cfg.M, seed=cfg.seed)

    # Simulate security arrivals per episode using specified distribution mode
    arrivals = gen_arrivals(cfg.M, cfg.mode, cfg.alpha, cfg.k, seed=cfg.seed)

    return {
        "theta_stars": theta_stars,
        "theta_star": theta_star,
        "x_ts": x_ts,
        "remaining_payments": remaining_payments,
        "coupon_rates": coupon_rates,
        "arrivals": arrivals,
        "noise": noise,
        "deltas": deltas,
    }


def ll_func_vectorized(
    y_us: np.ndarray,
    y_them: np.ndarray,
    X: np.ndarray,
    theta: np.ndarray,
    noise_mu: float = 0.05,
    noise_sigma: float = 0.05,
    noise_support: Tuple[float, float] = (0.02, 0.11),
    eps: float = 1e-9,
) -> float:
    """
    Computes the average log-likelihood for a batch of observations under a truncated
    normal noise model. This function accounts for the one-sided feedback structure
    common in over-the-counter credit markets: if a quote wins, the best competitor's
    yield is observed; otherwise, it's censored.

    Args:
        y_us: A 1D numpy array of our yields (quotes), shape (n_samples,)
        y_them: A 1D numpy array of competitor yields (best competing quotes), same shape
        X: A 2D numpy array of contextual features, shape (n_samples, d)
        theta: A 1D numpy array of model parameters, shape (d,)
        noise_mu: Mean of the truncated normal noise
        noise_sigma: Standard deviation of the noise
        noise_support: Tuple defining the support (a, b) for the truncated normal noise
        eps: A small constant to avoid log(0)

    Returns:
        Average log-likelihood across all samples.
    """
    # Predicted means from the model: mu = Xθ
    mu = X @ theta

    # Boolean mask: True if our quote wins (lower yield is better)
    we_won = y_us >= y_them

    # Observed yield: if we win, we observe y_them; otherwise, we only know y_us
    y_observed = np.where(we_won, y_them, y_us)

    # Standardize observed values using the truncated normal parameters
    a_std, b_std = [(x - noise_mu) / noise_sigma for x in noise_support]
    z_std = (y_observed - mu - noise_mu) / noise_sigma

    # Standard normal PDF and CDF for standardized values
    pdf_vals = norm.pdf(z_std)
    cdf_vals = norm.cdf(z_std)
    Z = norm.cdf(b_std) - norm.cdf(a_std)  # Truncation normalization constant

    # Log-likelihood computation:
    # - if we won: log truncated PDF
    # - if we lost: log (1 - truncated CDF)
    ll_values = np.where(
        we_won,
        np.log(pdf_vals + eps) - np.log(Z + eps),
        np.log(1.0 - cdf_vals + eps) - np.log(Z + eps)
    )

    return ll_values.mean()

def rescale_yield_truncnorm(y, a=0.01, b=8):
    """
    Rescales an input yield value using a sigmoid transformation to a bounded interval [a, b].

    This is typically used to map unbounded predicted values (e.g., from a linear model)
    into a realistic yield range, ensuring outputs stay within plausible financial limits.

    Args:
        y: Raw yield input (can be any real number, often model output).
        a: Lower bound of the target yield range (default is 0.01).
        b: Upper bound of the target yield range (default is 8).

    Returns:
        A value between [a, b], representing a rescaled yield.
    """
    # Apply sigmoid function to squash y into (0, 1), then scale to [a, b]
    return a + (b - a) * (1 / (1 + np.exp(-y)))

def yield_to_price(
        P_t: float,
        coupon_rate: float,
        y: float,
        future_payment_dates: List[float],
) -> float:
    """
    Computes the present value (price) of a bond given its yield, coupon rate, and future payment schedule.

    Args:
        P_t: Face value (par value) of the bond.
        coupon_rate: Annual coupon rate as a decimal (e.g., 0.05 for 5%).
        y: Yield value, typically predicted by a model, passed in unbounded form.
        future_payment_dates: A list of time points (in years) for all future payments.

    Returns:
        The computed bond price as a float.
    """
    value = 0

    # Rescale predicted yield into a valid range using a logistic transformation
    # to keep it within the truncation interval [0.02, 0.11]

    y = rescale_yield_truncnorm(y)

    # Discount all coupon payments (except the last one)
    for t in future_payment_dates[:-1]:
        value += P_t * coupon_rate / (1 + y ) ** t

    # Discount the final payment (coupon + principal)
    value += P_t / (1 + y ) ** future_payment_dates[-1]

    return value


def project_l2_ball_numpy(x: Union[np.ndarray, list, tuple], W: float) -> np.ndarray:
    """
    Projects a vector x onto the L2 ball of radius W, i.e.,
    returns the closest point to x such that its L2 norm is less than or equal to W.

    Args:
        x: Input vector (can be a NumPy array, list, or tuple).
        W: Radius of the L2 ball. Must be non-negative.

    Returns:
        A NumPy array with the same shape as x, scaled if needed to lie within the L2 ball.
    """
    # Convert input to NumPy array of type float
    x = np.asarray(x, dtype=float)

    # Validate radius
    if W < 0:
        raise ValueError("W must be non-negative")

    # Compute L2 norm of the vector (flattened)
    norm_x = np.linalg.norm(x.ravel(), ord=2)

    # If within the L2 ball or zero vector, return as-is
    if norm_x <= W or norm_x == 0.0:
        return x

    # Scale the vector to have norm exactly W
    return (W / norm_x) * x


def reward(
        y_t_us: float,
        y_t_them: float,
        coupon_rate: float,
        future_payment_dates: List[float],
        P_t: float = 1,
        gamma: float = 0,
) -> float:
    """
    Computes the monetary reward for quoting a yield in a competitive bond market setting.
    The reward is the revenue (price received) minus a penalty gamma, but only if our quote wins.

    Args:
        y_t_us: The yield we quote (our offer to the client).
        y_t_them: The yield quoted by the best competitor.
        coupon_rate: Annual coupon rate of the bond.
        future_payment_dates: List of future payment times (in years).
        P_t: Face value (par) of the bond. Defaults to 1.
        gamma: Aggressiveness parameter. A higher gamma penalizes overly cheap quotes.

    Returns:
        The reward value:
        - If our quote wins (i.e., our price is lower), reward = price_us - gamma.
        - Otherwise, reward = 0.
    """
    # Convert our quoted yield to bond price
    price_us = yield_to_price(P_t, coupon_rate, y_t_us, future_payment_dates)

    # Convert competitor's yield to bond price
    price_them = yield_to_price(P_t, coupon_rate, y_t_them, future_payment_dates)

    # If we offer a better (lower) price, we win and receive reward minus aggressiveness penalty
    if price_us <= price_them:
        gain = price_us - gamma
    else:
        # We lose the quote; no reward
        gain = 0

    return gain


def regret(
        y_t_opt: float,
        y_t_us: float,
        y_t_them: float,
        coupon_rate: float,
        future_payment_dates: List[float],
        P_t: float = 100,
) -> float:
    """
    Computes the regret of using our quoted yield compared to the optimal yield.
    Regret is defined as the difference in reward between the optimal policy and our actual decision.

    Args:
        y_t_opt: The yield that would have maximized expected reward (optimal quote).
        y_t_us: The yield we actually quoted.
        y_t_them: The yield quoted by the best competitor.
        coupon_rate: Annual coupon rate of the bond.
        future_payment_dates: List of future payment times (in years).
        P_t: Face value (par) of the bond. Default is 100.

    Returns:
        The regret value (non-negative): how much less reward we earned by not quoting optimally.
    """
    # Compute reward under the optimal quote
    reward_opt = reward(y_t_opt, y_t_them, coupon_rate, future_payment_dates, P_t)

    # Compute reward under our actual quote
    reward_us = reward(y_t_us, y_t_them, coupon_rate, future_payment_dates, P_t)

    # Regret is the loss in reward due to quoting sub-optimally
    return reward_opt - reward_us

def find_optimal_yield_with_cdf(
    theta_star_j: np.ndarray,
    x_t: np.ndarray,
    coupon_rate: float,
    payment_times: list,
    gamma: float = 0.0,
    noise_mu: float = 0.05,
    noise_sigma: float = 0.05,
    noise_support: Tuple[float, float] = (0.02, 0.11),
) -> float:
    """
    Finds the optimal yield (quote) that maximizes the expected reward in a dynamic pricing setting,
    accounting for truncated Gaussian noise in the competitor's yield.

    Args:
        theta_star_j: Parameter vector for security j (shape: [d,]).
        x_t: Context vector at time t (shape: [d,]).
        coupon_rate: Coupon rate of the bond.
        payment_times: List of future payment dates (in years).
        gamma: Aggressiveness penalty for quoting low prices (default: 0.0).
        noise_mu: Mean of the truncated normal noise.
        noise_sigma: Standard deviation of the noise.
        noise_support: Tuple representing the truncation range of the noise (min, max).

    Returns:
        Optimal yield (float) that maximizes expected reward.
        Falls back to predicted mean (mu) if optimization fails.
    """
    # Predicted mean of best competitor yield
    mu = np.dot(theta_star_j, x_t)

    # Standardize bounds of truncated normal
    a_std, b_std = [(x - noise_mu) / noise_sigma for x in noise_support]
    Z = norm.cdf(b_std) - norm.cdf(a_std)  # Normalization constant for truncated distribution

    def objective(y: float) -> float:
        # Convert yield to bond price
        price = yield_to_price(100.0, coupon_rate, y, payment_times)

        # Standardize yield under the noise model
        z_std = (y - mu - noise_mu) / noise_sigma

        # Compute win probability (truncated CDF tail)
        prob_win = (norm.cdf(z_std) - norm.cdf(a_std)) / (Z + 1e-9)

        # Negative expected reward (for minimization)
        return -((price - gamma) * prob_win)

    # Perform bounded scalar minimization to find optimal yield
    res = minimize_scalar(objective, bounds=(-3, 3), method="bounded")

    return res.x if res.success else mu


def run_tsmt_algo_synth(cfg: DictConfig, mode: str = "multi") -> Dict[str, Any]:
    """
    Runs the TSMT algorithm on synthetic data in one of three modes:
    - "multi": two-stage multi-task learning
    - "pooling": pooled estimation across all tasks
    - "individual": task-specific estimation

    Args:
        cfg: Configuration dictionary (OmegaConf) containing simulation parameters.
        mode: Learning mode ("multi", "pooling", or "individual").

    Returns:
        Dictionary containing step-wise regrets, cumulative regret, and cumulative regret per time step.
    """
    # Set synthetic problem parameters
    # cfg.delta_max = 0.1
    # cfg.W = 1 + cfg.delta_max
    # cfg.d = 30
    # cfg.M = 2
    # cfg.k = 12
    #
    # Generate synthetic dataset
    data_dict = get_synthetic_data(cfg)
    T = sum(2 ** (k - 1) for k in range(1, cfg.k))  # total rounds

    # Initialize histories for tracking
    x_history_all = []
    y_us_history_all = []
    y_them_history_all = []

    # Temporary buffers for previous episode
    prev_x = []
    prev_y_us = []
    prev_y_them = []
    prev_arrivals = []

    # For individual mode: maintain task-specific history
    per_task_history = {j: {"x": [], "y_us": [], "y_them": []} for j in range(cfg.M)}

    # Initial theta estimate (normalized and projected to L2 ball)
    theta_init_guess = np.random.randn(cfg.d)
    theta_init_guess /= np.linalg.norm(theta_init_guess)
    theta_init_guess = project_l2_ball_numpy(theta_init_guess, cfg.W)
    theta_star_true = data_dict["theta_star"]

    # Initialize estimates depending on mode
    if mode == "individual":
        theta_est_mode = [theta_init_guess.copy() for _ in range(cfg.M)]
    else:
        theta_bar = theta_init_guess.copy()
        theta_est_mode = [theta_bar.copy() for _ in range(cfg.M)]

    step_regrets = {mode: []}
    global_idx = 0

    for episode in range(1, cfg.k):
        episode_size = 2 ** (episode - 1)
        episode_arrivals = data_dict["arrivals"][episode - 1]
        episode_noises = data_dict["noise"][global_idx:global_idx + episode_size]
        x_episode = data_dict["x_ts"][global_idx:global_idx + episode_size]

        # === Estimation Phase ===
        if episode > 1:
            if mode == "multi":
                # Stage I: pooled MLE to estimate common theta_bar
                theta_bar = minimize(
                    lambda theta: -ll_func_vectorized(
                        np.array(prev_y_us),
                        np.array(prev_y_them),
                        np.array(prev_x),
                        theta
                    ),
                    theta_bar
                ).x
                theta_bar = project_l2_ball_numpy(theta_bar, cfg.W)

                # Stage II: task-specific refinement via regularized MLE
                for j in range(cfg.M):
                    indices = [i for i, z in enumerate(prev_arrivals) if z == j]
                    if not indices:
                        theta_est_mode[j] = theta_bar.copy()
                        continue
                    x_ts = [prev_x[i] for i in indices]
                    y_us = [prev_y_us[i] for i in indices]
                    y_them = [prev_y_them[i] for i in indices]

                    u_F = 1  # Lipschitz upper bound
                    x_max = max(np.linalg.norm(x) for x in x_ts) if x_ts else 1.0
                    n_j = len(indices)

                    # Regularization coefficient
                    lambda_j_k = np.sqrt(
                        8 * (u_F ** 2) * (x_max ** 2) * cfg.d * np.log(2 * cfg.d ** 2 * cfg.M) / n_j
                    )
                    lambda_j_k = 0.0001 * np.sqrt(cfg.d / n_j)

                    theta_j = minimize(
                        lambda theta: -ll_func_vectorized(
                            np.array(y_us),
                            np.array(y_them),
                            np.array(x_ts),
                            theta
                        ) + lambda_j_k * np.linalg.norm(theta - theta_bar),
                        theta_init_guess
                    ).x
                    theta_est_mode[j] = theta_j

            elif mode == "pooling":
                # Pooled MLE used for all tasks
                theta_bar = minimize(
                    lambda theta: -ll_func_vectorized(
                        np.array(prev_y_us),
                        np.array(prev_y_them),
                        np.array(prev_x),
                        theta
                    ),
                    theta_bar
                ).x
                theta_est_mode = [theta_bar.copy() for _ in range(cfg.M)]

            elif mode == "individual":
                # Per-task MLE on previous episode's data
                for j in range(cfg.M):
                    indices = [i for i, z in enumerate(prev_arrivals) if z == j]
                    if not indices:
                        continue
                    x_ts = [prev_x[i] for i in indices]
                    y_us = [prev_y_us[i] for i in indices]
                    y_them = [prev_y_them[i] for i in indices]

                    theta_j = minimize(
                        lambda theta: -ll_func_vectorized(
                            np.array(y_us),
                            np.array(y_them),
                            np.array(x_ts),
                            theta
                        ),
                        theta_est_mode[j]
                    ).x
                    theta_est_mode[j] = theta_j

        # Reset previous episode buffers
        prev_x = []
        prev_y_us = []
        prev_y_them = []
        prev_arrivals = []

        # === Quoting and Regret Calculation ===
        for i in range(episode_size):
            t = global_idx + i
            x_t = x_episode[i]
            z_t = episode_arrivals[i]
            noise = episode_noises[i]

            # Project estimate to feasible set
            theta_proj = project_l2_ball_numpy(theta_est_mode[z_t], cfg.W)

            # Competitor quote and our quote using estimated parameters
            quote_them = data_dict["theta_stars"][z_t] @ x_t + noise
            coupon_rate = data_dict["coupon_rates"][z_t]
            payment_times = np.arange(1, data_dict["remaining_payments"][z_t] + 1) * 0.5

            quote_us = find_optimal_yield_with_cdf(theta_proj, x_t, coupon_rate, payment_times)
            y_opt = find_optimal_yield_with_cdf(data_dict["theta_stars"][z_t], x_t, coupon_rate, payment_times)

            # Evaluate price and reward for both quotes
            price_us = yield_to_price(100, coupon_rate, quote_us, payment_times)
            price_them = yield_to_price(100, coupon_rate, quote_them, payment_times)

            reward_us = price_us if price_us <= price_them else 0
            reward_opt = yield_to_price(100, coupon_rate, y_opt, payment_times)
            reward_opt = reward_opt if reward_opt <= price_them else 0

            # Compute and store regret
            regret_val = reward_opt - reward_us
            step_regrets[mode].append(regret_val)

            # Update histories
            x_history_all.append(x_t)
            y_them_history_all.append(quote_them)
            y_us_history_all.append(quote_us)

            prev_x.append(x_t)
            prev_y_us.append(quote_us)
            prev_y_them.append(quote_them)
            prev_arrivals.append(z_t)

            if mode == "individual":
                per_task_history[z_t]["x"].append(x_t)
                per_task_history[z_t]["y_us"].append(quote_us)
                per_task_history[z_t]["y_them"].append(quote_them)

        global_idx += episode_size
        print(f"[t = {global_idx}] Cumulative Regret: {np.sum(step_regrets[mode]):.4f}")

    return {
        "step_regrets": step_regrets,
        "cumulative_regret": {mode: float(np.sum(step_regrets[mode]))},
        "cumulative_regret_per_t": {mode: list(np.cumsum(step_regrets[mode]))}
    }


def run_all_strategies(cfg, seeds: List[int] = [0, 1], n_jobs: int = -1) -> Dict[str, List[List[float]]]:
    """
    Runs each strategy multiple times (based on seeds) and collects cumulative regrets.

    Parameters:
        cfg (DictConfig): Configuration object.
        seeds (List[int]): List of seeds for repeat runs.
        n_jobs (int): Number of parallel jobs. Use -1 for all available cores.

    Returns:
        Dict[str, List[List[float]]]: regrets per strategy over multiple runs
    """

    strategies = ['multi', 'individual', 'pooling']

    def run_single(seed, strategy):
        local_cfg = deepcopy(cfg)
        local_cfg.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        print(f"→ Running strategy={strategy} with seed={seed}")
        out = run_tsmt_algo_synth(local_cfg, mode=strategy)
        return strategy, out["cumulative_regret_per_t"][strategy]

    # Dispatch in parallel
    results_raw = Parallel(n_jobs=n_jobs)(
        delayed(run_single)(seed, strategy)
        for seed in seeds
        for strategy in strategies
    )

    # Organize results by strategy
    all_results = {strategy: [] for strategy in strategies}
    for strategy, regret_series in results_raw:
        all_results[strategy].append(regret_series)

    return all_results


def plot_cumulative_regret(results: Dict[str, List[List[float]]], cfg):
    """
    Plot the mean and std of cumulative regrets across multiple runs.
    """
    plt.figure(figsize=(8, 5))
    for strategy, all_runs in results.items():
        all_runs = np.array(all_runs)
        mean_regret = np.mean(all_runs, axis=0)
        std_regret = np.std(all_runs, axis=0)

        label = strategy.capitalize()
        color = {'multi': 'blue', 'individual': 'red', 'pooling': 'green'}[strategy]
        T = mean_regret.shape[0]
        plt.plot(mean_regret, label=label, color=color)
        plt.fill_between(range(T), mean_regret - std_regret, mean_regret + std_regret, color=color, alpha=0.2)

    plt.xlabel("Time (T)")
    plt.ylabel("Cumulative Regret")
    plt.title(f"$M = {cfg.M}, \delta_{{\\max}} = {cfg.delta_max}$")
    plt.xticks(np.arange(0, T + 1, 250))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accumulated_regret_plot.png", dpi=300)
    plt.show()

def generate_seed_list(n=2):
    return [random.randint(0, 9999) for _ in range(n)]

@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    seeds = generate_seed_list(50)
    results = run_all_strategies(cfg, seeds=seeds, n_jobs=1)
    plot_cumulative_regret(results, cfg)

if __name__ == "__main__":
    # Load the configuration
    results = main()