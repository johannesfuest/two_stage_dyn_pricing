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

random.seed(42)
np.random.seed(42)


def get_synthetic_data(
        cfg: DictConfig,
) -> Dict[str, Any]:
    """
    Generate synthetic data for the algorithm using the provided configuration.
    Args:
        cfg (DictConfig): Configuration object containing parameters for data generation.
    Returns:
        Dict[str, Any]: Dictionary containing the generated synthetic data.
    """
    noise = gen_noise(cfg.k)
    theta_star = gen_theta_star(cfg.d)
    deltas = gen_deltas(cfg.M, cfg.d, delta_max=cfg.delta_max)
    theta_stars = gen_theta_stars(theta_star, deltas)
    T = sum(2 ** (k - 1) for k in range(1, cfg.k + 1))
    x_ts = gen_x_ts(T, cfg.d)
    remaining_payments = gen_n_remaining_payments(cfg.M)
    coupon_rates = gen_coupon_rates(cfg.M)
    arrivals = gen_arrivals(cfg.M, cfg.mode, cfg.alpha, cfg.k)
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


def ll_func(
        y_t_us: float,
        y_t_them: float,
        x_t: np.ndarray,
        theta: np.ndarray,
        a_noise: float = 0.02,
        b_noise: float = 0.11,
) -> float:
    """
    Log-likelihood function with safe handling of out-of-support values.
    If value is out of truncation bounds but win/loss is correct, reward with large positive value.
    """
    mu = (theta.T @ x_t)
    we_won = y_t_us > y_t_them
    z = y_t_us - mu

    if we_won:
        return np.log(noise_pdf(z))

    else:
        return np.log(1 - noise_cdf(z))


def ll_func_sum(
        yts_us: np.ndarray,
        yts_them: np.ndarray,
        x_ts: np.ndarray,
        theta: np.ndarray,
) -> float:
    """
    Calculates the log-likelihood function for a batch of observations.

    Args:
        yts_us (np.ndarray): The yields we quote.
        yts_them (np.ndarray): The best competitor's yields.
        x_ts (np.ndarray): The context vectors for the securities.
        theta (np.ndarray): The parameter vector.
    Returns:
        float: The log-likelihood value.
    """
    ll = 0
    for i in range(len(yts_us)):
        ll += ll_func(yts_us[i], yts_them[i], x_ts[i], theta)
    ll /= len(yts_us)
    return ll


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
    """Log-likelihood under truncated normal noise model."""
    # Calculate predicted means
    mu = X @ theta

    # Standardize values
    a_std, b_std = [(x - noise_mu) / noise_sigma for x in noise_support]
    z_std = (y_us - mu - noise_mu) / noise_sigma

    # Compute probabilities
    pdf_vals = norm.pdf(z_std)
    cdf_vals = norm.cdf(z_std)
    Z = norm.cdf(b_std) - norm.cdf(a_std)  # truncation normalizer

    # Likelihood terms
    we_won = y_us >= y_them
    ll_values = np.where(
        we_won,
        np.log(pdf_vals + eps) - np.log(Z + eps),  # Density term when we observe yield
        np.log(1.0 - cdf_vals + eps) - np.log(Z + eps)  # Survival term when censored
    )

    return ll_values.mean()

def yield_to_price_torch(
        P: Union[float, torch.Tensor],
        c: Union[float, torch.Tensor],
        payment_times: List[float],
        y: Union[float, torch.Tensor],
) -> torch.Tensor:
    """
    Bond price given yield y (annual compounding).

    Parameters
    ----------
    P : float or tensor
        Face / par value.
    c : float or tensor
        Coupon rate (annual fraction of par).
    payment_times : list of floats
        Future payment times in years (ascending).
    y : float or tensor
        Yield to maturity (discount rate).

    Returns
    -------
    torch.Tensor
        Present value (price).
    """
    # convert everything to tensors on the same device / dtype
    P = torch.as_tensor(P, dtype=torch.float64, device='cpu')
    c = torch.as_tensor(c, dtype=torch.float64, device=P.device)
    y = torch.as_tensor(y, dtype=torch.float64, device=P.device)
    t = torch.as_tensor(payment_times, dtype=torch.float64, device=P.device)

    # coupon leg (exclude final principal repayment)
    coupons = (P * c) / (1.0 + y).pow(t[:-1])
    # principal leg
    principal = P / (1.0 + y).pow(t[-1])

    return coupons.sum() + principal


def yield_to_price(
        P_t: float,
        coupon_rate: float,
        y: float,
        future_payment_dates: List[float],

) -> float:
    """
    Calculate the yield to price for a given set of parameters.

    Args:
        P_t (float): The face/par value of the bond
        coupon_rate (float): The coupon rate of the security.
        future_payment_dates (List[float]): The future payment dates for the security in years from now.
    Returns:
        float: The yield to price value.
    """
    value = 0
    for t in future_payment_dates[:-1]:
        value += P_t * coupon_rate / (1 + y) ** t
    value += P_t / (1 + y) ** future_payment_dates[-1]
    return value


def project_l2_ball_numpy(x: Union[np.ndarray, list, tuple], W: float) -> np.ndarray:
    """
    Project x onto the L2 ball { y : ||y||_2 <= W }.

    Parameters
    ----------
    x : array_like
        Input vector (any shape is fine; flattened norm is used).
    W : float
        Radius of the L2 ball (must be >= 0).

    Returns
    -------
    np.ndarray
        Projected vector with the same shape as x.
    """
    x = np.asarray(x, dtype=float)
    if W < 0:
        raise ValueError("W must be non-negative")

    norm_x = np.linalg.norm(x.ravel(), ord=2)
    if norm_x <= W or norm_x == 0.0:
        return x
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
    Calculate the reward based on the yields and the discount factor.

    Args:
        y_t_us (float): The yield we quote.
        y_t_them (float): The best competitor's yield.
        gamma (float): The aggressiveness parameter. Default is 0.
    Returns:
        float: The reward value.
    """
    price_us = yield_to_price(P_t, coupon_rate, y_t_us, future_payment_dates)
    price_them = yield_to_price(P_t, coupon_rate, y_t_them, future_payment_dates)

    if price_us <= price_them:
        gain = price_us - gamma
    else:
        gain = 0

    return gain


def regret(
        y_t_opt: float,
        y_t_us: float,
        y_t_them: float,
        coupon_rate: float,
        future_payment_dates: List[float],
        P_t: float = 1,
) -> float:
    """
    Calculate the regret based on the optimal price and the quoted price.

    Args:
        y_t_opt (float): The optimal yield.
        y_t_us (float): The quoted yield.
        y_t_them (float): The best competitor's yield.
    Returns:
        float: The regret value.
    """
    return reward(y_t_opt, y_t_them, coupon_rate, future_payment_dates, P_t) - \
        reward(y_t_us, y_t_them, coupon_rate, future_payment_dates, P_t)

from scipy.optimize import minimize_scalar
def find_optimal_yield_with_cdf(theta_star_j, x_t, coupon_rate, payment_times, gamma=0.0):
    mu = np.dot(theta_star_j, x_t)

    def objective(y):
        price = yield_to_price(1.0, coupon_rate, y, payment_times)
        prob_win = noise_cdf(y - mu)
        return -((price - gamma) * prob_win)  # maximize expected reward

    res = minimize_scalar(objective, bounds=(0.01, 0.12), method="bounded")
    return res.x if res.success else mu  # fallback to mean competitor yield

def sample_halfspace(theta: np.ndarray, rng=np.random.default_rng()) -> np.ndarray:
    """
    Draw x ~ N(0, I_d) conditioned on 〈theta, x〉 ≥ 0
    (rejection sampling; acceptance rate = 50 %).
    """
    while True:
        x = rng.standard_normal(theta.size)
        if x @ theta >= 0.0:          # inner product is non-negative
            return x


def run_tsmt_algo_synth_old(cfg: DictConfig, mode: str = "multi") -> Dict[str, Any]:
    cfg.delta_max = 0.1
    cfg.W = 1 + 0.1
    cfg.d = 30
    cfg.M = 2
    data_dict = get_synthetic_data(cfg)

    T = sum(2 ** (k - 1) for k in range(1, cfg.k + 1))

    # Cumulative history for pooled estimation
    x_history = []
    y_us_history = []
    y_them_history = []

    first_episode_arrivals = data_dict["arrivals"][0]
    episode_size = len(first_episode_arrivals)
    x_ts_first = data_dict["x_ts"][:episode_size]

    # Initial estimate
    theta_init_guess = np.random.randn(cfg.d)
    theta_init_guess /= np.linalg.norm(theta_init_guess)

    for i in range(episode_size):
        x_t = x_ts_first[i]
        arrival = first_episode_arrivals[i]
        quote_them = data_dict["theta_stars"][arrival] @ x_t + data_dict["noise"][i]
        coupon_rate = data_dict["coupon_rates"][arrival]
        payment_times = np.arange(1, data_dict["remaining_payments"][arrival] + 1) * 0.5
        quote_us = find_optimal_yield_with_cdf(theta_init_guess, x_t, coupon_rate, payment_times)

        x_history.append(x_t)
        y_them_history.append(quote_them)
        if quote_us >= quote_them:
            y_us_history.append(quote_them)
        else:
            y_us_history.append(quote_us)

    # Stage I: pooled MLE
    prev_theta_bar = minimize(
        lambda theta: -ll_func_vectorized(
            np.array(y_us_history),
            np.array(y_them_history),
            np.array(x_history),
            theta,
        ),
        theta_init_guess
    ).x

    theta_est = [prev_theta_bar.copy() for _ in range(cfg.M)]

    global_idx = 0
    prev_arrivals = data_dict["arrivals"][0]
    first_arrival = prev_arrivals[0]
    x_t = data_dict["x_ts"][global_idx]
    theta_star_true = data_dict["theta_star"]

    theta_proj = project_l2_ball_numpy(prev_theta_bar, cfg.W)
    quote_them = data_dict["theta_stars"][first_arrival] @ x_t + data_dict["noise"][global_idx]
    coupon_rate = data_dict["coupon_rates"][first_arrival]
    payment_times = np.arange(1, data_dict["remaining_payments"][first_arrival] + 1) * 0.5
    quote_us = find_optimal_yield_with_cdf(theta_proj, x_t, coupon_rate, payment_times)
    y_opt = find_optimal_yield_with_cdf(data_dict["theta_stars"][first_arrival], x_t, coupon_rate, payment_times)

    step_regrets = {mode: [regret(y_opt, quote_us, quote_them, coupon_rate, payment_times)]}
    global_idx += 1

    for episode in range(2, cfg.k):
        episode_size = 2 ** (episode - 1)
        episode_arrivals = data_dict["arrivals"][episode - 1]
        episode_noises = data_dict["noise"][global_idx:global_idx + episode_size]

        # Stage I: pooled theta_bar using full history
        theta_bar_pool = minimize(
            lambda theta: -ll_func_vectorized(
                np.array(y_us_history),
                np.array(y_them_history),
                np.array(x_history),
                theta
            ),
            prev_theta_bar
        ).x

        theta_bar_pool = project_l2_ball_numpy(theta_bar_pool, cfg.W)
        error = np.linalg.norm(theta_bar_pool - theta_star_true)
        print(f"Estimation error ―theta_bar_pool − theta_star―: {error:.4f}")

        if mode == "pool":
            theta_est_mode = [theta_bar_pool.copy() for _ in range(cfg.M)]
        else:
            theta_est_multi = [theta_bar_pool.copy() for _ in range(cfg.M)]
            for security in range(cfg.M):
                indices = [i for i, z in enumerate(prev_arrivals) if z == security]
                if not indices:
                    continue
                x_ts = [x_history[i] for i in indices]
                y_ts_us = [y_us_history[i] for i in indices]
                y_ts_them = [y_them_history[i] for i in indices]
                lambda_j_k = 0.1 * np.sqrt(cfg.d / len(indices))

                theta_j_est = minimize(
                    lambda theta: -ll_func_vectorized(
                        np.array(y_ts_us),
                        np.array(y_ts_them),
                        np.array(x_ts),
                        theta,
                    ) + (lambda_j_k * np.linalg.norm(theta - theta_bar_pool)**2 if mode == "multi" else 0),
                    theta_est[security]
                ).x

                theta_est_multi[security] = theta_j_est

            theta_est_mode = theta_est_multi

        prev_arrivals = episode_arrivals

        for i in range(episode_size):
            t_idx = global_idx + i
            arrival = episode_arrivals[i]
            x_t = data_dict["x_ts"][t_idx]

            theta_proj = project_l2_ball_numpy(theta_est_mode[arrival], cfg.W)
            quote_them = data_dict["theta_stars"][arrival] @ x_t + episode_noises[i]
            coupon_rate = data_dict["coupon_rates"][arrival]
            payment_times = np.arange(1, data_dict["remaining_payments"][arrival] + 1) * 0.5
            quote_us = find_optimal_yield_with_cdf(theta_proj, x_t, coupon_rate, payment_times)
            y_opt = find_optimal_yield_with_cdf(data_dict["theta_stars"][arrival], x_t, coupon_rate, payment_times)

            x_history.append(x_t)
            y_them_history.append(quote_them)
            if quote_us >= quote_them:
                y_us_history.append(quote_them)
            else:
                y_us_history.append(quote_us)

            r = regret(y_opt, quote_us, quote_them, coupon_rate, payment_times)
            step_regrets[mode].append(r)

        prev_theta_bar = theta_bar_pool
        theta_est = theta_est_mode
        global_idx += episode_size

        print(len(step_regrets[mode]))
        print({mode: float(np.sum(step_regrets[mode])) for mode in step_regrets})

    return {
        "step_regrets": step_regrets,
        "cumulative_regret": {mode: float(np.sum(step_regrets[mode])) for mode in step_regrets},
        "cumulative_regret_per_t": {mode: list(np.cumsum(step_regrets[mode])) for mode in step_regrets}
    }

def run_tsmt_algo_synth(cfg: DictConfig, mode: str = "multi") -> Dict[str, Any]:
    cfg.delta_max = 2
    cfg.W = 1 + 2
    cfg.d = 50
    cfg.M = 2

    cfg.delta_max = 0.1
    cfg.W = 1 + 0.1
    cfg.d = 30
    cfg.M = 10
    data_dict = get_synthetic_data(cfg)
    cfg.k = 11

    T = sum(2 ** (k - 1) for k in range(1, cfg.k))
    
    # Histories
    x_history_all = []
    y_us_history_all = []
    y_them_history_all = []

    prev_x = []
    prev_y_us = []
    prev_y_them = []
    prev_arrivals = []

    # Per-task data for individual mode
    per_task_history = {j: {"x": [], "y_us": [], "y_them": []} for j in range(cfg.M)}

    # Initialization
    theta_init_guess = np.random.randn(cfg.d)
    theta_init_guess /= np.linalg.norm(theta_init_guess)
    theta_star_true = data_dict["theta_star"]

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
                # Stage I: pooled MLE from previous episode
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

                # Stage II: refine per task
                for j in range(cfg.M):
                    indices = [i for i, z in enumerate(prev_arrivals) if z == j]
                    if not indices:
                        continue
                    x_ts = [prev_x[i] for i in indices]
                    y_us = [prev_y_us[i] for i in indices]
                    y_them = [prev_y_them[i] for i in indices]
                    lambda_j_k = 200 * np.sqrt(cfg.d / len(indices))
                    u_F = 1.0  # Lipschitz constant upper bound
                    x_max = max(np.linalg.norm(x) for x in x_ts) if x_ts else 1.0
                    n_j = len(indices)
                    lambda_j_k = np.sqrt(
                        8 * (u_F ** 2) * (x_max ** 2) * cfg.d * np.log(2 * cfg.d ** 2 * cfg.M) / n_j
                    )
                    #lambda_j_k = 0.1 * np.sqrt(cfg.d)/n_j
                    theta_j = minimize(
                        lambda theta: -ll_func_vectorized(
                            np.array(y_us),
                            np.array(y_them),
                            np.array(x_ts),
                            theta
                        ) + lambda_j_k * np.linalg.norm(theta - theta_bar),
                        theta_est_mode[j]
                    ).x
                    theta_est_mode[j] = theta_j

            elif mode == "pool":
                theta_bar = minimize(
                    lambda theta: -ll_func_vectorized(
                        np.array(y_us_history_all),
                        np.array(y_them_history_all),
                        np.array(x_history_all),
                        theta
                    ),
                    theta_bar
                ).x
                theta_est_mode = [theta_bar.copy() for _ in range(cfg.M)]

            elif mode == "individual":
                for j in range(cfg.M):
                    hist = per_task_history[j]
                    if len(hist["x"]) == 0:
                        continue
                    theta_j = minimize(
                        lambda theta: -ll_func_vectorized(
                            np.array(hist["y_us"]),
                            np.array(hist["y_them"]),
                            np.array(hist["x"]),
                            theta
                        ),
                        theta_est_mode[j]
                    ).x
                    theta_est_mode[j] = theta_j

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

            theta_proj = project_l2_ball_numpy(theta_est_mode[z_t], cfg.W)
            quote_them = data_dict["theta_stars"][z_t] @ x_t + noise
            coupon_rate = data_dict["coupon_rates"][z_t]
            payment_times = np.arange(1, data_dict["remaining_payments"][z_t] + 1) * 0.5
            quote_us = find_optimal_yield_with_cdf(theta_proj, x_t, coupon_rate, payment_times)
            y_opt = find_optimal_yield_with_cdf(data_dict["theta_stars"][z_t], x_t, coupon_rate, payment_times)

            price_us = yield_to_price(1, coupon_rate, quote_us, payment_times)
            price_them = yield_to_price(1, coupon_rate, quote_them, payment_times)
            reward_us = price_us if price_us <= price_them else 0
            reward_opt = yield_to_price(1, coupon_rate, y_opt, payment_times) if yield_to_price(1, coupon_rate, y_opt, payment_times) <= price_them else 0
            regret_val = reward_opt - reward_us
            step_regrets[mode].append(regret_val)

            # Track for history
            x_history_all.append(x_t)
            y_them_history_all.append(quote_them)
            y_us_history_all.append(min(quote_us, quote_them))

            prev_x.append(x_t)
            prev_y_us.append(min(quote_us, quote_them))
            prev_y_them.append(quote_them)
            prev_arrivals.append(z_t)

            if mode == "individual":
                per_task_history[z_t]["x"].append(x_t)
                per_task_history[z_t]["y_us"].append(min(quote_us, quote_them))
                per_task_history[z_t]["y_them"].append(quote_them)

        global_idx += episode_size
        print(f"[t = {global_idx}] Cumulative Regret: {np.sum(step_regrets[mode]):.4f}")

    return {
        "step_regrets": step_regrets,
        "cumulative_regret": {mode: float(np.sum(step_regrets[mode]))},
        "cumulative_regret_per_t": {mode: list(np.cumsum(step_regrets[mode]))}
    }


def run_all_strategies(cfg):
    results = {}
    for strategy in ['multi', 'individual', 'pool']:
        print(f"Running strategy: {strategy}")
        out = run_tsmt_algo_synth(cfg, mode=strategy)
        results[strategy] = out['cumulative_regret_per_t'][strategy]
    return results

def plot_cumulative_regret(results: dict, cfg):
    plt.figure(figsize=(8, 5))
    T = len(next(iter(results.values())))

    for strategy, regrets in results.items():
        regrets = np.array([regrets])  # wrap in a list to simulate a single run
        mean_regret = np.mean(regrets, axis=0)
        std_regret = np.std(regrets, axis=0)

        label = strategy.capitalize()
        color = {'multi': 'blue', 'individual': 'red', 'pool': 'green'}[strategy]
        plt.plot(mean_regret, label=label, color=color)
        plt.fill_between(range(T), mean_regret - std_regret, mean_regret + std_regret, color=color, alpha=0.2)

    plt.xlabel("Time (T)")
    plt.ylabel("Cumulative Regret")
    plt.title(f"$M = {cfg.M}, \delta_{{\max}} = {cfg.delta_max}$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    #return run_tsmt_algo_synth(cfg, mode = 'individual')
    #return run_tsmt_algo_synth(cfg, mode = "individual")
    results = run_all_strategies(cfg)
    plot_cumulative_regret(results, cfg)

if __name__ == "__main__":
    # Load the configuration
    results = main()  # run the TSMT algorithm and get the results
    # Save the results to a file
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)