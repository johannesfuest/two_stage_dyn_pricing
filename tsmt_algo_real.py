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
import pandas as pd

def load_real_data(csv_path: str):
    df = pd.read_csv(csv_path)

    # Context features: principal components
    x_ts = df[["PC1", "PC2", "PC3", "PC4"]].values

    # Our quoted yield: for oracle analysis, we use Ridge later
    y_them = df["weighted_avg_yield"].values

    # Coupon and payments
    coupon_rates = df["Coupon Rate"].values
    n_payments = df["remaining_payments"].fillna(0).astype(int).values
    payment_times = [list(0.5 * (i + 1) for i in range(n)) for n in n_payments]

    return {
        "x_ts": x_ts,
        "y_them": y_them,
        "coupon_rates": coupon_rates,
        "n_payments": n_payments,
        "payment_times": payment_times,
    }
from sklearn.linear_model import Ridge

def compute_oracle_and_regret(df_path: str):
    from sklearn.linear_model import Ridge

    # Load data
    data = load_real_data(df_path)
    X = data["x_ts"]
    y_them = data["y_them"]
    coupon_rates = data["coupon_rates"]
    payment_times = data["payment_times"]

    # Fit oracle theta via Ridge
    theta_star = Ridge(alpha=1e-2, fit_intercept=False).fit(X, y_them).coef_

    # Predict and compute regret
    regrets = []
    y_us = []
    for i, x_t in enumerate(X):
        coupon = coupon_rates[i]
        times = payment_times[i]

        if len(times) == 0:
            continue  # skip bonds with 0 future payments

        # Optimal yield using oracle theta
        y_star = find_optimal_yield_with_cdf(theta_star, x_t, coupon, times)
        y_us.append(y_star)

        # Regret
        r = regret(y_star, y_star, y_them[i], coupon, times)  # Self-regret is zero
        regrets.append(r)

    return {
        "oracle_theta": theta_star,
        "avg_regret": np.mean(regrets),
        "y_us": y_us,
        "regrets": regrets
    }

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
        price = yield_to_price(100, coupon_rate, y, payment_times)
        prob_win = noise_cdf(y - mu)
        return -((price - gamma) * prob_win)  # maximize expected reward

    res = minimize_scalar(objective, bounds=(0.01, 0.15), method="bounded")
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


def run_tsmt_real_data(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # --- Load and preprocess data ---
    df = pd.read_csv("D:/Project Bond Price Illiquidity/final_df.csv")
    df = df[~df["weighted_avg_yield"].isna()]
    df["remaining_payments"] = df["remaining_payments"].fillna(0).astype(int)
    df = df[df["remaining_payments"] > 0].copy()
    df["task_id"] = df["cusip_id"].astype("category").cat.codes

    x_all = df[[ "PC1", "PC2", "PC3", "PC4", "rolling_30d_BID_scaled", "Coupon Rate" ]].values

    y_all = df["weighted_avg_yield"].values
    coupons = df["Coupon Rate"].values
    n_payments = df["remaining_payments"].values
    payment_times = [list(0.5 * (i + 1) for i in range(n)) for n in n_payments]
    task_ids = df["task_id"].values
    tasks = np.unique(task_ids)

    d = x_all.shape[1]
    W = 0.005
    K = 11
    M = 37
    episode_lengths = [2 ** (k - 1) for k in range(1, K + 1)]
    total_required = sum(episode_lengths)
    assert len(x_all) >= total_required, "Not enough data"

    # --- Oracle theta_j* estimates ---
    from sklearn.metrics import r2_score
    theta_star_per_task = {}
    r2_per_task = {}
    for task in tasks:
        mask = task_ids == task
        X_j = x_all[mask]
        y_j = y_all[mask]
        model = Ridge(alpha=0.1, fit_intercept=True)
        y_pred = model.fit(X_j, y_j).predict(X_j)
        theta_j = model.coef_

        if theta_j.shape[0] == d:
            theta_star_per_task[task] = theta_j

        r2 = r2_score(y_j, y_pred)
        r2_per_task[task] = r2

    df["task_id"] = df["cusip_id"].astype("category").cat.codes
    cusip_map = dict(enumerate(df["cusip_id"].astype("category").cat.categories))
    r2_per_cusip = {cusip_map[task]: r2 for task, r2 in r2_per_task.items()}
    sorted_r2 = sorted(r2_per_cusip.items(), key=lambda x: x[1])

    norms = [np.linalg.norm(theta_j) for theta_j in theta_star_per_task.values()]
    W = 1.2 * max(norms)  # Or use np.percentile(norms, 95) for robustness
    # --- Initialize TSMT ---
    x_history, y_us_history, y_them_history, task_history = [], [], [], []
    step_regrets = {"multi": []}
    theta_est_per_task = {task: np.random.randn(d) for task in tasks}
    start_idx = 0
    mode = "multi"

    for episode_idx, ep_len in enumerate(episode_lengths):
        end_idx = start_idx + ep_len
        x_ep = x_all[start_idx:end_idx]
        y_ep = y_all[start_idx:end_idx]
        c_ep = coupons[start_idx:end_idx]
        pt_ep = payment_times[start_idx:end_idx]
        task_ep = task_ids[start_idx:end_idx]

        # --- Stage I: Pooled MLE ---
        if x_history:
            pooled_init = np.mean(np.stack([v for v in theta_est_per_task.values() if v.shape[0] == d]), axis=0)
            theta_bar = minimize(
                lambda theta: -ll_func_vectorized(
                    np.array(y_us_history),
                    np.array(y_them_history),
                    np.array(x_history),
                    theta,
                ),
                pooled_init,
            ).x
        else:
            theta_bar = np.random.randn(d)
            theta_bar /= np.linalg.norm(theta_bar)
        theta_norm = np.linalg.norm(theta_bar)
        print(f"[Episode {episode_idx + 1}] ||theta_bar|| = {theta_norm:.6f}")


        theta_bar = project_l2_ball_numpy(theta_bar, W)

        # --- Stage II: Task-specific refinement ---
        for task in tasks:
            indices = [i for i, t in enumerate(task_history) if t == task]
            if len(indices) < 1:
                continue
            X_j = np.array([x_history[i] for i in indices])
            y_us_j = np.array([y_us_history[i] for i in indices])
            y_them_j = np.array([y_them_history[i] for i in indices])
            lam = 100 * np.sqrt(d / len(indices))
            u_F = 1.0  # Placeholder: replace with actual max gradient
            x_max = np.max([np.linalg.norm(x) for x in x_history])  # Max context norm
            lambda_j_k = np.sqrt(8 * (u_F ** 2) * (x_max ** 2) * d * np.log(2 * d ** 2 * M) / len(indices))
            lam = 0.1 * np.sqrt(d / len(indices))

            theta_init = theta_est_per_task.get(task)
            if theta_init is None or theta_init.shape[0] != d:
                theta_init = np.random.randn(d)

            theta_j = minimize(
                lambda theta: -ll_func_vectorized(y_us_j, y_them_j, X_j, theta)
                + lambda_j_k * np.linalg.norm(theta - theta_bar) ,
                theta_init
            ).x

            theta_est_per_task[task] = project_l2_ball_numpy(theta_j, W)

        # --- Quoting and regret ---
        for i in range(ep_len):
            x_t = x_ep[i]
            y_them_i = y_ep[i]
            coupon = c_ep[i]
            ptimes = pt_ep[i]
            task = task_ep[i]

            theta_proj = theta_est_per_task.get(task, theta_bar)
            quote_us = find_optimal_yield_with_cdf(theta_proj, x_t, coupon, ptimes)
            quote_them = y_them_i

            theta_star_j = theta_star_per_task.get(task, theta_bar)
            y_opt = find_optimal_yield_with_cdf(theta_star_j, x_t, coupon, ptimes)

            # --- Logging consistent with your original structure ---
            x_history.append(x_t)
            y_them_history.append(quote_them)
            if quote_us >= quote_them:
                y_us_history.append(quote_them)
            else:
                y_us_history.append(quote_us)

            r = regret(y_opt, quote_us, quote_them, coupon, ptimes)
            step_regrets[mode].append(r)

        task_history += list(task_ep)
        start_idx = end_idx

        print(len(step_regrets[mode]))
        print({mode: float(np.sum(step_regrets[mode])) for mode in step_regrets})

    return {
        "step_regrets": step_regrets,
        "cumulative_regret": {mode: float(np.sum(step_regrets[mode])) for mode in step_regrets},
        "cumulative_regret_per_t": {mode: list(np.cumsum(step_regrets[mode])) for mode in step_regrets}
    }

def run_tsmt_real_data_pool(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # --- Load and preprocess data ---
    df = pd.read_csv("D:/Project Bond Price Illiquidity/final_df.csv")
    df = df[~df["weighted_avg_yield"].isna()]
    df["remaining_payments"] = df["remaining_payments"].fillna(0).astype(int)
    df = df[df["remaining_payments"] > 0].copy()
    df["task_id"] = df["cusip_id"].astype("category").cat.codes

    x_all = df[["PC1", "PC2", "PC3", "PC4"]].values
    y_all = df["weighted_avg_yield"].values
    coupons = df["Coupon Rate"].values
    n_payments = df["remaining_payments"].values
    payment_times = [list(0.5 * (i + 1) for i in range(n)) for n in n_payments]
    task_ids = df["task_id"].values
    tasks = np.unique(task_ids)

    d = x_all.shape[1]
    K = 11
    M = len(tasks)
    episode_lengths = [2 ** (k - 1) for k in range(1, K + 1)]
    total_required = sum(episode_lengths)
    assert len(x_all) >= total_required, "Not enough data"

    # --- Oracle theta_j* estimates ---
    from sklearn.linear_model import Ridge
    theta_star_per_task = {}
    for task in tasks:
        mask = task_ids == task
        X_j = x_all[mask]
        y_j = y_all[mask]
        model = Ridge(alpha = 0.1, fit_intercept=True)
        theta_j = model.fit(X_j, y_j).coef_
        if theta_j.shape[0] == d:
            theta_star_per_task[task] = theta_j

    W = 1.2 * max(np.linalg.norm(theta) for theta in theta_star_per_task.values())
    W = 1

    # --- Initialize ---
    x_history, y_us_history, y_them_history, task_history = [], [], [], []
    step_regrets = {"pool": []}
    start_idx = 0

    for episode_idx, ep_len in enumerate(episode_lengths):
        end_idx = start_idx + ep_len
        x_ep = x_all[start_idx:end_idx]
        y_ep = y_all[start_idx:end_idx]
        c_ep = coupons[start_idx:end_idx]
        pt_ep = payment_times[start_idx:end_idx]
        task_ep = task_ids[start_idx:end_idx]

        # --- Stage I: Pooled MLE using all history ---
        if x_history:
            pooled_init = np.random.randn(d)
            theta_bar = minimize(
                lambda theta: -ll_func_vectorized(
                    np.array(y_us_history),
                    np.array(y_them_history),
                    np.array(x_history),
                    theta
                ),
                pooled_init
            ).x
        else:
            theta_bar = np.random.randn(d)
            theta_bar /= np.linalg.norm(theta_bar)

        theta_bar = project_l2_ball_numpy(theta_bar, W)
        print(f"[Episode {episode_idx + 1}] ||theta_bar|| = {np.linalg.norm(theta_bar):.6f}")

        # --- Quoting and regret logging ---
        for i in range(ep_len):
            x_t = x_ep[i]
            y_them_i = y_ep[i]
            coupon = c_ep[i]
            ptimes = pt_ep[i]
            task = task_ep[i]

            quote_us = find_optimal_yield_with_cdf(theta_bar, x_t, coupon, ptimes)
            quote_them = y_them_i
            theta_star_j = theta_star_per_task.get(task, theta_bar)
            y_opt = find_optimal_yield_with_cdf(theta_star_j, x_t, coupon, ptimes)

            x_history.append(x_t)
            y_them_history.append(quote_them)
            if quote_us >= quote_them:
                y_us_history.append(quote_them)
            else:
                y_us_history.append(quote_us)
            task_history.append(task)

            r = regret(y_opt, quote_us, quote_them, coupon, ptimes, P_t = 100)
            step_regrets["pool"].append(r)

        start_idx = end_idx
        print(f"Episode {episode_idx + 1}: Total steps = {len(step_regrets['pool'])}, Regret = {np.sum(step_regrets['pool']):.4f}")

    return {
        "step_regrets": step_regrets,
        "cumulative_regret": {"pool": float(np.sum(step_regrets["pool"]))},
        "cumulative_regret_per_t": {"pool": list(np.cumsum(step_regrets["pool"]))}
    }




def run_tsmt_real_data_multi(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # --- Load and preprocess data ---
    df = pd.read_csv("D:/Project Bond Price Illiquidity/final_df.csv")
    df = df[~df["weighted_avg_yield"].isna()]
    df["remaining_payments"] = df["remaining_payments"].fillna(0).astype(int)
    df = df[df["remaining_payments"] > 0].copy()
    df["task_id"] = df["cusip_id"].astype("category").cat.codes

    x_all = df[["PC1", "PC2", "PC3", "PC4"]].values


    y_all = df["weighted_avg_yield"].values
    coupons = df["Coupon Rate"].values
    n_payments = df["remaining_payments"].values
    payment_times = [list(0.5 * (i + 1) for i in range(n)) for n in n_payments]
    task_ids = df["task_id"].values
    tasks = np.unique(task_ids)

    d = x_all.shape[1]
    K = 11
    M = len(tasks)
    episode_lengths = [2 ** (k - 1) for k in range(1, K + 1)]
    total_required = sum(episode_lengths)
    assert len(x_all) >= total_required, "Not enough data"

    # --- Oracle theta_j* estimates ---
    from sklearn.linear_model import Ridge
    theta_star_per_task = {}
    for task in tasks:
        mask = task_ids == task
        X_j = x_all[mask]
        y_j = y_all[mask]
        model = Ridge( alpha = 0.1, fit_intercept=True)
        theta_j = model.fit(X_j, y_j).coef_
        if theta_j.shape[0] == d:
            theta_star_per_task[task] = theta_j

    W = 1.2 * max(np.linalg.norm(theta) for theta in theta_star_per_task.values())
    W = 1

    # --- Initialize ---
    x_history, y_us_history, y_them_history, task_history = [], [], [], []
    theta_est_per_task = {task: np.random.randn(d) for task in tasks}
    step_regrets = {"multi": []}
    start_idx = 0

    for episode_idx, ep_len in enumerate(episode_lengths):
        end_idx = start_idx + ep_len
        x_ep = x_all[start_idx:end_idx]
        y_ep = y_all[start_idx:end_idx]
        c_ep = coupons[start_idx:end_idx]
        pt_ep = payment_times[start_idx:end_idx]
        task_ep = task_ids[start_idx:end_idx]

        # --- Stage I: Pooled MLE using only previous episode data ---
        if episode_idx > 0:
            prev_len = episode_lengths[episode_idx - 1]
            pooled_init = np.mean(np.stack([v for v in theta_est_per_task.values()]), axis=0)
            theta_bar = minimize(
                lambda theta: -ll_func_vectorized(
                    np.array(y_us_history[-prev_len:]),
                    np.array(y_them_history[-prev_len:]),
                    np.array(x_history[-prev_len:]),
                    theta
                ),
                pooled_init
            ).x
        else:
            theta_bar = np.random.randn(d)
            theta_bar /= np.linalg.norm(theta_bar)

        theta_bar = project_l2_ball_numpy(theta_bar, W)
        print(f"[Episode {episode_idx + 1}] ||theta_bar|| = {np.linalg.norm(theta_bar):.6f}")

        # --- Stage II: Task-specific MLE using only previous episode data ---
        if episode_idx > 0:
            prev_start = start_idx - episode_lengths[episode_idx - 1]
            prev_end = start_idx
            for task in tasks:
                indices = [
                    i for i in range(prev_start, prev_end)
                    if task_history[i] == task
                ]
                if not indices:
                    continue

                X_j = np.array([x_history[i] for i in indices])
                y_us_j = np.array([y_us_history[i] for i in indices])
                y_them_j = np.array([y_them_history[i] for i in indices])

                u_F = 1.0
                x_max = np.max([np.linalg.norm(x) for x in x_history[prev_start:prev_end]]) if x_history else 1.0
                lambda_j_k = np.sqrt(8 * (u_F ** 2) * (x_max ** 2) * d * np.log(2 * d ** 2 * M) / len(indices))
                #lambda_j_k = 0.1 * np.sqrt(d/len(indices))

                theta_init = theta_est_per_task[task]
                theta_j = minimize(
                    lambda theta: -ll_func_vectorized(y_us_j, y_them_j, X_j, theta) +
                                  lambda_j_k * np.linalg.norm(theta - theta_bar),
                    theta_bar
                ).x

                theta_est_per_task[task] = project_l2_ball_numpy(theta_j, W)

        # --- Quoting and regret logging ---
        for i in range(ep_len):
            x_t = x_ep[i]
            y_them_i = y_ep[i]
            coupon = c_ep[i]
            ptimes = pt_ep[i]
            task = task_ep[i]
            theta_bar = project_l2_ball_numpy(theta_bar, W)
            theta_proj = theta_est_per_task.get(task, theta_bar)
            quote_us = find_optimal_yield_with_cdf(theta_proj, x_t, coupon, ptimes)
            quote_them = y_them_i
            theta_star_j = theta_star_per_task.get(task, theta_bar)
            y_opt = find_optimal_yield_with_cdf(theta_star_j, x_t, coupon, ptimes)

            x_history.append(x_t)
            y_them_history.append(quote_them)
            if quote_us >= quote_them:
                y_us_history.append(quote_them)
            else:
                y_us_history.append(quote_us)
            task_history.append(task)

            r = regret(y_opt, quote_us, quote_them, coupon, ptimes, P_t = 100)
            step_regrets["multi"].append(r)

        start_idx = end_idx
        print(f"Episode {episode_idx + 1}: Total steps = {len(step_regrets['multi'])}, Regret = {np.sum(step_regrets['multi']):.4f}")

    return {
        "step_regrets": step_regrets,
        "cumulative_regret": {"multi": float(np.sum(step_regrets["multi"]))},
        "cumulative_regret_per_t": {"multi": list(np.cumsum(step_regrets["multi"]))}
    }

def run_tsmt_real_data_individual(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # --- Load and preprocess data ---
    df = pd.read_csv("D:/Project Bond Price Illiquidity/final_df.csv")
    df = df[~df["weighted_avg_yield"].isna()]
    df["remaining_payments"] = df["remaining_payments"].fillna(0).astype(int)
    df = df[df["remaining_payments"] > 0].copy()
    df["task_id"] = df["cusip_id"].astype("category").cat.codes

    x_all = df[["PC1", "PC2", "PC3", "PC4"]].values

    y_all = df["weighted_avg_yield"].values
    coupons = df["Coupon Rate"].values
    n_payments = df["remaining_payments"].values
    payment_times = [list(0.5 * (i + 1) for i in range(n)) for n in n_payments]
    task_ids = df["task_id"].values
    tasks = np.unique(task_ids)

    d = x_all.shape[1]
    K = 11
    M = len(tasks)
    episode_lengths = [2 ** (k - 1) for k in range(1, K + 1)]
    total_required = sum(episode_lengths)
    assert len(x_all) >= total_required, "Not enough data"

    # --- Oracle theta_j* estimates ---
    from sklearn.linear_model import Ridge
    theta_star_per_task = {}
    for task in tasks:
        mask = task_ids == task
        X_j = x_all[mask]
        y_j = y_all[mask]
        model = Ridge(alpha = 0.1, fit_intercept=True)
        theta_j = model.fit(X_j, y_j).coef_
        if theta_j.shape[0] == d:
            theta_star_per_task[task] = theta_j

    W = 1.2 * max(np.linalg.norm(theta) for theta in theta_star_per_task.values())
    W = 1
    #W = 1

    # --- Initialize ---
    x_history, y_us_history, y_them_history, task_history = [], [], [], []
    theta_est_per_task = {task: np.random.randn(d) for task in tasks}
    step_regrets = {"individual": []}
    start_idx = 0

    for episode_idx, ep_len in enumerate(episode_lengths):
        end_idx = start_idx + ep_len
        x_ep = x_all[start_idx:end_idx]
        y_ep = y_all[start_idx:end_idx]
        c_ep = coupons[start_idx:end_idx]
        pt_ep = payment_times[start_idx:end_idx]
        task_ep = task_ids[start_idx:end_idx]

        # --- Individual MLE per task (NO regularization) ---
        if episode_idx > 0:
            prev_start = start_idx - episode_lengths[episode_idx - 1]
            prev_end = start_idx
            for task in tasks:
                indices = [
                    i for i in range(start_idx)
                    if task_history[i] == task
                ]
                if not indices:
                    continue

                X_j = np.array([x_history[i] for i in indices])
                y_us_j = np.array([y_us_history[i] for i in indices])
                y_them_j = np.array([y_them_history[i] for i in indices])

                theta_init = theta_est_per_task[task]

                theta_j = minimize(
                    lambda theta: -ll_func_vectorized(y_us_j, y_them_j, X_j, theta),
                    theta_init
                ).x

                theta_est_per_task[task] = project_l2_ball_numpy(theta_j, W)

        # --- Quoting and regret logging ---
        for i in range(ep_len):
            x_t = x_ep[i]
            y_them_i = y_ep[i]
            coupon = c_ep[i]
            ptimes = pt_ep[i]
            task = task_ep[i]

            if task not in theta_est_per_task:
                continue

            theta_proj = theta_est_per_task[task]
            quote_us = find_optimal_yield_with_cdf(theta_proj, x_t, coupon, ptimes)
            quote_them = y_them_i
            theta_star_j = theta_star_per_task.get(task, theta_proj)
            y_opt = find_optimal_yield_with_cdf(theta_star_j, x_t, coupon, ptimes)

            x_history.append(x_t)
            y_them_history.append(quote_them)
            if quote_us >= quote_them:
                y_us_history.append(quote_them)
            else:
                y_us_history.append(quote_us)
            task_history.append(task)

            r = regret(y_opt, quote_us, quote_them, coupon, ptimes, P_t = 100)
            step_regrets["individual"].append(r)

        start_idx = end_idx
        print(f"Episode {episode_idx + 1}: Total steps = {len(step_regrets['individual'])}, Regret = {np.sum(step_regrets['individual']):.4f}")

    return {
        "step_regrets": step_regrets,
        "cumulative_regret": {"individual": float(np.sum(step_regrets["individual"]))},
        "cumulative_regret_per_t": {"individual": list(np.cumsum(step_regrets["individual"]))}
    }

def plot_tsmt_regrets(cfg):
    """
    Run TSMT on real data for all 3 strategies (multi, pool, individual),
    and plot their cumulative regret.

    Args:
        cfg (Dict[str, Any]): Configuration dictionary used by TSMT runners.
    """
    print("Running MULTI strategy...")
    multi_results = run_tsmt_real_data_multi(cfg)
    print("Running POOL strategy...")
    pool_results = run_tsmt_real_data_pool(cfg)
    print("Running INDIVIDUAL strategy...")
    indiv_results = run_tsmt_real_data_individual(cfg)

    cum_multi = multi_results["cumulative_regret_per_t"]["multi"]
    cum_pool = pool_results["cumulative_regret_per_t"]["pool"]
    cum_indiv = indiv_results["cumulative_regret_per_t"]["individual"]

    # Ensure all are the same length
    T = min(len(cum_multi), len(cum_pool), len(cum_indiv))
    cum_multi = cum_multi[:T]
    cum_pool = cum_pool[:T]
    cum_indiv = cum_indiv[:T]

    plt.figure(figsize=(8, 6))
    plt.plot(cum_multi, label="multi", color="blue")
    plt.plot(cum_indiv, label="individual", color="red")
    plt.plot(cum_pool, label="pooling", color="green")

    plt.xlabel(r"$T$")
    plt.ylabel("Realized Accumulated Regret")
    plt.title("The realized accumulated regrets of the three algorithms.")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    #return run_tsmt_real_data_multi(cfg)
    return plot_tsmt_regrets(cfg)
if __name__ == "__main__":
    results = main()  # run the real TSMT algorithm
    with open("results_real.json", "w") as f:
        json.dump(results, f, indent=4)