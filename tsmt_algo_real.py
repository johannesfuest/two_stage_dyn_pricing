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
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from scipy.optimize import minimize_scalar

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

def load_real_data(csv_path: str):
    """
    Loads bond-related data from a CSV file and extracts features needed for pricing and analysis.

    Args:
        csv_path: Path to the CSV file containing the bond data.

    Returns:
        A dictionary with the following keys:
            - x_ts: A NumPy array of context features (principal components).
            - y_them: A NumPy array of yields (used later for oracle analysis).
            - coupon_rates: A NumPy array of coupon rates for each bond.
            - n_payments: A NumPy array of the number of remaining payments for each bond.
            - payment_times: A list of lists, where each inner list contains future payment times
              (in years) for each bond, assuming semiannual payments.
    """
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_path)

    # Extract the contextual features (first 4 principal components) as a NumPy array
    x_ts = df[["PC1", "PC2", "PC3", "PC4"]].values

    # Extract the yield (used as the competitor's quoted yield for oracle evaluation)
    y_them = df["weighted_avg_yield"].values

    # Extract coupon rates
    coupon_rates = df["Coupon Rate"].values

    # Extract the number of remaining payments, filling missing values with 0 and converting to int
    n_payments = df["remaining_payments"].fillna(0).astype(int).values

    # Generate a list of future payment times (in years), assuming semiannual payments
    payment_times = [list(0.5 * (i + 1) for i in range(n)) for n in n_payments]

    return {
        "x_ts": x_ts,
        "y_them": y_them,
        "coupon_rates": coupon_rates,
        "n_payments": n_payments,
        "payment_times": payment_times,
    }

def compute_oracle_and_regret(df_path: str):
    """
    Computes the oracle pricing model and the corresponding regrets.

    Args:
        df_path: Path to the CSV file containing bond transaction and feature data.

    Returns:
        A dictionary with:
            - oracle_theta: The estimated coefficient vector (theta) from the oracle Ridge model.
            - avg_regret: The average regret over all applicable bond entries.
            - y_us: A list of predicted yields using the oracle model.
            - regrets: A list of regret values for each bond.
    """

    # Load preprocessed data using the helper function
    data = load_real_data(df_path)
    X = data["x_ts"]  # Feature matrix (context vectors)
    y_them = data["y_them"]  # Observed yields (used for comparison)
    coupon_rates = data["coupon_rates"]  # Coupon rates of the bonds
    payment_times = data["payment_times"]  # Future payment schedules

    # Fit the "oracle" model using Ridge regression on the full dataset
    # This serves as a best-case comparator (oracle)
    theta_star = Ridge(alpha=1e-2, fit_intercept=False).fit(X, y_them).coef_

    # Initialize storage for regrets and predicted yields
    regrets = []
    y_us = []

    # Evaluate each bond instance
    for i, x_t in enumerate(X):
        coupon = coupon_rates[i]
        times = payment_times[i]

        # Skip bonds with no future payments (no valuation needed)
        if len(times) == 0:
            continue

        # Compute the predicted yield using the oracle model
        y_star = find_optimal_yield_with_cdf(theta_star, x_t, coupon, times)
        y_us.append(y_star)

        # Compute regret (comparing oracle's quote to observed competitor yield)
        # Here, self-regret is zero since y_star is used for both bids
        r = regret(y_star, y_star, y_them[i], coupon, times)
        regrets.append(r)

    # Return results including average regret and oracle predictions
    return {
        "oracle_theta": theta_star,
        "avg_regret": np.mean(regrets),
        "y_us": y_us,
        "regrets": regrets
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
    for t in future_payment_dates[:-1]:
        value += P_t * coupon_rate / (1 + y) ** t
    value += P_t / (1 + y) ** future_payment_dates[-1]
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
    return reward(y_t_opt, y_t_them, coupon_rate, future_payment_dates, P_t) - \
        reward(y_t_us, y_t_them, coupon_rate, future_payment_dates, P_t)

def find_optimal_yield_with_cdf(theta_star_j, x_t, coupon_rate, payment_times, gamma=0.0):
    """
    Finds the yield that maximizes the expected revenue given a competitor yield model and bond features.

    Args:
        theta_star_j: The estimated parameter vector (oracle theta) for bond j.
        x_t: The feature (context) vector for the current bond instance.
        coupon_rate: The coupon rate of the bond.
        payment_times: List of future payment times (in years).
        gamma: A price floor or cost term to adjust aggressiveness of pricing (default = 0.0).

    Returns:
        The yield that maximizes the expected revenue, or the predicted mean yield if optimization fails.
    """

    # Compute the predicted mean yield from the competitor model
    mu = np.dot(theta_star_j, x_t)

    # Define the objective function: negative expected reward
    def objective(y):
        # Convert yield to bond price using the coupon and payment schedule
        price = yield_to_price(100, coupon_rate, y, payment_times)

        # Estimate the probability of winning with this yield
        prob_win = noise_cdf(y - mu)

        # Return negative expected reward (for minimization)
        return -((price - gamma) * prob_win)

    # Use bounded scalar minimization to find the optimal yield in a reasonable range
    res = minimize_scalar(objective, bounds=(0.01, 0.15), method="bounded")

    # Return optimal yield if successful, otherwise fall back to the predicted mean yield
    return res.x if res.success else mu



def run_tsmt_real_data_pool(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs the pooled version of the TSMT (Two-Stage Multi-Task) pricing policy using real-world bond data.

    Args:
        cfg: Configuration dictionary (currently unused, but allows for future parameterization).

    Returns:
        A dictionary with:
            - step_regrets: Per-step regret values.
            - cumulative_regret: Total cumulative regret over all episodes.
            - cumulative_regret_per_t: Cumulative regret at each time step.
    """

    # --- Load and preprocess the bond dataset ---
    df = pd.read_csv("D:/Project Bond Price Illiquidity/final_df.csv")
    df = df[~df["weighted_avg_yield"].isna()]  # Remove rows with missing yields
    df["remaining_payments"] = df["remaining_payments"].fillna(0).astype(int)
    df = df[df["remaining_payments"] > 0].copy()  # Remove bonds with no remaining payments
    df["task_id"] = df["cusip_id"].astype("category").cat.codes  # Assign numeric task IDs per bond (by CUSIP)

    # Extract features and labels
    x_all = df[["PC1", "PC2", "PC3", "PC4"]].values  # Contextual features
    y_all = df["weighted_avg_yield"].values  # Competitor yields
    coupons = df["Coupon Rate"].values
    n_payments = df["remaining_payments"].values
    payment_times = [list(0.5 * (i + 1) for i in range(n)) for n in n_payments]  # Semiannual future payments
    task_ids = df["task_id"].values
    tasks = np.unique(task_ids)

    d = x_all.shape[1]  # Feature dimension
    K = 11  # Number of episodes
    M = len(tasks)  # Number of unique tasks (CUSIPs)

    # Predefine episode lengths and check sufficient data
    episode_lengths = [2 ** (k - 1) for k in range(1, K + 1)]
    total_required = sum(episode_lengths)
    assert len(x_all) >= total_required, "Not enough data for TSMT episodes"

    # --- Oracle model estimation: fit Ridge regression for each task ---
    from sklearn.linear_model import Ridge
    theta_star_per_task = {}
    for task in tasks:
        mask = task_ids == task
        X_j = x_all[mask]
        y_j = y_all[mask]
        model = Ridge(alpha=0.1, fit_intercept=True)
        theta_j = model.fit(X_j, y_j).coef_
        if theta_j.shape[0] == d:
            theta_star_per_task[task] = theta_j

    # Estimate W as an upper bound on the L2-norm of oracle thetas
    W = 1.2 * max(np.linalg.norm(theta) for theta in theta_star_per_task.values())
    W = 1  # Override if fixed norm is desired

    # --- Initialize tracking variables ---
    x_history, y_us_history, y_them_history, task_history = [], [], [], []
    step_regrets = {"pool": []}
    start_idx = 0

    # --- Run through episodes ---
    for episode_idx, ep_len in enumerate(episode_lengths):
        end_idx = start_idx + ep_len

        # Slice current episode's data
        x_ep = x_all[start_idx:end_idx]
        y_ep = y_all[start_idx:end_idx]
        c_ep = coupons[start_idx:end_idx]
        pt_ep = payment_times[start_idx:end_idx]
        task_ep = task_ids[start_idx:end_idx]

        # --- Stage I: Pooled MLE using historical data ---
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
            # If first episode, use random initialization
            theta_bar = np.random.randn(d)
            theta_bar /= np.linalg.norm(theta_bar)

        # Project the pooled estimate to the L2 ball of radius W
        theta_bar = project_l2_ball_numpy(theta_bar, W)
        print(f"[Episode {episode_idx + 1}] ||theta_bar|| = {np.linalg.norm(theta_bar):.6f}")

        # --- Quote prices and compute regrets for each round in the episode ---
        for i in range(ep_len):
            x_t = x_ep[i]
            y_them_i = y_ep[i]
            coupon = c_ep[i]
            ptimes = pt_ep[i]
            task = task_ep[i]

            # Our quote using pooled theta_bar
            quote_us = find_optimal_yield_with_cdf(theta_bar, x_t, coupon, ptimes)

            # Best competitor quote
            quote_them = y_them_i

            # Oracle quote using task-specific theta
            theta_star_j = theta_star_per_task.get(task, theta_bar)
            y_opt = find_optimal_yield_with_cdf(theta_star_j, x_t, coupon, ptimes)

            # Update history for the next episodes
            x_history.append(x_t)
            y_them_history.append(quote_them)
            y_us_history.append(quote_us)
            task_history.append(task)

            # Compute regret and log it
            r = regret(y_opt, quote_us, quote_them, coupon, ptimes, P_t=100)
            step_regrets["pool"].append(r)

        # Update start index for the next episode
        start_idx = end_idx
        print(f"Episode {episode_idx + 1}: Total steps = {len(step_regrets['pool'])}, Regret = {np.sum(step_regrets['pool']):.4f}")

    # --- Return final results ---
    return {
        "step_regrets": step_regrets,
        "cumulative_regret": {"pool": float(np.sum(step_regrets["pool"]))},
        "cumulative_regret_per_t": {"pool": list(np.cumsum(step_regrets["pool"]))}
    }

def run_tsmt_real_data_multi(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes the multi-task variant of the TSMT (Two-Stage Multi-Task) pricing policy on real-world bond data.

    Args:
        cfg: Configuration dictionary (placeholder for future extensions).

    Returns:
        A dictionary containing:
            - step_regrets: List of regrets per time step.
            - cumulative_regret: Total regret at the end of all episodes.
            - cumulative_regret_per_t: Running cumulative regret over time.
    """

    # --- Load and preprocess the bond dataset ---
    df = pd.read_csv("D:/Project Bond Price Illiquidity/final_df.csv")
    df = df[~df["weighted_avg_yield"].isna()]
    df["remaining_payments"] = df["remaining_payments"].fillna(0).astype(int)
    df = df[df["remaining_payments"] > 0].copy()
    df["task_id"] = df["cusip_id"].astype("category").cat.codes  # Map CUSIP to numeric task ID

    # Extract features and relevant bond data
    x_all = df[["PC1", "PC2", "PC3", "PC4"]].values  # Context vectors
    y_all = df["weighted_avg_yield"].values  # Competitor's yield quotes
    coupons = df["Coupon Rate"].values
    n_payments = df["remaining_payments"].values
    payment_times = [list(0.5 * (i + 1) for i in range(n)) for n in n_payments]  # Semiannual schedule
    task_ids = df["task_id"].values
    tasks = np.unique(task_ids)

    d = x_all.shape[1]  # Dimensionality of context features
    K = 11  # Number of episodes
    M = len(tasks)  # Number of unique tasks (CUSIPs)
    episode_lengths = [2 ** (k - 1) for k in range(1, K + 1)]
    total_required = sum(episode_lengths)
    assert len(x_all) >= total_required, "Not enough data for all episodes"

    # --- Estimate oracle theta_j* per task using Ridge regression ---
    from sklearn.linear_model import Ridge
    theta_star_per_task = {}
    for task in tasks:
        mask = task_ids == task
        X_j = x_all[mask]
        y_j = y_all[mask]
        model = Ridge(alpha=0.1, fit_intercept=True)
        theta_j = model.fit(X_j, y_j).coef_
        if theta_j.shape[0] == d:
            theta_star_per_task[task] = theta_j

    # Set the L2 ball radius W to regularize theta estimates
    W = 1.2 * max(np.linalg.norm(theta) for theta in theta_star_per_task.values())
    W = 1  # Optionally override with fixed norm

    # --- Initialize history and estimators ---
    x_history, y_us_history, y_them_history, task_history = [], [], [], []
    theta_est_per_task = {task: np.random.randn(d) for task in tasks}  # Task-specific estimators
    step_regrets = {"multi": []}
    start_idx = 0

    # --- Iterate over episodes ---
    for episode_idx, ep_len in enumerate(episode_lengths):
        end_idx = start_idx + ep_len

        # Extract current episode data
        x_ep = x_all[start_idx:end_idx]
        y_ep = y_all[start_idx:end_idx]
        c_ep = coupons[start_idx:end_idx]
        pt_ep = payment_times[start_idx:end_idx]
        task_ep = task_ids[start_idx:end_idx]

        # --- Stage I: Estimate shared theta_bar from previous episode's data ---
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

        # Project to L2 ball to ensure bounded norm
        theta_bar = project_l2_ball_numpy(theta_bar, W)
        print(f"[Episode {episode_idx + 1}] ||theta_bar|| = {np.linalg.norm(theta_bar):.6f}")

        # --- Stage II: Task-specific refinement using previous episode ---
        if episode_idx > 0:
            prev_start = start_idx - episode_lengths[episode_idx - 1]
            prev_end = start_idx

            for task in tasks:
                # Filter historical samples for the current task
                indices = [
                    i for i in range(prev_start, prev_end)
                    if task_history[i] == task
                ]
                if not indices:
                    continue  # No data for this task yet

                X_j = np.array([x_history[i] for i in indices])
                y_us_j = np.array([y_us_history[i] for i in indices])
                y_them_j = np.array([y_them_history[i] for i in indices])

                # Compute regularization parameter for task refinement
                u_F = 1.0
                x_max = np.max([np.linalg.norm(x) for x in x_history[prev_start:prev_end]]) if x_history else 1.0
                lambda_j_k = np.sqrt(8 * (u_F ** 2) * (x_max ** 2) * d * np.log(2 * d ** 2 * M) / len(indices))

                # Task-specific MLE with regularization around theta_bar
                theta_init = theta_est_per_task[task]
                theta_j = minimize(
                    lambda theta: -ll_func_vectorized(y_us_j, y_them_j, X_j, theta) +
                                  lambda_j_k * np.linalg.norm(theta - theta_bar),
                    theta_bar
                ).x

                # Project and store refined estimate
                theta_est_per_task[task] = project_l2_ball_numpy(theta_j, W)

        # --- Quoting and regret evaluation for this episode ---
        for i in range(ep_len):
            x_t = x_ep[i]
            y_them_i = y_ep[i]
            coupon = c_ep[i]
            ptimes = pt_ep[i]
            task = task_ep[i]

            # Select the refined estimator for this task
            theta_proj = theta_est_per_task.get(task, theta_bar)

            # Quote using refined theta and evaluate regret
            quote_us = find_optimal_yield_with_cdf(theta_proj, x_t, coupon, ptimes)
            quote_them = y_them_i
            theta_star_j = theta_star_per_task.get(task, theta_bar)
            y_opt = find_optimal_yield_with_cdf(theta_star_j, x_t, coupon, ptimes)

            # Update historical data
            x_history.append(x_t)
            y_them_history.append(quote_them)
            y_us_history.append(quote_us)
            task_history.append(task)

            # Compute and log regret
            r = regret(y_opt, quote_us, quote_them, coupon, ptimes, P_t=100)
            step_regrets["multi"].append(r)

        # Update episode boundary
        start_idx = end_idx
        print(f"Episode {episode_idx + 1}: Total steps = {len(step_regrets['multi'])}, Regret = {np.sum(step_regrets['multi']):.4f}")

    # --- Final results ---
    return {
        "step_regrets": step_regrets,
        "cumulative_regret": {"multi": float(np.sum(step_regrets["multi"]))},
        "cumulative_regret_per_t": {"multi": list(np.cumsum(step_regrets["multi"]))}
    }


def run_tsmt_real_data_individual(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes the individual-task (non-pooled, non-regularized) version of the TSMT pricing policy
    on real bond data, where each task (CUSIP) is estimated independently using MLE.

    Args:
        cfg: Configuration dictionary (currently unused, reserved for future options).

    Returns:
        A dictionary with:
            - step_regrets: List of per-step regrets for the individual strategy.
            - cumulative_regret: Total regret at the end of all episodes.
            - cumulative_regret_per_t: List of cumulative regret values at each time step.
    """

    # --- Load and preprocess data ---
    df = pd.read_csv("D:/Project Bond Price Illiquidity/final_df.csv")
    df = df[~df["weighted_avg_yield"].isna()]  # Drop rows with missing yield
    df["remaining_payments"] = df["remaining_payments"].fillna(0).astype(int)
    df = df[df["remaining_payments"] > 0].copy()  # Keep only bonds with future payments
    df["task_id"] = df["cusip_id"].astype("category").cat.codes  # Assign numeric ID per bond

    # Extract input features and labels
    x_all = df[["PC1", "PC2", "PC3", "PC4"]].values
    y_all = df["weighted_avg_yield"].values
    coupons = df["Coupon Rate"].values
    n_payments = df["remaining_payments"].values
    payment_times = [list(0.5 * (i + 1) for i in range(n)) for n in n_payments]  # Semiannual times
    task_ids = df["task_id"].values
    tasks = np.unique(task_ids)

    d = x_all.shape[1]  # Context dimension
    K = 11  # Number of episodes
    M = len(tasks)  # Number of tasks (CUSIPs)
    episode_lengths = [2 ** (k - 1) for k in range(1, K + 1)]
    total_required = sum(episode_lengths)
    assert len(x_all) >= total_required, "Not enough data for all episodes"

    # --- Estimate oracle thetas using full historical data (for evaluation only) ---
    from sklearn.linear_model import Ridge
    theta_star_per_task = {}
    for task in tasks:
        mask = task_ids == task
        X_j = x_all[mask]
        y_j = y_all[mask]
        model = Ridge(alpha=0.1, fit_intercept=True)
        theta_j = model.fit(X_j, y_j).coef_
        if theta_j.shape[0] == d:
            theta_star_per_task[task] = theta_j

    # Set L2 norm bound W for projection
    W = 1.2 * max(np.linalg.norm(theta) for theta in theta_star_per_task.values())
    W = 1  # Fixed bound override

    # --- Initialize histories and task-specific estimators ---
    x_history, y_us_history, y_them_history, task_history = [], [], [], []
    theta_est_per_task = {task: np.random.randn(d) for task in tasks}
    step_regrets = {"individual": []}
    start_idx = 0

    # --- Run through episodes ---
    for episode_idx, ep_len in enumerate(episode_lengths):
        end_idx = start_idx + ep_len
        x_ep = x_all[start_idx:end_idx]
        y_ep = y_all[start_idx:end_idx]
        c_ep = coupons[start_idx:end_idx]
        pt_ep = payment_times[start_idx:end_idx]
        task_ep = task_ids[start_idx:end_idx]

        # --- Stage: Individual task-specific MLE (no regularization or pooling) ---
        if episode_idx > 0:
            prev_start = start_idx - episode_lengths[episode_idx - 1]
            prev_end = start_idx
            for task in tasks:
                # Collect historical data points for current task
                indices = [i for i in range(start_idx) if task_history[i] == task]
                if not indices:
                    continue

                X_j = np.array([x_history[i] for i in indices])
                y_us_j = np.array([y_us_history[i] for i in indices])
                y_them_j = np.array([y_them_history[i] for i in indices])

                theta_init = theta_est_per_task[task]
                theta_init /= np.linalg.norm(theta_init)  # Normalize for stability

                # MLE without regularization
                theta_j = minimize(
                    lambda theta: -ll_func_vectorized(y_us_j, y_them_j, X_j, theta),
                    theta_init
                ).x

                # Project to L2 ball
                theta_est_per_task[task] = project_l2_ball_numpy(theta_j, W)

        # --- Pricing and regret logging ---
        for i in range(ep_len):
            x_t = x_ep[i]
            y_them_i = y_ep[i]
            coupon = c_ep[i]
            ptimes = pt_ep[i]
            task = task_ep[i]

            # If task hasn't been estimated, skip
            if task not in theta_est_per_task:
                continue

            theta_proj = theta_est_per_task[task]
            quote_us = find_optimal_yield_with_cdf(theta_proj, x_t, coupon, ptimes)
            quote_them = y_them_i
            theta_star_j = theta_star_per_task.get(task, theta_proj)
            y_opt = find_optimal_yield_with_cdf(theta_star_j, x_t, coupon, ptimes)

            # Record history
            x_history.append(x_t)
            y_them_history.append(quote_them)
            y_us_history.append(quote_us)
            task_history.append(task)

            # Compute regret
            r = regret(y_opt, quote_us, quote_them, coupon, ptimes, P_t=100)
            step_regrets["individual"].append(r)

        # Move to next episode
        start_idx = end_idx
        print(f"Episode {episode_idx + 1}: Total steps = {len(step_regrets['individual'])}, Regret = {np.sum(step_regrets['individual']):.4f}")

    # --- Return final statistics ---
    return {
        "step_regrets": step_regrets,
        "cumulative_regret": {"individual": float(np.sum(step_regrets["individual"]))},
        "cumulative_regret_per_t": {"individual": list(np.cumsum(step_regrets["individual"]))}
    }

def plot_tsmt_regrets(cfg):
    """
    Run TSMT on real bond data using three strategies:
        - Multi-task (TSMT with task-specific refinement),
        - Pooled (single shared model),
        - Individual (independent models per task with no pooling),
    and plot their cumulative regrets over time.

    Args:
        cfg (Dict[str, Any]): Configuration dictionary passed to each TSMT runner.
    """

    # --- Run multi-task strategy ---
    print("Running MULTI strategy...")
    multi_results = run_tsmt_real_data_multi(cfg)

    # --- Run pooled (single-model) strategy ---
    print("Running POOL strategy...")
    pool_results = run_tsmt_real_data_pool(cfg)

    # --- Run individual (per-task MLE) strategy ---
    print("Running INDIVIDUAL strategy...")
    indiv_results = run_tsmt_real_data_individual(cfg)

    # --- Extract cumulative regrets from results ---
    cum_multi = multi_results["cumulative_regret_per_t"]["multi"]
    cum_pool = pool_results["cumulative_regret_per_t"]["pool"]
    cum_indiv = indiv_results["cumulative_regret_per_t"]["individual"]

    # --- Align all sequences to the same length (for fair comparison) ---
    T = min(len(cum_multi), len(cum_pool), len(cum_indiv))
    cum_multi = cum_multi[:T]
    cum_pool = cum_pool[:T]
    cum_indiv = cum_indiv[:T]

    # --- Plot the cumulative regrets ---
    plt.figure(figsize=(10, 7))
    plt.plot(cum_multi, label="Multi", color="blue")       # TSMT multi-task
    plt.plot(cum_indiv, label="Individual", color="red")   # Per-task model
    plt.plot(cum_pool, label="Pooled", color="green")      # Shared model

    plt.xlabel(r"$T$")  # Time steps
    plt.ylabel("Realized Accumulated Regret")
    plt.title("Accumulated Regret of Online Pricing Strategies")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save figure to file and show plot
    plt.savefig("accumulated_regret.png", dpi=300)
    plt.show()

@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    #return run_tsmt_real_data_multi(cfg)
    return plot_tsmt_regrets(cfg)
if __name__ == "__main__":
    results = main()  # run the real TSMT algorithm
    with open("results_real.json", "w") as f:
        json.dump(results, f, indent=4)