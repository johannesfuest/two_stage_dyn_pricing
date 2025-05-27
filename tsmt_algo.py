import numpy as np
from omegaconf import DictConfig, OmegaConf
import json
from typing import List, Dict, Any, Union
from scipy.optimize import minimize, minimize_scalar
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
    sample_w_sphere,
)
from loglikelihood import ll_and_grad


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
    x_ts = gen_x_ts(cfg.M, cfg.d)
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
) -> float:
    """
    Calculates the log-likelihood function a single observation.
    
    Args:
        y_t_us (float): The yield we quote.
        y_t_them (float): The best competitor's yield.
        x_t (np.ndarray): The context vector for the security.
        theta (np.ndarray): The parameter vector.
        noise (float): The noise term.
        j (int): The index of the security.
        noise_pdf (function): The probability density function of the noise.
        noise_cdf (function): The cumulative distribution function of the noise.
    Returns:
        float: The log-likelihood value.
    """
    we_won = y_t_us > y_t_them
    if we_won:
        # We won the auction
        val = np.log(1 - np.clip(noise_cdf(y_t_them - (theta.T@x_t).item()), 0, 1-1e-15))
        return val 
    else:
        # We lost the auction
        return np.log(noise_pdf(y_t_us - (theta.T@x_t).item()))


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
        step = ll_func(
            yts_us[i],
            yts_them[i],
            x_ts[i],
            theta,
        )
        if np.isnan(step) or np.isinf(step):  # Skip invalid log-likelihood values
            raise ValueError("Invalid log-likelihood value encountered.")
        ll += step
    ll /= len(yts_us)
    return ll


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
        yield (float): The yield to maturity (discount rate).
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


def gen_optimal_quote(
    theta_star_est: np.ndarray,
    x_t: np.ndarray,
    W: float,
    P_t: float,
    coupon_rate: float,
    future_payment_dates: List[float],
    y_bar: float=20,
    gamma: float=0,
) -> float:
    """
    Generate the optimal quote based on the estimated parameters and context.
    Args:
        theta_star_est (np.ndarray): The estimated parameters.
        x_t (np.ndarray): The context vector for the security.
        W (float): The radius of the L2 ball we project onto.
        P_t (float): The face/par value of the bond.
        coupon_rate (float): The coupon rate of the security.
        future_payment_dates (List[float]): The future payment dates for the security in years from now.
        p_bar (float): Upper bound for the yield. Default is 1. # TODO: check what this is in the paper
        gamma (float): The aggressiveness parameter. Default is 0.
    Returns:
        float: The optimal quote value.
    """
    theta_star_est = project_l2_ball_numpy(theta_star_est, W)
    
    def objective(
        b: float,
        our_yield: float,
        P_t: float=P_t,
        coupon_rate: float=coupon_rate,
        future_payment_dates: List[float]=future_payment_dates,
        gamma: float=gamma,
    ) -> float:
        """
        Objective function to maximize expected reward.
        Args:
            b (float): The bond price.
            our_yield (float): The yield we quote.
        Returns:
            float: The optimal yield we should quote at.
        """
        our_yield = our_yield.item()
        return (yield_to_price(P_t, coupon_rate, our_yield, future_payment_dates) - gamma) * noise_cdf(our_yield - b)
    
    optimal_yield = minimize_scalar(
        lambda y: -objective(
            b=(theta_star_est.T@x_t).item(),  # best competitor's yield
            our_yield=y,
            P_t=P_t,
            coupon_rate=coupon_rate,
            future_payment_dates=future_payment_dates,
            gamma=gamma,
        ),
        bounds=[-0.999, y_bar],
    )
    optimal_yield = optimal_yield.x.item()
    return optimal_yield
    

def reward(
    y_t_us: float,
    y_t_them: float,
    gamma: float,
    P_t: float,  # Default face value of the bond
    coupon_rate: float,  # Default coupon rate
    future_payment_dates: List[float],  # Default future payment dates in years
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
    price = yield_to_price(
        P_t=P_t,
        coupon_rate=coupon_rate,
        y=y_t_us,
        future_payment_dates=future_payment_dates
    )
    return price - gamma if y_t_us > y_t_them else 0


def regret(
    y_t_opt: float,
    y_t_us: float,
    y_t_them: float,
    gamma: float,
    P_t: float,
    coupon_rate: float,
    future_payment_dates: List[float],
) -> float:
    """
    Calculate the regret based on the optimal yield and the quoted yield.
    
    Args:
        y_t_opt (float): The optimal yield.
        y_t_us (float): The quoted yield.
        y_t_them (float): The best competitor's yield.
        gamma (float): The aggressiveness parameter.
        P_t (float): The face/par value of the bond.
        coupon_rate (float): The coupon rate of the security.
        future_payment_dates (List[float]): The future payment dates for the security in years from now.
        
    Returns:
        float: The regret value.
    """
    reward_opt = reward(
        y_t_opt,
        y_t_them,
        gamma,
        P_t,
        coupon_rate,
        future_payment_dates,
    )
    reward_us = reward(
        y_t_us,
        y_t_them,
        gamma,
        P_t,
        coupon_rate,
        future_payment_dates,
    )
    return reward_opt - reward_us


def run_tsmt_algo_synth(
    cfg: DictConfig,
) -> Dict[str, Any]:
    """
    Run the TSMT algorithm using the provided configuration and synthetic data.
    
    Args:
        cfg (DictConfig): Configuration object containing parameters for the TSMT algorithm.
        
    Returns:
        Dict[str, Any]: Dictionary containing the results of the TSMT algorithm.
    """
    # Generate synthetic data
    data_dict = get_synthetic_data(cfg)
    results = {}
    match cfg.init_mode:
            case "zero":
                prev_theta_bar = np.zeros((cfg.d, 1))
                theta_est = [prev_theta_bar for _ in range(cfg.M)]
            case "sample":
                prev_theta_bar = sample_w_sphere(
                    d=cfg.d,
                    W=cfg.W,
                )
                theta_est = [prev_theta_bar for _ in range(cfg.M)]
            case _:
                raise ValueError("Invalid initialization mode")
    # episode 1
    prev_arrivals = data_dict["arrivals"][0]
    prev_quotes_us = [gen_optimal_quote(
        theta_star_est=prev_theta_bar,
        x_t=data_dict["x_ts"][prev_arrivals[0]],
        W=cfg.W,
        P_t=1,
        coupon_rate=data_dict["coupon_rates"][prev_arrivals[0]],
        future_payment_dates=data_dict["remaining_payments"][prev_arrivals[0]],
        y_bar=1,
        gamma=cfg.gamma,
    )]

    prev_quotes_them = [(data_dict["theta_stars"][prev_arrivals[0]].T@data_dict["x_ts"][prev_arrivals[0]]).item() + data_dict["noise"][0]]
    prev_noises = [data_dict["noise"][0]]
    prev_optimal_quotes = [
        gen_optimal_quote(
            theta_star_est=data_dict["theta_stars"][prev_arrivals[0]],
            x_t=data_dict["x_ts"][prev_arrivals[0]],
            W=cfg.W,
            P_t=1,
            coupon_rate=data_dict["coupon_rates"][prev_arrivals[0]],
            future_payment_dates=data_dict["remaining_payments"][prev_arrivals[0]],
            y_bar=1,
            gamma=cfg.gamma,
        )
    ]
    results["data_dict"] = data_dict
    reg = regret(
        prev_optimal_quotes[0],
        prev_quotes_us[0],
        prev_quotes_them[0],
        cfg.gamma,
        P_t=1,
        coupon_rate=data_dict["coupon_rates"][prev_arrivals[0]],
        future_payment_dates=data_dict["remaining_payments"][prev_arrivals[0]],
    )
    results["1"] = {
        "theta_est": theta_est,
        "theta_bar_est": prev_theta_bar,
        "quotes_us": prev_quotes_us,
        "quotes_them": prev_quotes_them,
        "optimal_yields": prev_optimal_quotes,
        "noises": prev_noises,
        "arrivals": prev_arrivals,
        "regret": reg,
    }
    start = 1
    for episode in range(2, cfg.k):
        # Stage 1: MLE across all securities from prev episode
        x_ts = []
        for arrival in prev_arrivals:
            x_ts.append(data_dict["x_ts"][arrival])
        
        # theta_bar_est_res = minimize(
        #     lambda theta: -ll_func_sum(
        #         prev_quotes_us,
        #         prev_quotes_them,
        #         x_ts,
        #         theta.reshape(-1, 1),
        #     ),
        #     prev_theta_bar.reshape(cfg.d,),
        # )
        theta_bar_est_res = minimize(
            fun=ll_and_grad,
            x0=prev_theta_bar.reshape(cfg.d,),
            args=(prev_quotes_us, prev_quotes_them, x_ts),
            jac=True,
            method='L-BFGS-B',
            options=dict(maxiter=1000, gtol=1e-6),
        )
        
        theta_bar_est = theta_bar_est_res.x.reshape(-1, 1)
        theta_est_new = [theta_bar_est for _ in range(cfg.M)]
        # Stage 2: MLE for each security individually
        for security in range(cfg.M):
            n_security = sum([1 for arrival in prev_arrivals if arrival == security])
            if n_security == 0:
                # if no arrivals, skip this security and use theta bar
                continue
            x_ts = [data_dict["x_ts"][security] for _ in range(n_security)]
            y_ts_us = [y for i, y in enumerate(prev_quotes_us) if prev_arrivals[i] == security]
            yts_them = [y for i, y in enumerate(prev_quotes_them) if prev_arrivals[i] == security]
            lambda_j_k = 0.1 * np.sqrt((cfg.d / n_security))
            # theta_j_est = minimize(
            #     lambda theta: -ll_func_sum(
            #         y_ts_us,
            #         yts_them,
            #         x_ts,
            #         theta.reshape(-1, 1),
            #     ) + lambda_j_k * np.linalg.norm(theta - theta_bar_est), # regularization term is different for synthetic data (see page 20)
            #     theta_est[security].reshape(cfg.d,),
            # ).x
            theta_j_est_res = minimize(
                fun=ll_and_grad,
                x0=theta_est[security].reshape(cfg.d,),
                args=(y_ts_us, yts_them, x_ts),
                jac=True,
                method='L-BFGS-B',
                options=dict(maxiter=1000, gtol=1e-6),
            ) # TODO: add regularization term
            theta_est_new[security] = theta_j_est_res.x.reshape(-1, 1)
        episode_size = 2**(episode-1)
        episode_arrivals = data_dict["arrivals"][episode-1]
        
        episode_noises = data_dict["noise"][start:start + episode_size]
        prev_arrivals = episode_arrivals
        prev_quotes_us = []
        prev_quotes_them = []
        prev_optimal_yields = []
        for i in range(episode_size):
            security = episode_arrivals[i]
            prev_quotes_us.append(
                gen_optimal_quote(
                    theta_star_est=theta_est_new[security],
                    x_t=data_dict["x_ts"][security],
                    W=cfg.W,
                    P_t=1,
                    coupon_rate=data_dict["coupon_rates"][security],
                    future_payment_dates=data_dict["remaining_payments"][security],
                    y_bar=1,
                    gamma=cfg.gamma,
                )
            )
            prev_quotes_them.append((data_dict["theta_stars"][security].T @ data_dict["x_ts"][security]).item() + episode_noises[i])
            prev_optimal_yields.append(
                gen_optimal_quote(
                    theta_star_est=data_dict["theta_stars"][security],
                    x_t=data_dict["x_ts"][security],
                    W=cfg.W,
                    P_t=1,
                    coupon_rate=data_dict["coupon_rates"][security],
                    future_payment_dates=data_dict["remaining_payments"][security],
                    y_bar=1,
                    gamma=cfg.gamma,
                )
            )
        
        regs = []
        for i in range(len(episode_noises)):
            regs.append(
                regret(
                    prev_optimal_yields[i],
                    prev_quotes_us[i],
                    prev_quotes_them[i],
                    cfg.gamma,
                    P_t=1,
                    coupon_rate=data_dict["coupon_rates"][prev_arrivals[i]],
                    future_payment_dates=data_dict["remaining_payments"][prev_arrivals[i]],
                )
            )
        start += episode_size
        results[str(episode)] = {
            "theta_est": theta_est_new,
            "theta_bar_est": theta_bar_est,
            "quotes_us": prev_quotes_us,
            "quotes_them": prev_quotes_them,
            "optimal_yields": prev_optimal_yields,
            "noises": episode_noises,
            "arrivals": episode_arrivals,
            "regret": regs,
        }
    return results


def main(cfg: DictConfig):
    return run_tsmt_algo_synth(cfg)

if __name__ == "__main__":
    # Load the configuration
    cfg = OmegaConf.load("config.yaml")
    results = main(cfg) # run the TSMT algorithm and get the results
    # Save the results to a file
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)