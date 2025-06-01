import numpy as np
import scipy.stats as stats

EPS = 1e-12

def safe_log1m(x: float) -> float:
    x = np.clip(x, EPS, 1.0 - EPS)
    return np.log1p(-x)          # log(1-x) but stable

def safe_log(x: float) -> float:
    return np.log(np.clip(x, EPS, None))

def _trunc_pdf(z, mu, sigma, a, b):
    a_std, b_std = (a-mu)/sigma, (b-mu)/sigma
    Z = stats.truncnorm(a_std, b_std, loc=mu, scale=sigma)
    return Z.pdf(z)

def _trunc_cdf(z, mu, sigma, a, b):
    a_std, b_std = (a-mu)/sigma, (b-mu)/sigma
    Z = stats.truncnorm(a_std, b_std, loc=mu, scale=sigma)
    return Z.cdf(z)

def _grad_one(y_us, y_them, x, theta,
              mu=0.05, sigma=0.05, a=0.02, b=0.11):
    """Return log-lik and gradient for one observation."""
    x = x.reshape(-1, 1)              # ensure column vector
    win = y_us > y_them
    if not win:
        z = y_us - float(theta.T @ x)
        cdf = np.clip(_trunc_cdf(z, mu, sigma, a, b), EPS, 1 - EPS)
        pdf = _trunc_pdf(z, mu, sigma, a, b)
        ll  = np.log1p(-cdf)
        grad = (pdf / (1.0 - cdf)) * x   
    else:
        z = y_them - float(theta.T @ x)
        pdf = np.clip(_trunc_pdf(z, mu, sigma, a, b), EPS, None)
        ll  = np.log(pdf)
        grad = ((z - mu) / sigma**2) * x
    return ll, grad


def ll_and_grad(theta_flat, y_us, y_them, X):
    """Return *negative* log-likelihood and its gradient (for minimise)."""
    theta = theta_flat.reshape(-1, 1)
    ll_sum = 0.0
    grad_sum = np.zeros_like(theta)
    for i in range(len(y_us)):
        ll_i, g_i = _grad_one(y_us[i], y_them[i], X[i], theta)
        ll_sum   += ll_i
        grad_sum += g_i
    ll_mean   = ll_sum   / len(y_us)
    grad_mean = grad_sum / len(y_us)
    return -ll_mean.item(), -grad_mean.ravel()    # negate for minimisation


def _reg_value_grad(theta: np.ndarray,
                    theta_fixed: np.ndarray,
                    lam: float,
                    eps: float = 1e-12
) -> tuple[float, np.ndarray]:
    """
    Return the value and gradient of the regularisation term.
    Args:
        theta:          The parameter vector.
        theta_fixed:    A fixed parameter vector (e.g., the true parameter).
        lam:            Regularisation strength.
        eps:            Small value to avoid division-by-zero.
        
    Returns:
        tuple: A tuple containing the value and gradient of the regularisation term.

    * Uses a small `eps` only to avoid division-by-zero if the two
      vectors coincide; mathematically the sub-gradient is 0 there.
    """
    diff  = theta - theta_fixed
    norm  = np.linalg.norm(diff)
    if norm < eps:              # sub-gradient at 0
        grad = np.zeros_like(theta)
    else:
        grad = lam * diff / norm
    value = lam * norm
    return value, grad


def ll_reg_and_grad(theta_flat: np.ndarray,
                    y_us:   np.ndarray | list,
                    y_them: np.ndarray | list,
                    X,                     # 2-D array (T,d) or list of length T
                    lam: float,
                    theta_fixed: np.ndarray
) -> tuple[float, np.ndarray]:
    """
    Return (objective, gradient) for the case with regularisation.

    Ready for `scipy.optimize.minimize(..., jac=True)`.
    """
    # --- reshape and basic containers -------------------------------------
    theta = theta_flat.reshape(-1, 1)               # (d,1)

    ll_sum  = 0.0
    g_sum   = np.zeros_like(theta)

    T = len(y_us)
    for i in range(T):
        ll_i, g_i = _grad_one(float(y_us[i]),
                              float(y_them[i]),
                              X[i], theta)
        ll_sum  += ll_i
        g_sum   += g_i

    mean_ll   = ll_sum / T               # scalar
    mean_grad = g_sum  / T               # (d,1)

    # --- regularisation ----------------------------------------------------
    reg_val, reg_grad = _reg_value_grad(theta, theta_fixed, lam)

    # --- total objective (minimisation) -----------------------------------
    obj  = -mean_ll + reg_val
    grad = (-mean_grad + reg_grad).ravel()   # flatten for SciPy

    return obj, grad