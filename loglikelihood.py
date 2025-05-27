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
    if win:
        z = y_them - float(theta.T @ x)
        cdf = np.clip(_trunc_cdf(z, mu, sigma, a, b), EPS, 1 - EPS)
        pdf = _trunc_pdf(z, mu, sigma, a, b)
        ll  = np.log1p(-cdf)          # log(1-cdf)
        grad = (pdf / (1.0 - cdf)) * x
    else:
        z = y_us   - float(theta.T @ x)
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