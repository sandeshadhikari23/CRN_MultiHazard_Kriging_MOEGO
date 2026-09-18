import numpy as np
from scipy.stats import norm
from scipy.linalg import cholesky, solve_triangular
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

NUGGET = np.float64(1e-6)

# ------- New functions for paper’s KPLS kernel -------

def _weighted_sqdist(Z1, Z2, eta):
    """
    Squared distance with per‑dimension weights eta (1D array of length d).
    Z1, Z2 are (n,d) arrays scaled by the PLS standardisation.
    """
    Z1 = np.asarray(Z1, dtype=np.float64)
    Z2 = np.asarray(Z2, dtype=np.float64)
    # scale coordinates by sqrt(eta) so that Euclidean distance gives sum_i eta_i (Δz_i)^2
    scale = np.sqrt(eta)
    return cdist(Z1 * scale, Z2 * scale, 'sqeuclidean')

def fit_kpls_model(X, Y, n_components=3, initial_theta=None):
    """
    Fit KPLS model exactly as in Bouhlel et al. (2016).
    (No dependency on PLSRegression.x_mean_ / x_std_)
    """
    from sklearn.cross_decomposition import PLSRegression

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64).reshape(-1, 1)
    n, d = X.shape
    h = min(n_components, n - 1, d)

    # ----- 1. Manual scaling of X (centre + unit variance) -----
    x_mean = np.mean(X, axis=0, keepdims=True)
    x_std  = np.std(X, axis=0, ddof=1, keepdims=True)   # ddof=1 matches PLS's default
    # Prevent division by zero (constant features)
    x_std[x_std < 1e-12] = 1.0
    Z_train = (X - x_mean) / x_std

    # ----- 2. PLS on the scaled data (scale=False to avoid internal re‑scaling) -----
    pls = PLSRegression(n_components=h, scale=False, max_iter=500)
    pls.fit(Z_train, Y.ravel())

    # The weight matrix W* (paper’s notation)
    W = pls.x_rotations_   # shape (d, h) – compatible with all sklearn versions

    # ----- 3. Build effective eta for a given theta (h,) -----
    def make_eta(theta_pls):
        # eta_i = sum_l theta_l * W[i,l]^2
        return (W ** 2) @ theta_pls   # (d, h) @ (h,) -> (d,)

    # ----- 4. Negative log‑likelihood -----
    def neg_log_likelihood(params):
        theta_pls = np.exp(params)          # ensures positivity
        eta = make_eta(theta_pls)
        # correlation matrix: R_ij = exp( - sum_k eta_k (z_ik - z_jk)^2 )
        dist2 = _weighted_sqdist(Z_train, Z_train, eta)
        R = np.exp(-dist2)
        R += np.eye(n, dtype=np.float64) * NUGGET
        R = 0.5 * (R + R.T)
        try:
            L = cholesky(R, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return np.float64(1e12)

        one = np.ones((n, 1), dtype=np.float64)
        v = solve_triangular(L, Y, lower=True, check_finite=False)
        w = solve_triangular(L, one, lower=True, check_finite=False)
        mu = (w.T @ v) / (w.T @ w)
        residual = Y - mu
        tmp = solve_triangular(L, residual, lower=True, check_finite=False)
        alpha = solve_triangular(L.T, tmp, lower=False, check_finite=False)
        sigma2 = (residual.T @ alpha) / n
        if sigma2 <= 0 or not np.isfinite(sigma2):
            return np.float64(1e12)
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        return 0.5 * (log_det + n * np.log(sigma2))

    # ----- 5. Optimise theta_pls -----
    init_log_theta = np.full(h, np.log(0.5 if initial_theta is None else initial_theta))
    bounds = [(-12.0, 12.0)] * h
    res = minimize(neg_log_likelihood, init_log_theta,
                   method='L-BFGS-B', bounds=bounds)
    if not res.success:
        print("Warning: KPLS MLE did not converge:", res.message)

    theta_pls_opt = np.exp(res.x)
    eta_opt = make_eta(theta_pls_opt)

    # ----- 6. Final stable correlation matrix & MLE parameters -----
    R = np.exp(-_weighted_sqdist(Z_train, Z_train, eta_opt))
    R += np.eye(n, dtype=np.float64) * NUGGET
    jitter = NUGGET
    L = None
    for _ in range(8):
        try:
            L = cholesky(R + np.eye(n, dtype=np.float64) * jitter, lower=True, check_finite=False)
            break
        except np.linalg.LinAlgError:
            jitter *= np.float64(10.0)
    if L is None:
        vals, vecs = np.linalg.eigh(R)
        floor = max(np.float64(1e-12), np.max(np.abs(vals)) * np.float64(1e-12))
        vals_clipped = np.clip(vals, floor, None)
        R = (vecs * vals_clipped) @ vecs.T
        R = 0.5 * (R + R.T)
        L = cholesky(R, lower=True, check_finite=False)

    one = np.ones((n, 1), dtype=np.float64)
    v = solve_triangular(L, Y, lower=True, check_finite=False)
    w = solve_triangular(L, one, lower=True, check_finite=False)
    mu_hat = np.float64((w.T @ v) / (w.T @ w))
    residual = Y - mu_hat
    tmp = solve_triangular(L, residual, lower=True, check_finite=False)
    alpha = solve_triangular(L.T, tmp, lower=False, check_finite=False)
    sigma_hat_sq = np.float64((residual.T @ alpha) / n)
    if sigma_hat_sq <= 0 or not np.isfinite(sigma_hat_sq):
        sigma_hat_sq = np.float64(1e-8)

    R_inv = np.linalg.inv(R + np.eye(n, dtype=np.float64) * np.float64(1e-12))

    return (mu_hat, sigma_hat_sq, R_inv, theta_pls_opt, W, eta_opt,
            x_mean, x_std)

def kpls_predict(x, Z_train, Y_train, mu_hat, sigma_hat_sq,
                 R_inv, eta, x_mean, x_std):
    """
    Predict mean and std at a single (raw) point x using KPLS model.
    Z_train is the standardised training matrix used in fitting.
    """
    x = np.asarray(x, dtype=np.float64).reshape(1, -1)
    # standardise
    z = (x - x_mean) / x_std
    # r vector
    r = np.exp(-_weighted_sqdist(z, Z_train, eta)).T  # (n,1)
    n = Z_train.shape[0]
    one = np.ones((n, 1), dtype=np.float64)
    Y_minus_mu = Y_train - mu_hat
    y_hat = mu_hat + (r.T @ (R_inv @ Y_minus_mu))
    term1 = 1.0 - (r.T @ (R_inv @ r))
    denom = one.T @ (R_inv @ one)
    if denom <= 0 or not np.isfinite(denom):
        denom += 1e-12
    term2 = ((1.0 - (one.T @ (R_inv @ r))) ** 2) / denom
    s2 = sigma_hat_sq * (term1 + term2)
    if s2 < 0 or not np.isfinite(s2):
        s2 = 0.0
    return float(y_hat), float(np.sqrt(s2))

def kpls_expected_improvement(x, Z_train, Y_train, mu_hat, sigma_hat_sq,
                              R_inv, eta, x_mean, x_std, y_min):
    """
    Expected Improvement using the paper’s KPLS model.
    x : 1‑D array (raw binary vector)
    """
    y_hat, s_hat = kpls_predict(x, Z_train, Y_train, mu_hat, sigma_hat_sq,
                                R_inv, eta, x_mean, x_std)
    if s_hat < 1e-12:
        return np.float64(0.0)
    Z = (np.float64(y_min) - np.float64(y_hat)) / np.float64(s_hat)
    ei = (np.float64(y_min) - np.float64(y_hat)) * norm.cdf(Z) + np.float64(s_hat) * norm.pdf(Z)
    if not np.isfinite(ei) or ei < 0.0:
        return np.float64(0.0)
    return np.float64(ei)