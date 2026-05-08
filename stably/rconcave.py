"""
r-concave tail probability bound from Shah & Samworth (2013), Appendix A.4.

Implements the D(eta, t, N, r) function and the threshold selection procedure
from equation (8) of the paper:

    E|S^CPSS ∩ L_theta| <= min{D(theta^2, 2*tau-1, B, -1/2),
                                D(theta, tau, 2B, -1/4)} * p

This is the tighter bound that is the main contribution of Shah & Samworth
over the Meinshausen & Buhlmann (2010) worst-case bound.
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import logsumexp


def _log_vals(a, k, inv_r):
    """Compute log((a + i)^{1/r}) = inv_r * log(a + i) for i = 0, ..., k."""
    return inv_r * np.log(np.arange(k + 1) + a)


def _expectation_linear(a, k, N, r):
    """
    Expectation of the r-concave PMF f_{a,k} whose r-th power is linear.

    f_{a,k;i} = (a + i)^{1/r} / sum_{j=0}^{k} (a + j)^{1/r}

    E(f_{a,k}) = (1/N) * sum_i i * f_i
    """
    inv_r = 1.0 / r
    log_v = _log_vals(a, k, inv_r)
    log_total = logsumexp(log_v)

    indices = np.arange(k + 1, dtype=float)
    # E = (1/N) * sum(i * exp(log_v_i - log_total))
    # Only sum over i >= 1 (i=0 contributes nothing)
    if k == 0:
        return 0.0
    log_weighted = log_v[1:] + np.log(indices[1:])
    log_numerator = logsumexp(log_weighted)
    return np.exp(log_numerator - log_total) / N


def _find_a_k(k, N, r, eta):
    """
    Find a_k such that E(f_{a_k, k}) = eta.

    E is monotonically increasing in a (for r < 0), ranging from 0 to k/(2N).
    Returns None if no solution exists.
    """
    # Check that eta is achievable: need eta < k/(2N) approximately
    # For safety, check at extreme a values
    log_a_low = -25.0   # a ~ 1e-11
    log_a_high = 18.0    # a ~ 6e7

    def obj(log_a):
        return _expectation_linear(np.exp(log_a), k, N, r) - eta

    try:
        e_low = obj(log_a_low)
        e_high = obj(log_a_high)
    except (FloatingPointError, ValueError):
        return None

    if e_low > 0 or e_high < 0:
        return None
    if e_low * e_high > 0:
        return None

    try:
        log_a = brentq(obj, log_a_low, log_a_high, xtol=1e-12, maxiter=300)
        return np.exp(log_a)
    except (ValueError, RuntimeError):
        return None


def _tail_probability_general(a, k, N, r, t_idx, eta):
    """
    Tail probability T_t(g_{a,k}) for the general extremal distribution.

    From equation (17) in Shah & Samworth (2013):

        T_t(g_{a,k}) = 1 - ((k+1 - N*eta) * sum_{i=0}^{t_idx-1} (a+i)^{1/r})
                            / (sum_{i=0}^{k} (k+1-i) * (a+i)^{1/r})
    """
    inv_r = 1.0 / r
    log_v = _log_vals(a, k, inv_r)

    # Head sum: sum_{i=0}^{t_idx-1} (a+i)^{1/r}
    if t_idx <= 0:
        return 1.0
    if t_idx > k + 1:
        # All mass is below threshold
        # But there may be mass at k+1 > t_idx
        log_head = logsumexp(log_v)
    else:
        log_head = logsumexp(log_v[:t_idx])

    # Weighted sum: sum_{i=0}^{k} (k+1-i) * (a+i)^{1/r}
    weights = np.arange(k + 1, 0, -1, dtype=float)  # k+1, k, ..., 1
    log_weighted = log_v + np.log(weights)
    log_denom = logsumexp(log_weighted)

    # T = 1 - (k+1 - N*eta) * head / denom
    coeff = k + 1 - N * eta
    if coeff <= 0:
        return 1.0

    log_numerator = np.log(coeff) + log_head
    result = 1.0 - np.exp(log_numerator - log_denom)
    return max(0.0, min(1.0, result))


def compute_D(eta, t, N, r):
    """
    Compute D(eta, t, N, r): the maximum tail probability P(X >= t) over all
    r-concave distributions on {0, 1/N, 2/N, ..., 1} with E(X) <= eta.

    Implements the numerical algorithm from Shah & Samworth (2013), Appendix A.4.

    Parameters
    ----------
    eta : float
        Upper bound on the expected value of the distribution
    t : float
        Tail threshold (compute P(X >= t))
    N : int
        Lattice parameter: distribution is on {0, 1/N, ..., 1}
    r : float
        r-concavity parameter (e.g. -1/2 or -1/4)

    Returns
    -------
    float
        D(eta, t, N, r) — maximum tail probability
    """
    # Trivial cases
    if t <= 0:
        return 1.0
    if eta <= 0:
        return 0.0
    if eta >= t:
        return 1.0

    t_idx = int(round(N * t))
    if t_idx <= 0:
        return 1.0
    if t_idx > N:
        return 0.0

    # Step 1: find a_k for each k from t_idx to N
    a_dict = {}
    for k in range(t_idx, N + 1):
        a_k = _find_a_k(k, N, r, eta)
        if a_k is not None:
            a_dict[k] = a_k

    if not a_dict:
        # Fallback to Markov bound
        return min(eta / t, 1.0)

    # Step 2: for each k, optimise tail probability over a in [a_{k+1}, a_k]
    best_tail = 0.0
    sorted_ks = sorted(a_dict.keys())

    for idx, k in enumerate(sorted_ks):
        a_k = a_dict[k]

        # Evaluate at a_k itself (the f_{a_k,k} distribution, c=0 case)
        tail_at_ak = _tail_probability_general(a_k, k, N, r, t_idx, eta)
        best_tail = max(best_tail, tail_at_ak)

        # Optimise over [a_{k+1}, a_k] for the g_{a,k} family
        if idx + 1 < len(sorted_ks):
            a_next = a_dict[sorted_ks[idx + 1]]
            a_lo = min(a_next, a_k)
            a_hi = max(a_next, a_k)

            if a_hi - a_lo > 1e-15:
                try:
                    result = minimize_scalar(
                        lambda a: -_tail_probability_general(
                            a, k, N, r, t_idx, eta
                        ),
                        bounds=(a_lo, a_hi),
                        method='bounded',
                        options={'xatol': 1e-12, 'maxiter': 200},
                    )
                    best_tail = max(best_tail, -result.fun)
                except (ValueError, RuntimeError):
                    pass

    return min(best_tail, 1.0)


def compute_rconcave_threshold(q, p, B, pfer_target):
    """
    Find the minimum selection probability threshold tau such that the
    r-concave CPSS bound (equation 8 of Shah & Samworth 2013) satisfies
    PFER <= pfer_target.

    The bound is:

        E[V] <= min{D(theta^2, 2*tau - 1, B, -1/2),
                     D(theta, tau, 2*B, -1/4)} * p

    where theta = q / p.

    Parameters
    ----------
    q : int
        Expected number of features selected per subsample (max_candidates)
    p : int
        Total number of features after preprocessing
    B : int
        Number of complementary pairs (stability_iterations / 2)
    pfer_target : float
        Target PFER (expected number of false positives)

    Returns
    -------
    tau : float
        Minimum threshold satisfying the PFER bound
    bound_at_tau : float
        The PFER bound evaluated at the returned tau
    mb_threshold : float
        The Meinshausen & Buhlmann threshold for comparison
    """
    theta = q / p

    # M&B worst-case threshold for comparison
    mb_threshold = min(0.5 + q ** 2 / (2.0 * p * pfer_target), 1.0)

    def bound_at_tau(tau):
        """Evaluate the r-concave PFER bound at a given tau."""
        t1 = 2 * tau - 1
        # D1: via simultaneous selection statistic (Π̃_B)
        D1 = compute_D(theta ** 2, t1, B, -0.5) if t1 > 0 else 1.0
        # D2: directly on CPSS statistic (Π̂_B)
        D2 = compute_D(theta, tau, 2 * B, -0.25)
        return min(D1, D2) * p

    # Search on the natural lattice: tau in {1/(2B), 2/(2B), ..., 1}
    # Start from just above theta (below theta the bound is trivially >= p)
    step = 1.0 / (2 * B)
    start = max(int(np.ceil(theta / step)) + 1, 1)

    best_tau = 1.0
    best_bound = bound_at_tau(1.0)

    for i in range(start, 2 * B + 1):
        tau = i * step
        bound = bound_at_tau(tau)
        if bound <= pfer_target:
            best_tau = tau
            best_bound = bound
            break

    return best_tau, best_bound, mb_threshold


def print_threshold_comparison(q, p, B, pfer_target):
    """
    Compute and print a comparison of the M&B and r-concave thresholds.

    Parameters
    ----------
    q : int
        Expected number of features selected per subsample
    p : int
        Total number of features after preprocessing
    B : int
        Number of complementary pairs
    pfer_target : float
        Target PFER
    """
    theta = q / p
    tau_rc, bound_rc, tau_mb = compute_rconcave_threshold(q, p, B, pfer_target)

    print(f"\n    Shah & Samworth (2013) threshold comparison")
    print(f"    {'─' * 50}")
    print(f"    θ = q/p = {q}/{p} = {theta:.4f}")
    print(f"    B = {B} complementary pairs ({2 * B} subsamples)")
    print(f"    PFER target = {pfer_target:.2f}")
    print(f"    {'─' * 50}")
    print(f"    M&B worst-case threshold:  π = {tau_mb:.4f}")
    print(f"    S&S r-concave threshold:   π = {tau_rc:.4f}")
    print(f"    Threshold reduction:       Δπ = {tau_mb - tau_rc:.4f}")
    print(f"    PFER bound at r-concave τ: {bound_rc:.3f}")
    print(f"    {'─' * 50}")
