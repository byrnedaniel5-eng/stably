"""Feature selection using the true Shah & Samworth (2013) r-concave bound.

Uses the r-concave PFER bound (equation 8) instead of the Meinshausen &
Buhlmann worst-case bound for threshold computation.
"""

import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel, delayed
from typing import List, Dict, Tuple

from .preprocessing import Preprocessor
from .rconcave import compute_rconcave_threshold, print_threshold_comparison


# Design notes
# ------------
# 1. C_ref calibration. C_ref is calibrated once on the FULL globally-
#    preprocessed training set (with a one-shot local stage fit) and reused
#    across every complementary-pairs subsample. This is the Shah & Samworth
#    (2013) design — regularisation is treated as a fixed hyperparameter of
#    the stability procedure, not something re-tuned per subsample. It is
#    not a cross-validation leakage bug: stability selection does not
#    evaluate model performance on held-out data; it counts how often each
#    feature is selected across subsamples at a fixed regularisation
#    strength.
# 2. Per-subsample local preprocessing. Imputation and scaling ARE refit
#    inside each subsample. Fitting them globally would let one subsample's
#    imputed values depend on rows in the complementary subsample, breaking
#    the independence-across-pairs assumption the S&S bound relies on. The
#    cohort-wide missing-value filter and per-sample log transform are
#    applied once (globally) because they do not borrow information across
#    rows in a way that violates the bound.


def _sklearn_ge(major: int, minor: int) -> bool:
    """True if the installed scikit-learn is at least the given version."""
    parts = []
    for piece in sklearn.__version__.split(".")[:2]:
        digits = "".join(c for c in piece if c.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 2:
        parts.append(0)
    return tuple(parts) >= (major, minor)


# scikit-learn 1.8 deprecated passing ``penalty`` to LogisticRegression and
# infers elastic net from a float ``l1_ratio`` instead; 1.10 removes it.
#
# The argument cannot simply be dropped for all versions: before 1.8, omitting
# ``penalty`` defaults to 'l2' and ``l1_ratio`` is IGNORED, which would quietly
# turn every fit in this package into ridge regression — a silent change to the
# selection behaviour rather than a visible failure. So it is passed only where
# it is still required.
_PENALTY_KWARG_REQUIRED = not _sklearn_ge(1, 8)


def _elasticnet_logreg(C, l1_ratio, random_state, max_iter):
    """LogisticRegression configured for elastic net, across sklearn versions."""
    kwargs = dict(
        C=C,
        l1_ratio=l1_ratio,
        solver='saga',
        random_state=random_state,
        max_iter=max_iter,
    )
    if _PENALTY_KWARG_REQUIRED:
        kwargs['penalty'] = 'elasticnet'
    return LogisticRegression(**kwargs)


def _find_elasticnet_C_for_q(X, y, q, random_state, l1_ratio, max_iter=1000):
    """Largest ElasticNet C (in a log-spaced sweep) that keeps <= q non-zero coefs."""
    Cs = np.logspace(-4, 2, 50)
    C_ref = Cs[0]
    for C in Cs:
        model = _elasticnet_logreg(C, l1_ratio, random_state, max_iter)
        model.fit(X, y)
        n_nonzero = int(np.sum(model.coef_[0] != 0))
        if n_nonzero <= q:
            C_ref = C
    return C_ref


def _stability_iteration_elasticnet_path(
    X_global, y, q, iteration_idx, random_seed, C_ref, l1_ratio, max_iter, config
):
    """Single complementary-pairs iteration with per-subsample local
    preprocessing (imputation + scaling refit on the subsample only).

    ``X_global`` is expected to be the post-missing-filter, post-log matrix
    with NaNs preserved.

    Returns (selected_features, converged).
    """
    # Complementary pairs: (2i, 2i+1) share the split seed
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=0.5,
        random_state=random_seed + (iteration_idx // 2),
    )
    train_indices, test_indices = next(splitter.split(X_global, y))
    subsample_indices = train_indices if iteration_idx % 2 == 0 else test_indices

    X_sub_raw = X_global.iloc[subsample_indices]
    y_sub = y[subsample_indices]

    # Fit a fresh local preprocessor (imputer + scaler) on this subsample.
    sub_pp = Preprocessor(config, verbose=False)
    X_sub = sub_pp.fit_transform_local(X_sub_raw)

    model = _elasticnet_logreg(
        C_ref, l1_ratio, random_seed + iteration_idx, max_iter
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X_sub, y_sub)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)

    coefs = np.abs(model.coef_[0])
    nonzero_idx = np.where(coefs > 1e-6)[0]
    if len(nonzero_idx) == 0:
        return [], converged
    top_q_idx = nonzero_idx[np.argsort(coefs[nonzero_idx])[::-1][:q]]
    return X_sub.columns[top_q_idx].tolist(), converged


def stability_selection_elasticnet(
    X, y, config, q=None, n_iterations=None, n_jobs=None
) -> Tuple[List[str], float, Dict[str, float], float, dict]:
    """
    Shah & Samworth (2013) stability selection with the true r-concave bound.

    ``X`` is the *globally* preprocessed feature matrix: missing-value filter
    and (optional) log transform already applied, but imputation and scaling
    NOT yet applied (NaNs are preserved). Imputation and scaling are refit
    inside each complementary-pairs subsample to preserve the bound's
    independence assumption. If ``X`` contains no NaNs (e.g. callers that
    handled imputation upstream), the local stage simply re-scales each
    subsample, which is harmless.

    Returns
    -------
    stable_features : list of str
    threshold : float (r-concave PFER-derived selection probability)
    selection_probs : dict[str, float]
    C_ref : float
    threshold_info : dict with rconcave/MB comparison and convergence stats
    """
    q = q or config.MAX_CANDIDATES
    n_iterations = n_iterations or config.STABILITY_ITERATIONS
    n_jobs = n_jobs or config.N_JOBS
    l1_ratio = config.L1_RATIO

    p = X.shape[1]
    B = n_iterations // 2
    pfer = config.STABILITY_PFER

    # Compute the r-concave threshold (true S&S bound)
    print(f"    Computing r-concave threshold (q={q}, p={p}, B={B}, PFER={pfer:.1f})...")
    threshold, bound_at_tau, mb_threshold = compute_rconcave_threshold(q, p, B, pfer)
    threshold = min(threshold, 1.0)

    print_threshold_comparison(q, p, B, pfer)

    threshold_info = {
        'rconcave_threshold': threshold,
        'mb_threshold': mb_threshold,
        'threshold_reduction': mb_threshold - threshold,
        'pfer_bound': bound_at_tau,
        'pfer_target': pfer,
        'theta': q / p,
        'B': B,
    }

    # Calibrate C on a full-cohort local-stage fit. This is a hyperparameter
    # selection step (see design note 1); the per-iteration fits below use
    # subsample-only local stages.
    print(f"    Calibrating ElasticNet regularisation from path (l1_ratio={l1_ratio})...")
    cref_pp = Preprocessor(config, verbose=False)
    X_for_cref = cref_pp.fit_transform_local(X)
    C_ref = _find_elasticnet_C_for_q(
        X_for_cref, y, q, config.RANDOM_STATE, l1_ratio, config.MAX_ITERATIONS
    )
    print(f"    C_ref={C_ref:.2e}  |  r-concave threshold: π={threshold:.4f}  |  PFER≤{pfer:.1f}")
    print(f"    Running {n_iterations} complementary-pairs iterations (n_jobs={n_jobs})...")

    iteration_results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_stability_iteration_elasticnet_path)(
            X, y, q, i, config.RANDOM_STATE, C_ref, l1_ratio, config.MAX_ITERATIONS, config
        )
        for i in range(n_iterations)
    )

    all_selected = [features for features, _ in iteration_results]
    n_non_converged = sum(1 for _, converged in iteration_results if not converged)
    threshold_info['n_iterations'] = n_iterations
    threshold_info['n_non_converged'] = n_non_converged
    threshold_info['non_convergence_rate'] = n_non_converged / n_iterations

    if n_non_converged / n_iterations > 0.05:
        print(
            f"    WARNING: {n_non_converged}/{n_iterations} "
            f"({100 * n_non_converged / n_iterations:.1f}%) iterations did not converge. "
            f"Consider increasing config.MAX_ITERATIONS."
        )
    elif n_non_converged > 0:
        print(f"    Convergence: {n_iterations - n_non_converged}/{n_iterations} iterations")

    # Empirical selection probabilities
    feature_counts: Dict[str, int] = {}
    for selected in all_selected:
        for feature in selected:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1

    selection_probs = {f: c / n_iterations for f, c in feature_counts.items()}

    stable_features = [f for f, prob in selection_probs.items() if prob >= threshold]

    print(f"    Completed: {len(stable_features)} stable features (π̂ ≥ {threshold:.4f})")
    return stable_features, threshold, selection_probs, C_ref, threshold_info


def calculate_cohens_d(X, y, features):
    """
    Cohen's d and per-group sample sizes for each feature between case and control.

    Returns
    -------
    dict[str, dict]
        Per-feature entries with keys:
            cohens_d   — signed (group1 - group0) / pooled_std
            n_case     — number of non-NaN samples with y == 1
            n_control  — number of non-NaN samples with y == 0
            direction  — 'up_case', 'up_control', or 'zero'
    """
    results = {}
    for feature in features:
        group0 = X.loc[y == 0, feature].dropna()
        group1 = X.loc[y == 1, feature].dropna()
        n0, n1 = len(group0), len(group1)

        if n0 < 2 or n1 < 2:
            results[feature] = {
                'cohens_d': float('nan'),
                'n_case': n1,
                'n_control': n0,
                'direction': 'undetermined',
            }
            continue

        pooled_std = np.sqrt(
            ((n0 - 1) * group0.std() ** 2 + (n1 - 1) * group1.std() ** 2) / (n0 + n1 - 2)
        )
        if pooled_std > 0:
            d = (group1.mean() - group0.mean()) / pooled_std
        else:
            d = 0.0

        if d > 0:
            direction = 'up_case'
        elif d < 0:
            direction = 'up_control'
        else:
            direction = 'zero'

        results[feature] = {
            'cohens_d': float(d),
            'n_case': n1,
            'n_control': n0,
            'direction': direction,
        }
    return results
