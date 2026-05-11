"""Per-subsample local preprocessing (S&S 2013 leakage fix).

The Shah & Samworth bound on E(V) requires complementary-pairs subsamples
to be (conditionally) independent. Fitting the imputer and scaler on the
full cohort before subsampling introduces cross-subsample dependence that
violates the bound. These tests lock in the design fix:

    1. _stability_iteration_elasticnet_path tolerates NaN-bearing input
       (i.e. it does NOT assume the caller has already imputed).
    2. Perturbing samples *outside* a given iteration's subsample does not
       change that iteration's selected features. This is what proves the
       local stage is refit per subsample, not inherited from the cohort.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from stably.config import Config
from stably.feature_selection import (
    _stability_iteration_elasticnet_path,
    stability_selection_elasticnet,
)


def _make_signal_data(n_samples=40, n_features=20, n_signal=4, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    X = rng.normal(10.0, 1.0, size=(n_samples, n_features))
    # Inject a real case-vs-control shift on the first n_signal features
    X[:, :n_signal] += 1.5 * y[:, None]
    # A handful of NaNs to exercise the per-subsample imputation path
    nan_mask = rng.random(X.shape) < 0.05
    X[nan_mask] = np.nan
    return pd.DataFrame(X, columns=[f"f{j:03d}" for j in range(n_features)]), y


def test_iteration_accepts_nan_bearing_input(base_config_dict):
    """The iteration must not crash on NaN-bearing X (i.e. imputation
    happens inside the iteration, not assumed upstream)."""
    cfg_dict = dict(base_config_dict)
    cfg_dict.update({
        'imputation_strategy': 'minimum',
        'log_transform': False,
        'l1_ratio': 0.5,
        'max_iterations': 500,
        'random_state': 0,
    })
    config = Config(cfg_dict)

    X, y = _make_signal_data(seed=0)
    assert X.isna().any().any(), "Test data must contain NaNs to be meaningful"

    selected, converged = _stability_iteration_elasticnet_path(
        X, y, q=5, iteration_idx=0, random_seed=0,
        C_ref=0.1, l1_ratio=0.5, max_iter=500, config=config,
    )
    assert isinstance(selected, list)
    assert converged in (True, False)


def test_subsample_selection_insensitive_to_out_of_subsample_values(base_config_dict):
    """Perturbations to samples NOT in iteration 0's subsample must not
    change iteration 0's selected features. This is the direct fingerprint
    of per-subsample local preprocessing — if imputer/scaler were fit on
    the full cohort, out-of-subsample outliers would distort the
    transformed values inside the subsample."""
    cfg_dict = dict(base_config_dict)
    cfg_dict.update({
        'imputation_strategy': 'minimum',
        'log_transform': False,
        'l1_ratio': 0.5,
        'max_iterations': 1000,
        'random_state': 0,
    })
    config = Config(cfg_dict)

    X, y = _make_signal_data(seed=1)

    # Replicate iteration 0's split so we know which rows are out-of-subsample
    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=0.5, random_state=config.RANDOM_STATE + 0,
    )
    train_idx, _ = next(splitter.split(X, y))
    sub_mask = np.zeros(len(X), dtype=bool)
    sub_mask[train_idx] = True
    out_of_sub_idx = np.where(~sub_mask)[0]

    selected_a, _ = _stability_iteration_elasticnet_path(
        X, y, q=5, iteration_idx=0, random_seed=config.RANDOM_STATE,
        C_ref=0.1, l1_ratio=0.5, max_iter=1000, config=config,
    )

    # Inject large outliers into rows that iteration 0 will never see.
    X_perturbed = X.copy()
    X_perturbed.iloc[out_of_sub_idx, :] = (
        X_perturbed.iloc[out_of_sub_idx, :] * 1000.0 + 1e6
    )

    selected_b, _ = _stability_iteration_elasticnet_path(
        X_perturbed, y, q=5, iteration_idx=0, random_seed=config.RANDOM_STATE,
        C_ref=0.1, l1_ratio=0.5, max_iter=1000, config=config,
    )

    assert selected_a == selected_b, (
        "Iteration 0's selection changed when out-of-subsample rows were "
        "perturbed — local preprocessing is leaking cohort-wide statistics "
        "into the subsample."
    )


def test_full_pipeline_runs_on_nan_input(base_config_dict):
    """End-to-end smoke: stability_selection_elasticnet on globally-
    preprocessed (NaN-bearing) X completes and produces a list."""
    cfg_dict = dict(base_config_dict)
    cfg_dict.update({
        'imputation_strategy': 'minimum',
        'log_transform': False,
        'max_candidates': 6,
        'stability_iterations': 12,
        'stability_pfer': 5.0,
        'l1_ratio': 0.5,
        'max_iterations': 1000,
        'n_jobs': 1,
        'random_state': 0,
    })
    config = Config(cfg_dict)

    X, y = _make_signal_data(n_samples=60, n_features=30, n_signal=6, seed=2)
    stable, threshold, sel_probs, c_ref, info = stability_selection_elasticnet(
        X, y, config
    )
    assert isinstance(stable, list)
    assert 0.0 < threshold <= 1.0
    assert c_ref > 0.0
