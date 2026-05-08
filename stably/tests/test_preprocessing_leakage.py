"""Verify that Preprocessor.fit_transform / transform do not leak test-set
statistics back into training.

The pipeline is intended to be used with nested CV in a follow-up project,
so guarding against leakage now (even though full_dataset_stability_elasticnet
operates on the entire dataset) is worthwhile.
"""

import numpy as np
import pandas as pd

from stably.config import Config
from stably.preprocessing import Preprocessor


def _make_data_with_known_stats(n_samples, n_features, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=10.0, scale=2.0, size=(n_samples, n_features))
    # Insert NaNs at known positions
    for j in range(n_features):
        if j % 5 == 0:
            X[0, j] = np.nan
    df = pd.DataFrame(
        X, columns=[f"f{j}" for j in range(n_features)],
        index=[f"S{i:03d}" for i in range(n_samples)],
    )
    return df


def test_scaler_uses_train_only_statistics(base_config_dict):
    cfg_dict = dict(base_config_dict)
    cfg_dict.update({'imputation_strategy': 'minimum', 'log_transform': False})
    config = Config(cfg_dict)

    X = _make_data_with_known_stats(60, 30)
    X_train = X.iloc[:40]
    X_test = X.iloc[40:].copy()
    # Inject extreme values into test set; if leakage exists, the train
    # scaler would somehow reflect these.
    X_test.iloc[0, 0] = 1e6

    pp = Preprocessor(config, verbose=False)
    pp.fit_transform(X_train)
    train_scaler_mean = pp.scaler.mean_.copy()
    train_scaler_var = pp.scaler.var_.copy()

    pp.transform(X_test)
    np.testing.assert_array_equal(pp.scaler.mean_, train_scaler_mean)
    np.testing.assert_array_equal(pp.scaler.var_, train_scaler_var)


def test_imputer_uses_train_only_statistics(base_config_dict):
    cfg_dict = dict(base_config_dict)
    cfg_dict.update({'imputation_strategy': 'minimum', 'log_transform': False})
    config = Config(cfg_dict)

    X = _make_data_with_known_stats(60, 30)
    X_train = X.iloc[:40]
    X_test = X.iloc[40:].copy()
    X_test.iloc[1, 1] = -999.0  # extreme value in test

    pp = Preprocessor(config, verbose=False)
    pp.fit_transform(X_train)
    train_min = pp.min_values.copy()

    pp.transform(X_test)
    pd.testing.assert_series_equal(pp.min_values, train_min)


def test_features_after_missing_filter_locked_at_fit(base_config_dict):
    """The set of surviving features is decided at fit_transform; transform
    must not re-evaluate it on the test set."""
    cfg_dict = dict(base_config_dict)
    cfg_dict.update({'max_missing': 0.2, 'imputation_strategy': 'minimum', 'log_transform': False})
    config = Config(cfg_dict)

    rng = np.random.default_rng(0)
    X_train = pd.DataFrame(
        rng.normal(10, 2, size=(40, 20)),
        columns=[f"f{j}" for j in range(20)],
    )
    # Make f0 mostly missing in train (should be filtered out)
    X_train.iloc[:35, 0] = np.nan

    X_test = pd.DataFrame(
        rng.normal(10, 2, size=(20, 20)),
        columns=[f"f{j}" for j in range(20)],
    )
    # f0 fully observed in test — but it should still be dropped because the
    # filter was set during fit.

    pp = Preprocessor(config, verbose=False)
    pp.fit_transform(X_train)
    assert "f0" not in pp.features_after_missing

    out = pp.transform(X_test)
    assert "f0" not in out.columns
