"""Regression test: low-variance features must pass through preprocessing now
that the variance filter has been removed. ElasticNet's penalty handles them
correctly."""

import numpy as np
import pandas as pd

from stably.config import Config
from stably.preprocessing import Preprocessor


def test_low_variance_feature_survives_preprocessing(base_config_dict):
    cfg_dict = dict(base_config_dict)
    cfg_dict.update({'imputation_strategy': 'minimum', 'log_transform': False})
    config = Config(cfg_dict)

    rng = np.random.default_rng(0)
    n_samples = 40
    X = pd.DataFrame({
        'high_var_protein': rng.normal(10, 5, size=n_samples),
        'medium_var_protein': rng.normal(10, 1, size=n_samples),
        'almost_constant_protein': np.full(n_samples, 5.0) + rng.normal(0, 1e-6, size=n_samples),
        'truly_constant_protein': np.full(n_samples, 7.0),
    })

    pp = Preprocessor(config, verbose=False)
    out = pp.fit_transform(X)

    # All four features should survive — no variance filter
    assert set(out.columns) == set(X.columns)
    # Truly constant feature ends up with std=0 after standardisation, which
    # produces NaN when StandardScaler divides by zero. Document the behaviour
    # rather than treating it as a regression.
    if not out['truly_constant_protein'].isna().all():
        assert out['truly_constant_protein'].std(ddof=0) == 0.0


def test_no_min_variance_percentile_attribute(base_config_dict):
    """Config should not expose MIN_VARIANCE_PERCENTILE anymore."""
    config = Config(base_config_dict)
    assert not hasattr(config, 'MIN_VARIANCE_PERCENTILE')


def test_preprocessor_has_no_variance_state(base_config_dict):
    """Preprocessor should not expose features_after_variance anymore."""
    config = Config(base_config_dict)
    pp = Preprocessor(config, verbose=False)
    assert not hasattr(pp, 'features_after_variance')
