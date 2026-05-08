"""Empirical verification of the package's central scientific claim:
the expected number of falsely-selected features stays within the PFER bound
when the data has no signal.

Marked `@pytest.mark.slow` so it can be opt-in via:
    pytest --runslow stably/tests/test_pfer_empirical.py

This is the HC-TEST-02 analogue at the selection-FDR layer (the package does
not estimate FDR for proteins; it estimates PFER for selected features).
"""

import numpy as np
import pandas as pd
import pytest

from stably.config import Config
from stably.feature_selection import stability_selection_elasticnet


def _make_null_data(n_samples, n_features, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_samples, n_features))
    y = rng.integers(0, 2, size=n_samples)
    if y.sum() < 5 or (n_samples - y.sum()) < 5:
        # Force a balanced split to avoid degenerate iterations
        y = np.array([0, 1] * (n_samples // 2) + [0] * (n_samples % 2))
    feature_names = [f"f{i:04d}" for i in range(n_features)]
    return pd.DataFrame(X, columns=feature_names), y


@pytest.mark.slow
def test_pfer_empirical_within_bound(base_config_dict):
    """On null data (no signal), the empirical mean count of stable features
    across replicates should be within the PFER budget."""
    pfer_target = 5.0
    n_replicates = 8
    n_samples = 80
    n_features = 200

    cfg_dict = dict(base_config_dict)
    cfg_dict.update({
        'max_candidates': 15,
        'stability_iterations': 30,
        'stability_pfer': pfer_target,
        'l1_ratio': 0.5,
        'imputation_strategy': 'minimum',
        'log_transform': False,
        'max_iterations': 2000,
        'n_jobs': 1,
    })
    config = Config(cfg_dict)

    selected_counts = []
    for replicate in range(n_replicates):
        X, y = _make_null_data(n_samples, n_features, seed=1000 + replicate)
        cfg_dict['random_state'] = 1000 + replicate
        config = Config(cfg_dict)
        stable, _, _, _, _ = stability_selection_elasticnet(X, y, config)
        selected_counts.append(len(stable))

    mean_count = float(np.mean(selected_counts))
    # Allow a generous tolerance — PFER is an expectation bound, and 8
    # replicates have substantial Monte Carlo variance.
    assert mean_count <= pfer_target * 2, (
        f"Mean stable-feature count on null data ({mean_count}) exceeds 2x PFER target "
        f"({pfer_target}); this indicates the bound is not being honoured. Per-replicate: {selected_counts}"
    )


def test_pfer_empirical_smoke(base_config_dict):
    """Fast version of the empirical test: a single replicate with reduced
    iterations, asserting only that the call returns sensibly and doesn't
    select an absurd number of features."""
    cfg_dict = dict(base_config_dict)
    cfg_dict.update({
        'max_candidates': 10,
        'stability_iterations': 10,
        'stability_pfer': 5.0,
        'imputation_strategy': 'minimum',
        'log_transform': False,
        'max_iterations': 1000,
        'n_jobs': 1,
        'random_state': 7,
    })
    config = Config(cfg_dict)
    X, y = _make_null_data(60, 100, seed=7)
    stable, threshold, sel_probs, c_ref, info = stability_selection_elasticnet(X, y, config)

    assert isinstance(stable, list)
    assert 0.0 < threshold <= 1.0
    assert c_ref > 0.0
    assert info['pfer_target'] == 5.0
    # On a 60x100 null with the small budget, we should not be selecting more
    # than a small handful of features
    assert len(stable) <= 20
