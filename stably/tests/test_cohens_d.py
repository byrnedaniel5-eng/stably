"""Tests for calculate_cohens_d's numerical correctness and metadata fields."""

import math

import numpy as np
import pandas as pd
import pytest

from stably.feature_selection import calculate_cohens_d


def test_cohens_d_matches_analytical_value():
    """Construct features with known means and pooled standard deviations,
    verify the computed Cohen's d matches the closed-form value."""
    n_per_group = 50
    rng = np.random.default_rng(0)
    control = rng.normal(0, 1, size=n_per_group)
    case = rng.normal(0.8, 1, size=n_per_group)

    df = pd.DataFrame({'protein_A': np.concatenate([control, case])})
    y = np.concatenate([np.zeros(n_per_group), np.ones(n_per_group)])

    result = calculate_cohens_d(df, y, ['protein_A'])
    entry = result['protein_A']

    # Analytical Cohen's d using the same pooled-std formula
    n0, n1 = n_per_group, n_per_group
    s0 = control.std(ddof=1)
    s1 = case.std(ddof=1)
    pooled = math.sqrt(((n0 - 1) * s0 ** 2 + (n1 - 1) * s1 ** 2) / (n0 + n1 - 2))
    expected = (case.mean() - control.mean()) / pooled

    assert entry['cohens_d'] == pytest.approx(expected, abs=1e-10)
    assert entry['n_case'] == n_per_group
    assert entry['n_control'] == n_per_group
    assert entry['direction'] == 'up_case'


def test_cohens_d_negative_direction_when_control_higher():
    df = pd.DataFrame({'p1': [10, 11, 12, 13, 14, 1, 2, 3, 4, 5]})
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    result = calculate_cohens_d(df, y, ['p1'])
    assert result['p1']['direction'] == 'up_control'
    assert result['p1']['cohens_d'] < 0


def test_cohens_d_zero_when_means_equal():
    df = pd.DataFrame({'p1': [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]})
    y = np.array([0, 0, 0, 1, 1, 1])
    result = calculate_cohens_d(df, y, ['p1'])
    assert result['p1']['cohens_d'] == pytest.approx(0.0, abs=1e-12)
    assert result['p1']['direction'] == 'zero'


def test_cohens_d_undetermined_when_group_too_small():
    df = pd.DataFrame({'p1': [1.0, 2.0, 3.0, 4.0]})
    y = np.array([0, 0, 0, 1])  # only 1 sample in case
    result = calculate_cohens_d(df, y, ['p1'])
    assert result['p1']['direction'] == 'undetermined'
    assert math.isnan(result['p1']['cohens_d'])
    assert result['p1']['n_case'] == 1
    assert result['p1']['n_control'] == 3


def test_cohens_d_handles_missing_values():
    """NaNs are dropped per-feature; n_case / n_control reflect non-NaN counts."""
    df = pd.DataFrame({'p1': [1.0, 2.0, np.nan, np.nan, 5.0, 6.0, 7.0]})
    y = np.array([0, 0, 0, 1, 1, 1, 1])
    result = calculate_cohens_d(df, y, ['p1'])
    assert result['p1']['n_control'] == 2  # NaN dropped
    assert result['p1']['n_case'] == 3  # NaN dropped


def test_cohens_d_zero_pooled_std_returns_zero():
    """When both groups have zero variance, cohens_d falls back to 0."""
    df = pd.DataFrame({'p1': [5.0, 5.0, 5.0, 5.0]})
    y = np.array([0, 0, 1, 1])
    result = calculate_cohens_d(df, y, ['p1'])
    assert result['p1']['cohens_d'] == 0.0
    assert result['p1']['direction'] == 'zero'
