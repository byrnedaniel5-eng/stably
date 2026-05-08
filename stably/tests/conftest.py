"""Pytest fixtures and a synthetic-data builder for the package's tests.

The tests in this directory must run without any proprietary data, raw files,
or licensed databases (HC-TEST-04). All fixtures here construct synthetic data
in memory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (run with '-m slow' or '--runslow')"
    )


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def make_synthetic_pg_matrix(
    n_samples: int = 60,
    n_proteins: int = 200,
    n_signal: int = 5,
    effect_size: float = 1.5,
    missing_rate: float = 0.05,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a synthetic DIA-NN-style pg_matrix and matching label dataframe.

    Returns
    -------
    pg_matrix : pd.DataFrame
        Six metadata columns (Protein.Group, Protein.Ids, Protein.Names,
        Genes, First.Protein.Description, N.Proteotypic.Sequences) followed
        by sample-named abundance columns. Sample IDs are 'S001'..'S{n}'.
    labels : pd.DataFrame
        Two columns: 'Biobank Number' (sample IDs) and 'Group'
        (alternating 'Case' / 'Control').
    """
    rng = np.random.default_rng(seed)

    sample_ids = [f"S{i:03d}" for i in range(1, n_samples + 1)]
    groups = np.array(['Case' if i % 2 == 0 else 'Control' for i in range(n_samples)])
    y_indicator = (groups == 'Case').astype(float)

    # Log-scale baseline abundances; a few features get a true case-vs-control shift.
    X = rng.normal(loc=20.0, scale=1.0, size=(n_proteins, n_samples))
    for j in range(n_signal):
        X[j, :] += effect_size * y_indicator
    # Convert log-scale to intensity-scale so the pipeline's log_transform path
    # exercises the realistic input shape.
    X_intensity = np.exp2(X)

    if missing_rate > 0:
        missing_mask = rng.random(X_intensity.shape) < missing_rate
        X_intensity[missing_mask] = np.nan

    metadata_cols = pd.DataFrame({
        'Protein.Group': [f"P{idx:05d}" for idx in range(n_proteins)],
        'Protein.Ids': [f"P{idx:05d}" for idx in range(n_proteins)],
        'Protein.Names': [f"PROT{idx}_HUMAN" for idx in range(n_proteins)],
        'Genes': [f"GENE{idx}" for idx in range(n_proteins)],
        'First.Protein.Description': [f"Synthetic protein {idx}" for idx in range(n_proteins)],
        'N.Proteotypic.Sequences': rng.integers(2, 12, size=n_proteins),
    })

    sample_frame = pd.DataFrame(X_intensity, columns=sample_ids)
    pg_matrix = pd.concat([metadata_cols, sample_frame], axis=1)

    labels = pd.DataFrame({
        'Biobank Number': sample_ids,
        'Group': groups,
    })

    return pg_matrix, labels


def make_synthetic_pr_matrix(n_samples: int = 20, n_peptides: int = 40, seed: int = 0) -> pd.DataFrame:
    """Synthetic peptide-level pg_matrix-shaped DataFrame to exercise the
    rejection path in load_data."""
    rng = np.random.default_rng(seed)
    sample_ids = [f"S{i:03d}" for i in range(1, n_samples + 1)]
    metadata = pd.DataFrame({
        'Protein.Group': [f"P{idx // 4:05d}" for idx in range(n_peptides)],
        'Protein.Ids': [f"P{idx // 4:05d}" for idx in range(n_peptides)],
        'Protein.Names': [f"PROT{idx // 4}_HUMAN" for idx in range(n_peptides)],
        'Genes': [f"GENE{idx // 4}" for idx in range(n_peptides)],
        'First.Protein.Description': ['x'] * n_peptides,
        'N.Proteotypic.Sequences': rng.integers(2, 6, size=n_peptides),
        'Proteotypic': [1] * n_peptides,
        'Stripped.Sequence': [f"PEPTIDE{idx}" for idx in range(n_peptides)],
        'Modified.Sequence': [f"PEPTIDE{idx}" for idx in range(n_peptides)],
        'Precursor.Charge': rng.integers(2, 4, size=n_peptides),
    })
    intensity = pd.DataFrame(
        rng.normal(20, 1, size=(n_peptides, n_samples)), columns=sample_ids
    )
    return pd.concat([metadata, intensity], axis=1)


@pytest.fixture
def synthetic_pg_data():
    """Default-shape synthetic pg_matrix + labels."""
    return make_synthetic_pg_matrix()


@pytest.fixture
def base_config_dict(tmp_path):
    """A minimal Config-shaped dict; tests can override fields as needed."""
    return {
        'data_file': str(tmp_path / 'data.csv'),
        'label_file': str(tmp_path / 'labels.xlsx'),
        'case_name': 'Case',
        'control_name': 'Control',
        'group_column': 'Group',
        'sample_id_column': 'Biobank Number',
        'random_state': 0,
        'max_missing': 0.3,
        'log_transform': True,
        'imputation_strategy': 'minimum',
        'min_proteotypic_peptides': 0,
        'knn_neighbors': 5,
        'max_iterations': 2000,
        'l1_ratio': 0.5,
        'max_candidates': 20,
        'stability_iterations': 20,
        'stability_pfer': 5.0,
        'n_jobs': 1,
        'output_dir': str(tmp_path / 'out'),
    }
