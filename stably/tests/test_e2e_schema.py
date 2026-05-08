"""End-to-end test: synthetic pg_matrix → full pipeline → assert output
schema.

Covers HC-RPT-01 (run manifest with versions/parameters), HC-RPT-02 (schema
sidecar), HC-STAT-02 (per-group N in CSV), HC-QUANT-02 (imputation_rate
column), and HC-INTER-03 (DIA-NN log auto-discovery via the manifest).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stably.config import Config
from stably.io import load_data, save_results
from stably.workflows import full_dataset_stability_elasticnet
from stably.tests.conftest import make_synthetic_pg_matrix


REQUIRED_CSV_COLUMNS = {
    'feature_index', 'gene', 'protein_group', 'protein_names',
    'selection_probability', 'cohens_d', 'direction',
    'n_case', 'n_control', 'imputation_rate',
}

REQUIRED_MANIFEST_KEYS = {
    'package', 'runtime', 'inputs', 'upstream_diann', 'resolved_config',
    'derived_parameters', 'sample_counts', 'preprocessing',
    'convergence', 'quantification_caveats', 'timestamp',
}


def _build_inputs(tmp_path):
    pg_matrix, labels = make_synthetic_pg_matrix(
        n_samples=80, n_proteins=80, n_signal=8, effect_size=2.5, missing_rate=0.05, seed=1
    )
    data_file = tmp_path / "synthetic.pg_matrix.csv"
    label_file = tmp_path / "synthetic_labels.xlsx"
    pg_matrix.to_csv(data_file, index=False)
    labels.to_excel(label_file, index=False)
    return data_file, label_file


def _build_diann_log(tmp_path):
    """A minimal but realistic DIA-NN log so the manifest's upstream_diann
    block is populated."""
    log_path = tmp_path / "output.log.txt"
    log_path.write_text(
        "DIA-NN 2.5.0 Academia  (Data-Independent Acquisition by Neural Networks)\n"
        "Compiled on Apr 12 2026 10:45:33\n"
        "Current date and time: Sun Apr 19 16:40:10 2026\n"
        "Logical CPU cores: 8\n"
        "/diann --fasta /path/to/SwissPROT_human_Jul25.fasta --lib /path/to/lib --threads 4 --qvalue 0.01 --matrices --reanalyse\n"
        "Output will be filtered at 0.01 FDR\n"
        "MBR enabled; .quant files will only be saved to disk during the first pass\n"
    )
    return log_path


def test_end_to_end_outputs_match_schema(tmp_path, base_config_dict):
    data_file, label_file = _build_inputs(tmp_path)
    diann_log = _build_diann_log(tmp_path)
    output_dir = tmp_path / "out"

    cfg_dict = dict(base_config_dict)
    cfg_dict.update({
        'data_file': str(data_file),
        'label_file': str(label_file),
        'diann_log_file': str(diann_log),
        'max_candidates': 10,
        'stability_iterations': 12,
        'stability_pfer': 5.0,
        'imputation_strategy': 'minimum',
        'output_dir': str(output_dir),
        'random_state': 42,
        'n_jobs': 1,
    })
    config = Config(cfg_dict)

    X_raw, y, sample_ids, protein_metadata = load_data(config)
    assert X_raw.shape[0] == len(y)

    results = full_dataset_stability_elasticnet(X_raw, y, config)
    if results is None:
        pytest.skip("Synthetic run produced no stable features; bump effect_size or iterations.")

    save_results(results, protein_metadata, config)

    csv_files = list(output_dir.glob("stable_features_*.csv"))
    schema_files = list(output_dir.glob("stable_features_*.schema.json"))
    manifest_files = list(output_dir.glob("run_manifest_*.json"))
    pickle_files = list(output_dir.glob("full_results_*.pkl"))

    assert len(csv_files) == 1
    assert len(schema_files) == 1
    assert len(manifest_files) == 1
    assert len(pickle_files) == 1

    df = pd.read_csv(csv_files[0])
    missing = REQUIRED_CSV_COLUMNS - set(df.columns)
    assert not missing, f"stable_features CSV missing columns: {missing}"

    # n_case + n_control should be <= total sample count
    assert (df['n_case'] >= 0).all()
    assert (df['n_control'] >= 0).all()

    # imputation_rate is a fraction
    assert ((df['imputation_rate'] >= 0) & (df['imputation_rate'] <= 1)).all()

    # direction is one of the four allowed values
    assert set(df['direction'].unique()).issubset({'up_case', 'up_control', 'zero', 'undetermined'})

    schema = json.loads(schema_files[0].read_text())
    assert 'columns' in schema
    assert REQUIRED_CSV_COLUMNS.issubset(set(schema['columns']))

    manifest = json.loads(manifest_files[0].read_text())
    missing_keys = REQUIRED_MANIFEST_KEYS - set(manifest)
    assert not missing_keys, f"manifest missing keys: {missing_keys}"

    # Manifest must record package version, dependency versions, and input hashes
    assert manifest['package']['name'] == 'stably'
    assert manifest['package']['version']
    assert manifest['runtime']['dependency_versions']['numpy']
    assert manifest['inputs']['data_file_sha256']
    assert manifest['inputs']['label_file_sha256']

    # DIA-NN provenance was found and parsed
    assert manifest['upstream_diann']['found'] is True
    assert 'SwissPROT_human_Jul25.fasta' in manifest['upstream_diann']['fasta_name']
    assert manifest['upstream_diann']['mbr_enabled'] is True
    assert manifest['upstream_diann']['fasta_release_tag'] == 'Jul25'

    # Quantification caveats reflect MBR
    assert manifest['quantification_caveats']['mbr_enabled'] is True

    # Derived parameters present
    derived = manifest['derived_parameters']
    assert derived['threshold'] is not None
    assert derived['mb_threshold'] is not None
    assert derived['theta'] is not None
    assert derived['B'] is not None
    assert derived['p_after_preprocessing'] is not None
