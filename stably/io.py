"""Input/output: data loading, result saving, and run manifest generation.

This package supports protein-level DIA-NN matrices (`pg_matrix`) only.
Peptide-level input (`pr_matrix`, detected via the `Stripped.Sequence` column)
is rejected explicitly — ElasticNet stability selection on peptide-level data
gives multiple independent votes to each protein in proportion to its
proteotypic-peptide count, which inflates selection probabilities for
well-represented proteins.

Quantification caveat (HC-FDR-06 principle): DIA-NN's `pg_matrix` does not
distinguish MS2-confirmed from MBR-transferred quantifications. The run
manifest surfaces whether MBR was enabled upstream so downstream readers can
contextualise the stable-feature list.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from . import __version__ as PKG_VERSION
from .diann_log import find_diann_log, parse_diann_log


def load_data(config) -> Tuple[pd.DataFrame, np.ndarray, List[str], pd.DataFrame]:
    """
    Load protein-level proteomics data and align samples to labels.

    Returns
    -------
    X : pd.DataFrame
        Abundance matrix, samples × proteins (raw, pre-preprocessing).
    y : np.ndarray
        Binary labels (0 = control, 1 = case).
    common_sample_ids : list of str
        Sample IDs in `X.index` order.
    protein_metadata : pd.DataFrame
        Metadata for every protein row in the input matrix.
    """
    pg_matrix = pd.read_csv(config.DATA_FILE)
    group_labels = pd.read_excel(config.LABEL_FILE)

    # Reject peptide-level input. Each peptide from the same protein would
    # contribute an independent feature to ElasticNet, concentrating stability
    # votes in well-represented proteins — scientifically unsound for this
    # pipeline. Use stats_analysis.py for peptide-level analyses.
    if 'Stripped.Sequence' in pg_matrix.columns:
        raise ValueError(
            "Peptide-level input (pr_matrix) is not supported. This pipeline "
            "requires protein-level input (pg_matrix). Use stats_analysis.py "
            "for peptide-level analyses."
        )

    n_metadata_cols = 6
    print("Detected protein-level matrix (6 metadata columns)")

    # Filter proteins with insufficient proteotypic peptides
    min_peptides = config.MIN_PROTEOTYPIC_PEPTIDES
    if min_peptides >= 1:
        if 'N.Proteotypic.Sequences' in pg_matrix.columns:
            print(f"  Filtering proteins with fewer than {min_peptides} proteotypic peptides...")
            proteins_before = len(pg_matrix)
            pg_matrix = pg_matrix[pg_matrix['N.Proteotypic.Sequences'] >= min_peptides]
            proteins_after = len(pg_matrix)
            print(f"    Removed {proteins_before - proteins_after} proteins "
                  f"with <{min_peptides} proteotypic peptides")
            print(f"    Retained {proteins_after} proteins")
        else:
            print(f"  Warning: 'N.Proteotypic.Sequences' column not found. "
                  f"Skipping proteotypic filter.")

    sample_columns = pg_matrix.columns[n_metadata_cols:]
    sample_ids_csv = sample_columns.tolist()

    group_labels_indexed = group_labels.set_index(config.SAMPLE_ID_COLUMN)
    available_label_ids = set(group_labels_indexed.index)

    sample_ids_csv_set = set(sample_ids_csv)
    common_samples = sample_ids_csv_set.intersection(available_label_ids)
    common_sample_ids = [sid for sid in sample_ids_csv if sid in common_samples]

    missing_in_labels = sample_ids_csv_set - available_label_ids
    if missing_in_labels:
        print(f"Warning: {len(missing_in_labels)} samples in data file have no labels "
              f"and will be excluded:")
        print(f"  {list(missing_in_labels)[:5]}{'...' if len(missing_in_labels) > 5 else ''}")

    protein_metadata = pg_matrix.iloc[:, :n_metadata_cols].copy()

    X = pg_matrix[common_sample_ids].T
    X.columns = pg_matrix.index

    group_labels_aligned = group_labels_indexed.loc[common_sample_ids]

    label_map = {config.CONTROL_NAME: 0, config.CASE_NAME: 1}
    raw_labels = group_labels_aligned[config.GROUP_COLUMN].astype(str).str.strip()
    y_series = raw_labels.map(label_map)

    unmapped = y_series.isna()
    if unmapped.any():
        bad_vals = raw_labels[unmapped].unique().tolist()
        print(f"Warning: {unmapped.sum()} samples have unrecognised labels in "
              f"'{config.GROUP_COLUMN}': {bad_vals}  — dropping them.")
        keep = ~unmapped
        y_series = y_series[keep]
        common_sample_ids = [sid for sid, k in zip(common_sample_ids, keep) if k]
        X = X.loc[common_sample_ids]

    y = y_series.values.astype(int)

    return X, y, common_sample_ids, protein_metadata


def save_results(results, protein_metadata, config):
    """
    Write the stable-features CSV, pickle, JSON schema sidecar, and run
    manifest to the configured output directory.
    """
    output_path = Path(config.OUTPUT_DIR)
    output_path.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*60}")
    print(f"SAVING RESULTS TO: {output_path.absolute()}")
    print(f"{'='*60}")

    _write_stable_features_csv(results, protein_metadata, output_path, timestamp, config)
    _write_pickle(results, output_path, timestamp)
    _write_schema_sidecar(output_path, timestamp, config)
    _write_run_manifest(results, config, output_path, timestamp)

    print(f"\n{'='*60}")
    print(f"All results saved to: {output_path.absolute()}")
    print(f"{'='*60}")


def _get_gene_info(feature_idx, protein_metadata):
    """Look up gene/protein info for a protein-level feature index."""
    if feature_idx in protein_metadata.index:
        return {
            'feature_index': feature_idx,
            'gene': protein_metadata.loc[feature_idx, 'Genes'],
            'protein_group': protein_metadata.loc[feature_idx, 'Protein.Group'],
            'protein_names': protein_metadata.loc[feature_idx, 'Protein.Names'],
        }
    return {
        'feature_index': feature_idx,
        'gene': str(feature_idx),
        'protein_group': 'Unknown',
        'protein_names': 'Unknown',
    }


def _write_stable_features_csv(results, protein_metadata, output_path, timestamp, config):
    feat_stab = results['feature_stability']
    cohens_d_map = feat_stab.get('feature_cohens_d', {})
    sel_probs_map = feat_stab.get('selection_probabilities', {})
    preprocessing_info = results.get('preprocessing', {})
    imputation_rate_per_feature = preprocessing_info.get('imputation_rate_per_feature', {})

    rows = []
    for feat_idx in feat_stab['feature_frequency']:
        info = _get_gene_info(feat_idx, protein_metadata)
        d_entry = cohens_d_map.get(feat_idx, {})
        rows.append({
            'feature_index': info['feature_index'],
            'gene': info['gene'],
            'protein_group': info['protein_group'],
            'protein_names': info['protein_names'],
            'selection_probability': sel_probs_map.get(feat_idx, float('nan')),
            'cohens_d': d_entry.get('cohens_d', float('nan')),
            'direction': d_entry.get('direction', 'undetermined'),
            'n_case': d_entry.get('n_case', 0),
            'n_control': d_entry.get('n_control', 0),
            'imputation_rate': float(imputation_rate_per_feature.get(feat_idx, 0.0)),
        })

    df = pd.DataFrame(rows)
    if 'selection_probability' in df.columns:
        df = df.sort_values('selection_probability', ascending=False)
    out_file = output_path / f'stable_features_{timestamp}.csv'
    df.to_csv(out_file, index=False)
    print(f"  Stable features: {out_file.name}")


def _write_pickle(results, output_path, timestamp):
    out_file = output_path / f'full_results_{timestamp}.pkl'
    with open(out_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"  Full results (pickle): {out_file.name}")


def _write_schema_sidecar(output_path, timestamp, config):
    schema = {
        'file': f'stable_features_{timestamp}.csv',
        'columns': {
            'feature_index': 'Row index from the DIA-NN pg_matrix, identifying the protein group.',
            'gene': 'Gene symbol(s) from the DIA-NN Genes column. Not version-pinned by itself — see run_manifest.upstream_diann.fasta_name.',
            'protein_group': 'Protein group accession(s) from DIA-NN.',
            'protein_names': 'Protein name(s) from DIA-NN.',
            'selection_probability': (
                'Empirical proportion of complementary-pairs subsamples in which this feature '
                'was among the top q ElasticNet coefficients (0..1). Higher = more stable. '
                'The threshold π for calling a feature stable is the r-concave PFER-derived '
                'threshold recorded in run_manifest.derived_parameters.threshold.'
            ),
            'cohens_d': (
                "Signed Cohen's d on the preprocessed (log + z-scored + imputed) matrix: "
                f"(mean_{config.CASE_NAME} - mean_{config.CONTROL_NAME}) / pooled_std. "
                f"Positive = higher in {config.CASE_NAME}."
            ),
            'direction': f"'up_case' (higher in {config.CASE_NAME}), 'up_control' (higher in {config.CONTROL_NAME}), 'zero', or 'undetermined' (n<2 in a group).",
            'n_case': f"Number of non-NaN {config.CASE_NAME} samples contributing to this feature's Cohen's d.",
            'n_control': f"Number of non-NaN {config.CONTROL_NAME} samples contributing to this feature's Cohen's d.",
            'imputation_rate': (
                'Fraction of samples (case + control) whose value for this feature was '
                'imputed. Computed on the matrix after the missing-value filter but before '
                'imputation. 0 = every observation was MS2-quantified. See '
                'run_manifest.resolved_config.imputation_strategy for the method.'
            ),
        },
    }
    out_file = output_path / f'stable_features_{timestamp}.schema.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(schema, f, indent=2)
    print(f"  Schema: {out_file.name}")


def _write_run_manifest(results, config, output_path, timestamp):
    manifest = {
        'package': {
            'name': 'stably',
            'version': PKG_VERSION,
        },
        'runtime': {
            'python': sys.version,
            'platform': platform.platform(),
            'dependency_versions': _capture_dep_versions(),
        },
        'inputs': {
            'data_file': str(config.DATA_FILE),
            'data_file_sha256': _file_sha256(config.DATA_FILE),
            'label_file': str(config.LABEL_FILE),
            'label_file_sha256': _file_sha256(config.LABEL_FILE),
        },
        'upstream_diann': _gather_diann_provenance(config),
        'resolved_config': _serialise_config(config),
        'derived_parameters': _derive_parameters(results),
        'sample_counts': results.get('sample_counts', {}),
        'preprocessing': _serialise_preprocessing(results.get('preprocessing', {})),
        'convergence': results.get('stability_selection', {}).get('threshold_info', {}),
        'quantification_caveats': {
            # HC-FDR-06 principle: DIA-NN's pg_matrix does not distinguish
            # MBR-transferred from MS2-confirmed values; this package treats
            # them uniformly. The flag here is informational.
            'mbr_enabled': _gather_diann_provenance(config).get('mbr_enabled'),
            'note': (
                'Quantifications come from DIA-NN pg_matrix; MBR-transferred '
                'values (if MBR was enabled upstream) are included without a '
                'per-value flag because DIA-NN does not expose this distinction '
                'in the matrix output.'
            ),
        },
        'timestamp': timestamp,
    }

    out_file = output_path / f'run_manifest_{timestamp}.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"  Run manifest: {out_file.name}")


def _capture_dep_versions() -> dict:
    versions = {}
    for mod_name in ('numpy', 'pandas', 'scipy', 'sklearn', 'joblib', 'matplotlib', 'seaborn', 'yaml', 'openpyxl'):
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, '__version__', 'unknown')
        except ImportError:
            versions[mod_name] = 'not installed'
    return versions


def _file_sha256(path) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _gather_diann_provenance(config) -> dict:
    log_path = getattr(config, 'DIANN_LOG_FILE', None)
    if log_path:
        log_path = Path(log_path)
        if not log_path.exists():
            return {'log_file': str(log_path), 'found': False, 'reason': 'configured path does not exist'}
    else:
        log_path = find_diann_log(config.DATA_FILE)
        if log_path is None:
            return {'found': False, 'reason': 'no DIA-NN log auto-discovered near data_file'}

    try:
        parsed = parse_diann_log(log_path)
        parsed['found'] = True
        return parsed
    except OSError as e:
        return {'log_file': str(log_path), 'found': False, 'reason': str(e)}


def _serialise_config(config) -> dict:
    out = {}
    for name in vars(config):
        if name.startswith('_'):
            continue
        value = getattr(config, name)
        if isinstance(value, (str, int, float, bool, type(None))):
            out[name.lower()] = value
        else:
            out[name.lower()] = str(value)
    return out


def _derive_parameters(results) -> dict:
    ss = results.get('stability_selection', {}) or {}
    info = ss.get('threshold_info', {}) or {}
    return {
        'C_ref': ss.get('C_ref'),
        'threshold': ss.get('threshold'),
        'mb_threshold': info.get('mb_threshold'),
        'threshold_reduction': info.get('threshold_reduction'),
        'theta': info.get('theta'),
        'B': info.get('B'),
        'pfer_target': info.get('pfer_target'),
        'pfer_bound_at_tau': info.get('pfer_bound'),
        'p_after_preprocessing': info.get('p_after_preprocessing'),
    }


def _serialise_preprocessing(pp: dict) -> dict:
    """Flatten only the summary fields into the manifest — per-feature rates live
    in the stable-features CSV and the pickle."""
    if not pp:
        return {}
    rate = pp.get('imputation_rate_per_feature')
    global_rate = None
    if rate is not None:
        try:
            global_rate = float(sum(rate.values()) / max(len(rate), 1))
        except (TypeError, AttributeError):
            global_rate = None
    return {
        'imputation_strategy': pp.get('imputation_strategy'),
        'n_features_after_missing_filter': pp.get('n_features_after_missing_filter'),
        'n_imputed_cells_total': pp.get('n_imputed_cells_total'),
        'global_imputation_rate': global_rate,
        'log_transform': pp.get('log_transform'),
        'max_missing': pp.get('max_missing'),
    }
