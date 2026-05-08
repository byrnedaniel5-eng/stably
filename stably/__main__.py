"""
Command-line entry point for stably.

Uses the r-concave bound (equation 8, Shah & Samworth 2013) for threshold
derivation — the main theoretical contribution of S&S over Meinshausen &
Buhlmann (2010).

Usage:
    python -m stably --config config_stably.yaml
"""

import argparse
import sys
import warnings

import numpy as np

# Force UTF-8 output so unicode glyphs (θ, π, π̂) print correctly on Windows
# consoles whose default codepage is cp1252.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from stably import (
    Config,
    load_data,
    save_results,
    create_visualizations,
    full_dataset_stability_elasticnet,
)

# Silence noisy non-actionable warnings only. ConvergenceWarning is captured
# per-iteration inside the stability selection loop and surfaced in the run
# manifest, so we deliberately do not blanket-filter warnings here.
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def main():
    """Run the complete biomarker discovery pipeline."""
    parser = argparse.ArgumentParser(
        description='Biomarker Discovery: true S&S (2013) r-concave ElasticNet stability selection'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    args = parser.parse_args()

    print(f"Loading configuration from: {args.config}")
    config = Config.from_yaml(args.config)

    np.random.seed(config.RANDOM_STATE)

    print("=" * 60)
    print("TRUE S&S (2013) r-CONCAVE ELASTICNET STABILITY SELECTION")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    X_raw, y, sample_ids, protein_metadata = load_data(config)
    print(f"Loaded {X_raw.shape[0]} samples and {X_raw.shape[1]} features.")
    print(f"Class distribution: {np.sum(y==0)} {config.CONTROL_NAME}, {np.sum(y==1)} {config.CASE_NAME}.")

    k = config.MAX_CANDIDATES
    B = config.STABILITY_ITERATIONS // 2
    pfer = config.STABILITY_PFER
    print(f"\nMAX_CANDIDATES = {k}, L1_RATIO = {config.L1_RATIO}")
    print(f"B = {B} complementary pairs ({config.STABILITY_ITERATIONS} subsamples)")
    print(f"PFER budget: {pfer:.1f} expected false positives")
    print(f"NOTE: θ = q/p will be computed after preprocessing (p is data-dependent)")

    if B > 50:
        print(f"\n  WARNING: S&S Section 3.4.1 recommends B ≤ 50 for the r-concavity")
        print(f"  assumption to hold.  Current B = {B}.  Consider reducing")
        print(f"  stability_iterations to 100 (= 2 × 50).")

    results = full_dataset_stability_elasticnet(X_raw, y, config)

    if results is None:
        print("\nAnalysis failed — no stable features found.")
        return None

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")

    feat_stab = results['feature_stability']
    ss_info = results.get('stability_selection', {})
    threshold_info = ss_info.get('threshold_info', {})

    def get_gene_name(feature_idx):
        if feature_idx in protein_metadata.index:
            gene = protein_metadata.loc[feature_idx, 'Genes']
            return f"{gene} (index {feature_idx})"
        return str(feature_idx)

    threshold = ss_info.get('threshold', float('nan'))
    c_ref = ss_info.get('C_ref', float('nan'))
    sel_probs = feat_stab.get('selection_probabilities', {})
    stable_list = feat_stab.get('stable_features', [])
    sorted_stable = sorted(stable_list, key=lambda f: sel_probs.get(f, 0), reverse=True)

    print(f"\nStable features (sorted by selection probability):")
    print(f"  r-concave threshold: π = {threshold:.4f}")
    if threshold_info:
        print(f"  M&B threshold:       π = {threshold_info.get('mb_threshold', 'N/A'):.4f}")
        print(f"  Threshold reduction: Δπ = {threshold_info.get('threshold_reduction', 0):.4f}")
    print(f"  C_ref = {c_ref:.2e}")
    print(f"  Total stable: {len(sorted_stable)}")

    for i, feat in enumerate(sorted_stable[:20], 1):
        print(f"    {i}. {get_gene_name(feat)}  π̂={sel_probs.get(feat, 0):.3f}")

    print(f"\n{'=' * 60}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 60}")

    results['permutation_test'] = None

    save_results(results, protein_metadata, config)
    create_visualizations(results, protein_metadata, config)

    return results


if __name__ == "__main__":
    main()
