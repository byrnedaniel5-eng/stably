"""Visualization functions for the true S&S ElasticNet stability selection pipeline."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def create_visualizations(results, protein_metadata, config):
    """
    Create visualizations for full-dataset stability selection results.

    Parameters
    ----------
    results : dict
        Results dictionary from full_dataset_stability_elasticnet
    protein_metadata : pd.DataFrame
        Protein metadata with Genes column for feature names
    config : Config
        Configuration object
    """
    output_dir = Path(config.OUTPUT_DIR) / 'figures'
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}")

    feature_stability = results['feature_stability']
    gene_mapping = dict(zip(protein_metadata.index, protein_metadata['Genes']))

    sorted_features = sorted(
        feature_stability['feature_frequency'].items(),
        key=lambda x: x[1], reverse=True
    )

    # 1. Selection probability bar chart
    sel_probs = feature_stability.get('selection_probabilities', {})
    if sel_probs:
        print("  Creating selection probability plot...")
        top_n = min(30, len(sorted_features))
        top_features_by_prob = sorted(
            [(f, sel_probs.get(f, 0.0)) for f, _ in sorted_features[:top_n]],
            key=lambda x: x[1], reverse=True
        )
        feature_names = [gene_mapping.get(f, str(f)) for f, _ in top_features_by_prob]
        probs = [p for _, p in top_features_by_prob]

        fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.3)))
        ax.barh(range(len(feature_names)), probs, color='steelblue', alpha=0.8)
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names, fontsize=10)
        ax.set_xlabel('Selection Probability (\u03c0\u0302)', fontsize=12)
        ax.set_ylabel('Feature (Gene)', fontsize=12)
        ax.set_title(f'Top {top_n} Features by Selection Probability', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1.05])
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        fig_path = output_dir / 'selection_probabilities.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    {fig_path}")

    # 2. Cohen's d bar chart for stable features
    cohens_d = feature_stability.get('feature_cohens_d', {})
    stable_features = feature_stability.get('stable_features', [])
    if cohens_d and stable_features:
        print("  Creating Cohen's d plot...")
        d_items = []
        for f in stable_features:
            entry = cohens_d.get(f, {})
            d_value = entry.get('cohens_d', float('nan')) if isinstance(entry, dict) else entry
            if d_value is None or np.isnan(d_value):
                continue
            d_items.append((f, d_value))
        d_items_sorted = sorted(d_items, key=lambda x: abs(x[1]), reverse=True)[:30]

        if d_items_sorted:
            feat_names = [gene_mapping.get(f, str(f)) for f, _ in d_items_sorted]
            d_values = [d for _, d in d_items_sorted]
            colors = ['firebrick' if d > 0 else 'steelblue' for d in d_values]

            fig, ax = plt.subplots(figsize=(10, max(8, len(d_items_sorted) * 0.3)))
            ax.barh(range(len(feat_names)), d_values, color=colors, alpha=0.8)
            ax.set_yticks(range(len(feat_names)))
            ax.set_yticklabels(feat_names, fontsize=10)
            ax.set_xlabel("Cohen's d", fontsize=12)
            ax.set_ylabel('Feature (Gene)', fontsize=12)
            ax.set_title("Stable Features: Effect Size (Cohen's d)", fontsize=14, fontweight='bold')
            ax.axvline(0, color='black', linewidth=0.8)
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)

            plt.tight_layout()
            fig_path = output_dir / 'cohens_d.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    {fig_path}")

    print(f"\n  All visualizations saved to: {output_dir}/")
