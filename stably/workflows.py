"""High-level workflow for the true Shah & Samworth (2013) r-concave ElasticNet
stability selection pipeline.

Uses the r-concave bound (equation 8) for threshold derivation instead of the
Meinshausen & Buhlmann worst-case bound. Collects preprocessing and run
provenance into the returned results dict so save_results / write_run_manifest
can serialise it.
"""

import numpy as np

from .preprocessing import Preprocessor
from .feature_selection import stability_selection_elasticnet, calculate_cohens_d


def full_dataset_stability_elasticnet(X_raw, y, config):
    """
    Full-dataset S&S ElasticNet stability selection with r-concave threshold.

    Returns a results dict containing feature_stability, stability_selection,
    preprocessing provenance, and sample counts. Returns None if no stable
    features are identified.
    """
    print("\nRunning true S&S ElasticNet stability selection (r-concave bound)...")
    print(f"All {X_raw.shape[0]} samples used — subsample size ≈ {X_raw.shape[0] // 2}.")

    preprocessor = Preprocessor(config, verbose=True)
    X = preprocessor.fit_transform(X_raw, y)

    p = X.shape[1]
    k = config.MAX_CANDIDATES
    B = config.STABILITY_ITERATIONS // 2
    pfer = config.STABILITY_PFER
    print(f"  p={p} features after preprocessing")
    print(f"  θ = q/p = {k}/{p} = {k/p:.4f}")
    print(f"  B = {B} complementary pairs ({config.STABILITY_ITERATIONS} subsamples)")

    stable_features, threshold, selection_probs, C_ref, threshold_info = (
        stability_selection_elasticnet(X, y, config)
    )

    threshold_info['p_after_preprocessing'] = p

    print(f"\n{len(stable_features)} stable features (π={threshold:.4f}, C={C_ref:.2e})")

    if not stable_features:
        print("No stable features found.")
        return None

    cohens_d = calculate_cohens_d(X, y, stable_features)

    n_case = int(np.sum(y == 1))
    n_control = int(np.sum(y == 0))

    return {
        'n_stable': len(stable_features),
        'feature_stability': {
            'stable_features': stable_features,
            'feature_frequency': {f: 1 for f in stable_features},
            'feature_cohens_d': cohens_d,
            'selection_probabilities': selection_probs,
        },
        'stability_selection': {
            'threshold': threshold,
            'C_ref': C_ref,
            'threshold_info': threshold_info,
        },
        'preprocessing': {
            'imputation_strategy': config.IMPUTATION_STRATEGY,
            'log_transform': config.LOG_TRANSFORM,
            'max_missing': config.MAX_MISSING,
            'n_features_initial': X_raw.shape[1],
            'n_features_after_missing_filter': len(preprocessor.features_after_missing or []),
            'n_imputed_cells_total': preprocessor.n_imputed_total,
            'imputation_rate_per_feature': (
                preprocessor.imputation_rate_per_feature.to_dict()
                if preprocessor.imputation_rate_per_feature is not None
                else {}
            ),
            'features_after_missing': preprocessor.features_after_missing,
        },
        'sample_counts': {
            'n_total': int(len(y)),
            'n_case': n_case,
            'n_control': n_control,
            'class_labels': {
                'case': config.CASE_NAME,
                'control': config.CONTROL_NAME,
            },
        },
    }
