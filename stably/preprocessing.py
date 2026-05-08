"""Preprocessing module for protein-level proteomics data.

Pipeline order:
    1. Missing-value filter (drop features with too many NaNs).
    2. Optional log2(x+1) transform.
    3. Imputation (minimum / knn / sklearn SimpleImputer strategy).
    4. Per-feature z-score scaling.

Imputation provenance (HC-QUANT-02) is recorded as:
    - self.missing_mask: DataFrame[bool], NaN locations in the post-missing-filter
      matrix (before imputation).
    - self.imputation_rate_per_feature: Series, fraction of samples imputed for
      each feature that survived the missing-value filter.

This package supports protein-level input only. Peptide-level (pr_matrix)
inputs are rejected upstream in io.load_data; no peptide deduplication is
performed here.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer


class Preprocessor:
    """
    Fit on training data, then transform train and test independently.

    The fit/transform split exists so nested-CV workflows can avoid leakage:
    imputer, scaler, and the post-missing-filter feature list are all learned
    on training data and reused on held-out data.
    """

    def __init__(self, config, verbose=True):
        self.config = config
        self.max_missing = config.MAX_MISSING
        self.log_transform = config.LOG_TRANSFORM
        self.verbose = verbose

        # Fitted state
        self.imputer = None
        self.scaler = None
        self.features_after_missing = None
        self.min_values = None  # for custom "minimum" strategy
        self.missing_mask = None
        self.imputation_rate_per_feature = None
        self.n_imputed_total = None

    def fit_transform(self, X, y=None):
        """Fit all transformers on X and return the preprocessed matrix."""
        n_features_initial = X.shape[1]

        # 1. Drop features with too many missing values
        missing_fraction = X.isnull().mean()
        self.features_after_missing = X.columns[missing_fraction <= self.max_missing].tolist()
        X = X[self.features_after_missing]
        n_features_after_missing = X.shape[1]
        if self.verbose:
            print(
                f"    Removed {n_features_initial - n_features_after_missing} features "
                f"with >{self.max_missing*100:.0f}% missing values "
                f"({n_features_after_missing} remaining)"
            )

        # 2. Optional log transform
        if self.log_transform:
            X = np.log2(X + 1)

        # 3. Capture imputation provenance before imputing (HC-QUANT-02)
        self.missing_mask = X.isnull()
        self.imputation_rate_per_feature = self.missing_mask.mean(axis=0)
        self.n_imputed_total = int(self.missing_mask.values.sum())
        if self.verbose and self.n_imputed_total > 0:
            print(
                f"    Imputation: {self.n_imputed_total} cells "
                f"({self.n_imputed_total / self.missing_mask.size * 100:.2f}% of matrix) "
                f"will be imputed via '{self.config.IMPUTATION_STRATEGY}'"
            )

        # 4. Impute
        X_imputed = self._fit_imputer(X)

        # 5. Scale
        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_imputed), columns=X.columns, index=X.index
        )
        return X_scaled

    def transform(self, X):
        """Apply fitted transformers to new data (no leakage)."""
        if self.features_after_missing is None:
            raise RuntimeError("Preprocessor has not been fit. Call fit_transform first.")

        # Align to the feature set learned on training data
        X = X.reindex(columns=self.features_after_missing)

        if self.log_transform:
            X = np.log2(X + 1)

        X_imputed = self._apply_imputer(X)
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_imputed), columns=X.columns, index=X.index
        )
        return X_scaled

    def _fit_imputer(self, X):
        strategy = self.config.IMPUTATION_STRATEGY
        if strategy == "minimum":
            self.min_values = X.min(axis=0)
            X_imputed = X.fillna(self.min_values)
            return pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
        if strategy == "knn":
            self.imputer = KNNImputer(n_neighbors=self.config.KNN_NEIGHBORS)
        else:
            self.imputer = SimpleImputer(strategy=strategy)
        return pd.DataFrame(
            self.imputer.fit_transform(X), columns=X.columns, index=X.index
        )

    def _apply_imputer(self, X):
        strategy = self.config.IMPUTATION_STRATEGY
        if strategy == "minimum":
            X_imputed = X.fillna(self.min_values)
            return pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
        return pd.DataFrame(
            self.imputer.transform(X), columns=X.columns, index=X.index
        )
