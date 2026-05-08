# stably

ElasticNet stability selection with the **Shah & Samworth (2013) r-concave PFER bound**, for protein-level biomarker discovery from DIA-NN output.

The selection-probability threshold π is derived per-run from the r-concave tail bound (S&S 2013, equation 8) using the data-dependent quantity θ = q/p. This is tighter than the Meinshausen & Buhlmann (2010) worst-case bound and so retains more features for the same PFER budget. The package operates on **protein-level** DIA-NN matrices (`pg_matrix`); peptide-level input is rejected explicitly.

---

## Installation

Requires Python ≥ 3.10.

```bash
# Clone or copy the package source, then from its parent directory:
cd "Protein-level analysis"
pip install -e ./stably
```

The `-e` (editable) install lets you keep tweaking the package source without reinstalling.

To also install test dependencies:

```bash
pip install -e "./stably[test]"
```

Verify the install:

```bash
python -c "import stably; print(stably.__version__)"
# 0.3.0
```

### Dependencies

Pulled in automatically by `pip install`:
`numpy`, `pandas`, `scipy`, `scikit-learn`, `joblib`, `matplotlib`, `seaborn`, `pyyaml`, `openpyxl`.

---

## Input data

Two files are required:

### 1. Protein-level abundance matrix (`pg_matrix`)

A CSV produced by DIA-NN. The first 6 columns are protein-level metadata (`Protein.Group`, `Protein.Ids`, `Protein.Names`, `Genes`, `First.Protein.Description`, `N.Proteotypic.Sequences`); every column from index 6 onwards is a sample, named by the run/biobank ID.

Peptide-level matrices (`pr_matrix`, detectable by a `Stripped.Sequence` column) are rejected at load time — see the docstring on [io.py](stably/io.py) for why.

### 2. Sample-label spreadsheet

An `.xlsx` file with at least:
- a column matching the sample IDs used as column headers in the pg_matrix (e.g. `Biobank Number`)
- a group column with two values matching `case_name` and `control_name` from the config (e.g. `Group` containing `PDAC` / `Non-cancer`)

Samples present in the pg_matrix but missing from the label file are dropped with a warning.

---

## Configuration

Runs are driven by a YAML config — see [config_stably.yaml](config_stably.yaml) for an annotated reference. The most important keys:

| Key | Meaning |
| --- | --- |
| `data_file`, `label_file` | Paths to the two input files |
| `case_name`, `control_name` | Labels in the group column to map to y=1 / y=0 |
| `group_column`, `sample_id_column` | Column names in the label file |
| `min_proteotypic_peptides` | Drops proteins with `N.Proteotypic.Sequences` below this (default 2) |
| `max_missing` | Per-protein NaN fraction allowed after sample alignment |
| `imputation_strategy` | `knn`, `minimum`, `mean`, or `median` |
| `log_transform` | Apply log2(x+1) before scaling |
| `max_candidates` (q) | Number of top-|β| coefficients retained per subsample |
| `stability_iterations` | 2 × B; **S&S §3.4.1 recommends B ≤ 50** so leave at 100 unless you understand the r-concavity assumption |
| `stability_pfer` | Absolute PFER budget E[V] |
| `l1_ratio` | ElasticNet mixing parameter (1 = lasso, 0 = ridge) |
| `output_dir` | Results destination |

The threshold π is **not** specified directly; it is derived per-run from `(q, p, B, PFER)` via the r-concave bound after preprocessing fixes p.

---

## Running on real data

Use the driver script [run_stably.py](run_stably.py):

```bash
python run_stably.py --config config_stably.yaml
```

The script prints class balance, q/p, B, the r-concave vs M&B threshold comparison, calibrated `C_ref`, the top-20 stable features by selection probability π̂, and writes results to `output_dir`.

---

## Toy example with synthetic data

Use this to sanity-check the install on a tiny fabricated dataset (no DIA-NN run required). It produces a `pg_matrix`-shaped CSV, an Excel label file, and a config, then runs the pipeline.

Save as `toy_run.py` next to the package and run with `python toy_run.py`:

```python
"""Minimal end-to-end smoke test for stably on synthetic data."""
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

rng = np.random.default_rng(0)
n_samples_per_group = 20
n_proteins = 200
n_true_signal = 10  # proteins with a real case/control difference

work = Path("toy_workspace")
work.mkdir(exist_ok=True)

# 1. Build a synthetic pg_matrix-shaped CSV.
sample_ids = [f"S{i:03d}" for i in range(2 * n_samples_per_group)]
y = np.array([0] * n_samples_per_group + [1] * n_samples_per_group)

X = rng.normal(loc=20.0, scale=1.0, size=(n_proteins, len(sample_ids)))
# Inject a real signal into the first n_true_signal proteins (up in cases).
X[:n_true_signal, y == 1] += rng.normal(loc=2.0, scale=0.3,
                                        size=(n_true_signal, n_samples_per_group))

metadata = pd.DataFrame({
    "Protein.Group":              [f"P{i:05d}" for i in range(n_proteins)],
    "Protein.Ids":                [f"P{i:05d}" for i in range(n_proteins)],
    "Protein.Names":              [f"PROT{i}_HUMAN" for i in range(n_proteins)],
    "Genes":                      [f"GENE{i}" for i in range(n_proteins)],
    "First.Protein.Description":  [f"Synthetic protein {i}" for i in range(n_proteins)],
    "N.Proteotypic.Sequences":    rng.integers(2, 8, size=n_proteins),
})
pg_matrix = pd.concat([metadata, pd.DataFrame(X, columns=sample_ids)], axis=1)
pg_matrix.to_csv(work / "toy.pg_matrix.csv", index=False)

# 2. Build a label spreadsheet matching the sample IDs.
labels = pd.DataFrame({
    "Biobank Number": sample_ids,
    "Group":          ["Non-cancer" if yi == 0 else "PDAC" for yi in y],
})
labels.to_excel(work / "toy_labels.xlsx", index=False)

# 3. Write a config.
cfg = {
    "data_file":               str(work / "toy.pg_matrix.csv"),
    "label_file":              str(work / "toy_labels.xlsx"),
    "control_name":            "Non-cancer",
    "case_name":               "PDAC",
    "group_column":            "Group",
    "sample_id_column":        "Biobank Number",
    "random_state":            42,
    "max_sample_missing":      0.7,
    "max_missing":             0.0,
    "log_transform":           False,   # data is already on a log-like scale
    "correlation_threshold":   0.95,
    "imputation_strategy":     "knn",
    "knn_neighbors":           5,
    "min_proteotypic_peptides": 2,
    "max_iterations":          5000,
    "max_candidates":          20,
    "stability_iterations":    100,     # 2 × B = 2 × 50
    "stability_pfer":          5,
    "l1_ratio":                0.3,
    "n_jobs":                  -1,
    "output_dir":              str(work / "toy_results"),
}
cfg_path = work / "toy_config.yaml"
with open(cfg_path, "w") as f:
    yaml.safe_dump(cfg, f)

# 4. Run the pipeline programmatically.
from stably import (
    Config, load_data, save_results, create_visualizations,
    full_dataset_stability_elasticnet,
)

config = Config.from_yaml(cfg_path)
np.random.seed(config.RANDOM_STATE)

X_raw, y, sample_ids, protein_metadata = load_data(config)
results = full_dataset_stability_elasticnet(X_raw, y, config)

if results is None:
    print("No stable features found.")
else:
    save_results(results, protein_metadata, config)
    create_visualizations(results, protein_metadata, config)

    sel_probs = results["feature_stability"]["selection_probabilities"]
    stable    = results["feature_stability"]["stable_features"]
    top = sorted(stable, key=lambda f: sel_probs[f], reverse=True)[:10]
    print("\nTop stable features:")
    for f in top:
        gene = protein_metadata.loc[f, "Genes"]
        print(f"  {gene:>10s}  π̂ = {sel_probs[f]:.3f}")
```

If the install is healthy you should see most of the first 10 injected proteins (`GENE0`–`GENE9`) show up in the stable list. The toy dataset is small enough to finish in a few seconds on a laptop.

---

## Outputs

`output_dir` will contain four timestamped artefacts per run:

| File | Contents |
| --- | --- |
| `stable_features_<ts>.csv` | One row per *measured* protein with `selection_probability`, `cohens_d`, `direction`, `n_case`, `n_control`, `imputation_rate`. Sorted by π̂. The "stable" subset is everything with π̂ ≥ threshold (recorded in the manifest). |
| `stable_features_<ts>.schema.json` | Per-column descriptions for the CSV. |
| `full_results_<ts>.pkl` | Pickled `results` dict (feature_stability, stability_selection, preprocessing, sample_counts). |
| `run_manifest_<ts>.json` | Reproducibility manifest: package + dependency versions, input SHA-256 hashes, upstream DIA-NN log provenance, resolved config, derived parameters (`C_ref`, threshold, M&B threshold, θ, B, PFER), preprocessing summary, convergence stats. |

`create_visualizations` additionally writes selection-probability and threshold-comparison plots to the same directory.

---

## Programmatic API

The driver script is just a thin wrapper. The same pipeline from Python:

```python
from stably import (
    Config, load_data, save_results, create_visualizations,
    full_dataset_stability_elasticnet,
)

config = Config.from_yaml("config_stably.yaml")
X_raw, y, sample_ids, protein_metadata = load_data(config)
results = full_dataset_stability_elasticnet(X_raw, y, config)
save_results(results, protein_metadata, config)
create_visualizations(results, protein_metadata, config)
```

Lower-level entry points exposed at the package root:

- `Preprocessor(config).fit_transform(X, y)` — leakage-safe missing-filter + log + impute + z-score; reusable on held-out data via `.transform`.
- `stability_selection_elasticnet(X, y, config)` — returns `(stable_features, threshold, selection_probs, C_ref, threshold_info)`.
- `compute_rconcave_threshold(q, p, B, pfer)` and `compute_D(...)` — the threshold maths in isolation.
- `calculate_cohens_d(X, y, features)` — signed effect size on the preprocessed matrix.

---

## Tests

```bash
pytest stably/tests          # fast suite
pytest stably/tests --runslow  # includes the empirical PFER calibration test
```

The suite covers the r-concave bound (`test_rconcave.py`), an empirical PFER check (`test_pfer_empirical.py`), Cohen's d edge cases, peptide-input rejection, preprocessing leakage guarantees, and the end-to-end CSV/manifest schema.

---

## Theory — one paragraph

For each subsample the procedure fits an ElasticNet logistic regression at a fixed regularisation `C_ref` (calibrated once on the full data so each subsample yields ≤ q non-zero coefficients), then records the top-q features by |β|. With B complementary pairs (2B subsamples) the selection probability π̂ for each feature is the empirical proportion of subsamples in which it appeared. A feature is called stable when π̂ ≥ π, where π is the smallest threshold for which the **r-concave bound** (Shah & Samworth 2013, eq. 8)

> E[V] ≤ min{ D(θ², 2τ-1, B, -½),  D(θ, τ, 2B, -¼) } × p

is below the user-specified PFER budget. θ = q/p is data-dependent and so π must be computed *after* preprocessing has fixed p. The bound reduces to M&B's 1/(2τ-1) bound when r-concavity is dropped, hence the manifest reports both for diagnostic purposes.

---

## Citation

Shah, R. D. & Samworth, R. J. (2013). *Variable selection with error control: another look at stability selection.* JRSS B, 75(1):55–80.
