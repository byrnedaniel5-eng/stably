"""Verify that peptide-level input (pr_matrix) is rejected with a clear error.

Each peptide from the same protein would contribute an independent feature to
ElasticNet, concentrating stability votes in well-represented proteins and
inflating selection probabilities. The pipeline only supports protein-level
input.
"""

import pytest

from stably.config import Config
from stably.io import load_data
from stably.tests.conftest import make_synthetic_pr_matrix


def test_peptide_matrix_is_rejected(tmp_path, base_config_dict):
    pr_matrix = make_synthetic_pr_matrix(n_samples=20, n_peptides=40, seed=0)
    data_file = tmp_path / "peptide.pr_matrix.csv"
    label_file = tmp_path / "labels.xlsx"
    pr_matrix.to_csv(data_file, index=False)

    import pandas as pd
    pd.DataFrame({
        'Biobank Number': [f"S{i:03d}" for i in range(1, 21)],
        'Group': ['Case' if i % 2 == 0 else 'Control' for i in range(20)],
    }).to_excel(label_file, index=False)

    cfg_dict = dict(base_config_dict)
    cfg_dict.update({'data_file': str(data_file), 'label_file': str(label_file)})
    config = Config(cfg_dict)

    with pytest.raises(ValueError, match=r"protein-level"):
        load_data(config)
