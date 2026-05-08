"""
True Shah & Samworth (2013) ElasticNet stability selection package.

Uses the r-concave PFER bound (equation 8) for threshold derivation,
which is the main theoretical contribution of the paper over Meinshausen
& Buhlmann (2010). Complementary pairs subsampling with data-adaptive
threshold computed from the D function (Appendix A.4).

Operates on protein-level DIA-NN matrices (pg_matrix). Peptide-level input
is rejected explicitly — see io.load_data and the package README for the
rationale.
"""

__version__ = "0.3.0"

from .config import Config
from .preprocessing import Preprocessor
from .io import load_data, save_results
from .visualization import create_visualizations

from .rconcave import compute_D, compute_rconcave_threshold, print_threshold_comparison
from .feature_selection import stability_selection_elasticnet, calculate_cohens_d
from .workflows import full_dataset_stability_elasticnet

__all__ = [
    'Config',
    'Preprocessor',
    'load_data',
    'save_results',
    'create_visualizations',
    'compute_D',
    'compute_rconcave_threshold',
    'print_threshold_comparison',
    'stability_selection_elasticnet',
    'calculate_cohens_d',
    'full_dataset_stability_elasticnet',
]
