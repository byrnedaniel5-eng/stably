"""Configuration management for the true S&S ElasticNet stability selection pipeline."""

import yaml
from pathlib import Path
from typing import Optional


class Config:
    """
    Configuration manager that loads settings from a YAML file.

    Usage:
        config = Config.from_yaml('config.yaml')
        print(config.DATA_FILE)
    """

    def __init__(self, config_dict: dict):
        """Initialize Config from a dictionary."""
        # Data files
        self.DATA_FILE = config_dict['data_file']
        self.LABEL_FILE = config_dict['label_file']

        # Sample information
        self.CASE_NAME = config_dict['case_name']
        self.CONTROL_NAME = config_dict['control_name']
        self.GROUP_COLUMN = config_dict['group_column']
        self.SAMPLE_ID_COLUMN = config_dict['sample_id_column']

        # Random seed
        self.RANDOM_STATE = config_dict['random_state']

        # Preprocessing parameters
        self.MAX_MISSING = config_dict['max_missing']
        self.LOG_TRANSFORM = config_dict['log_transform']
        self.CORRELATION_THRESHOLD = config_dict['correlation_threshold']
        self.IMPUTATION_STRATEGY = config_dict['imputation_strategy']
        self.KNN_NEIGHBORS = config_dict.get('knn_neighbors', 5)
        self.MIN_PROTEOTYPIC_PEPTIDES = config_dict.get('min_proteotypic_peptides', 2)

        # Upstream DIA-NN log (for database version capture, HC-INTER-03).
        # Optional: when None, load_data will try to auto-discover a *.log.txt
        # next to the data file or in a sibling PDC_outputs_*/ directory.
        self.DIANN_LOG_FILE = config_dict.get('diann_log_file', None)

        # Machine learning
        self.MAX_ITERATIONS = config_dict['max_iterations']

        # ElasticNet mixing parameter
        self.L1_RATIO = config_dict.get('l1_ratio', 0.5)

        # Stability selection
        self.MAX_CANDIDATES = config_dict['max_candidates']
        self.STABILITY_ITERATIONS = config_dict['stability_iterations']
        self.STABILITY_PFER = float(config_dict['stability_pfer'])
        self.N_JOBS = config_dict['n_jobs']

        # Output
        self.OUTPUT_DIR = config_dict.get('output_dir', 'biomarker_results')

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """
        Load configuration from a YAML file.

        Parameters
        ----------
        yaml_path : str
            Path to the YAML configuration file

        Returns
        -------
        Config
            Configuration object
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """Create Config from a dictionary (useful for testing)."""
        return cls(config_dict)
