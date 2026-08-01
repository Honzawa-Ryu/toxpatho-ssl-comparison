import os
import sys
import yaml
import json
import logging
import random
from pathlib import Path

# --- Generic Data Science & ML Imports ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

# --- Generic Deep Learning Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# =============================================================================
# Setup & Initialization
# =============================================================================

def setup_logger(output_dir: Path, exp_name: str):
    """Set up a logger that writes to both console and a file in the output directory."""
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if output_dir.exists():
        fh = logging.FileHandler(output_dir / "experiment.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def get_env_paths():
    """Fetch environment variables injected by the Slurm script."""
    project_root = os.environ.get("PROJECT_ROOT")
    output_dir = os.environ.get("OUTPUT_DIR")
    dataset_dir = os.environ.get("DATASET_DIR")
    exp_name = os.environ.get("EXP_NAME")

    if not all([project_root, output_dir, dataset_dir, exp_name]):
        print("Error: Required environment variables are missing.")
        print("Please ensure this script is executed via run_slurm.sh")
        sys.exit(1)

    project_root = Path(project_root)

    # append dependent experiments output directory
    dependent_exps_str = os.environ.get("DEPENDENT_EXPS", "")
    prev_data_paths = {}
    if dependent_exps_str:
        for exp in dependent_exps_str.split(","):
            exp = exp.strip()
            if exp:
                prev_data_paths[exp] = project_root / "outputs" / exp

    return {
        "project_root": project_root,
        "output_dir": Path(output_dir),
        "dataset_dir": Path(dataset_dir),
        "exp_name": exp_name,
        "prev_data_paths": prev_data_paths
    }

def load_config(exp_dir: Path):
    """Load config.yml if it exists in the experiment directory."""
    config_path = exp_dir / "config.yml"
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {} # Return empty dict if no config was generated

# =============================================================================
# Experiment Components
# =============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

def load_data(dataset_dir: Path, logger: logging.Logger):
    """Load and preprocess the dataset."""
    # TODO: Implement data loading and preprocessing logic
    pass

def run_experiment(data, output_dir: Path, logger: logging.Logger):
    """The main logic for your experiment, modeling, or analysis."""
    # TODO: Implement the core experiment logic here
    
    return {}

# =============================================================================
# Main Execution
# =============================================================================

def main():
    # 1. Load infrastructure paths
    paths = get_env_paths()
    project_root = paths["project_root"]
    output_dir = paths["output_dir"]
    dataset_dir = paths["dataset_dir"]
    exp_name = paths["exp_name"]

    # 2. Setup Logger
    logger = setup_logger(output_dir, "experiment")
    logger.info(f"--- Starting Experiment: {exp_name} ---")
    logger.info(f"Project Root: {project_root}")
    logger.info(f"Output Dir:   {output_dir}")
    logger.info(f"Dataset Dir: {dataset_dir}")

    # 3. Load Configuration (Optional)
    exp_dir = project_root / "experiments" / exp_name
    config = load_config(exp_dir)
    
    if config:
        logger.info("Configuration loaded successfully.")
    else:
        logger.info("No config.yml found.")

    set_seed(config.get("seed", 42))
    
    # 4. setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 5. Data Processing
    logger.info("Loading and processing data...")
    data = load_data(dataset_dir, logger)

    # 6. Run Experiment
    logger.info("Executing main experiment...")
    results = run_experiment(data, output_dir, logger)
    
    # 7. Save Final Outputs
    logger.info("Saving experiment results...")
    results_path = output_dir / "results.json"
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
        
    logger.info(f"Experiment finished successfully. Results saved to {output_dir}")

if __name__ == "__main__":
    main()