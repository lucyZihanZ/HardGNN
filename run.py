# ========================================================================
# CELL 2: Dataset Configuration and Module Import
# ========================================================================

# --- Import Environment Setup Utilities ---
import sys
try:
    # Attempt to import the setup functions from the dedicated environment setup script.
    # Ensure 'quest_env_setup.py' is in your Python path or in the same directory.
    from quest_env_setup import install_dependencies, setup_tensorflow_gpu_compatibility
    print("✅ Successfully imported environment setup utilities.")
except ImportError as e:
    print(f"❌ Failed to import quest_env_setup: {e}")
    print("Please ensure 'quest_env_setup.py' is in your project directory or PYTHONPATH.")
    sys.exit(1) # Exit if essential setup utilities cannot be found

# --- Initial Environment Setup ---
# 1. Install/Upgrade all dependencies (including TensorFlow)
# This function will handle pip installs for all necessary libraries.
install_dependencies()

# 2. Import TensorFlow and configure it for GPU and v1 compatibility
# This assumes TensorFlow was successfully installed by install_dependencies().
print("\n🔄 Importing TensorFlow and configuring GPU...")
tf_module = None # Initialize to None
try:
    import tensorflow as tf
    tf_module = tf
    print(f"✅ TensorFlow version loaded: {tf_module.__version__} from {tf_module.__file__}")

    # Call the GPU and compatibility setup function
    setup_successful = setup_tensorflow_gpu_compatibility(tf_module=tf_module)
    if not setup_successful:
        raise RuntimeError("TensorFlow GPU and compatibility setup failed. Please check logs.")

except ImportError as e:
    print(f"❌ FAILED to import TensorFlow after installation: {e}")
    print("   Please ensure TensorFlow is correctly installed and accessible in your virtual environment.")
    sys.exit(1)
except RuntimeError as e:
    print(f"❌ CRITICAL ERROR during TensorFlow setup: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ An unexpected error occurred while importing or setting up TensorFlow: {e}")
    sys.exit(1)

# Make the globally configured TensorFlow available as tf
tf = tf_module

# Core Python Standard Library imports
import os
import random
import pickle
import json
from datetime import datetime
import time
import gc
import traceback
import queue
import threading
import multiprocessing as mp

# Third-party library imports
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import pandas as pd

# --- Define Global Configuration Variables (Adjust as needed) ---
# These would typically come from an external config file or command-line arguments.
GRID_SEARCH_ENABLED = False # Set to True to enable grid search for K and Lambda
DATASET = 'yelp'          # Choose 'yelp', 'amazon', 'gowalla', 'movielens'
SINGLE_K = 5              # K for hard negative sampling in single experiment mode
SINGLE_LAMBDA = 0.1       # Lambda (contrastive weight) in single experiment mode
SINGLE_EPOCHS = 150       # Number of epochs for single experiment mode

# Grid Search specific configurations (only relevant if GRID_SEARCH_ENABLED is True)
HARD_NEG_SAMPLES_K = [0, 1, 5, 10] # K values to test in grid search
CONTRASTIVE_WEIGHTS = [0.0, 0.01, 0.1, 1.0] # Lambda values to test in grid search
GRID_SEARCH_EPOCHS = 50 # Reduced epochs for grid search to speed up testing
DRIVE_RESULTS_PATH = '~/hardgnn_results' # Path to save grid search results (adjust for your system)


# Import our custom modules
print("\n🔧 Importing custom project modules...")
try:
    # Assuming Params.py, Utils/TimeLogger.py, and DataHandler.py are in the current directory or Python path
    from Params import args # This assumes Params.py defines an 'args' object/class
    import Utils.TimeLogger as logger
    from Utils.TimeLogger import log
    from DataHandler import DataHandler
    print("✅ Successfully imported core project modules.")
except ImportError as e:
    print(f"❌ Failed to import core project modules: {e}")
    print("Please ensure Params.py, Utils/TimeLogger.py, and DataHandler.py are accessible.")
    sys.exit(1) # Exit if core modules cannot be found

# Import the HardGNN model
print("\n🔧 Importing HardGNN model...")
try:
    from HardGNN_model import Recommender
    print("✅ Successfully imported HardGNN model")
except ImportError as e:
    print(f"❌ Failed to import HardGNN model: {e}")
    print("Please ensure 'HardGNN_model.py' is in your project directory and all its dependencies are installed.")
    sys.exit(1)

def configure_dataset(dataset_name, hard_neg_k=5, contrastive_weight=0.1):
    """
    Configures global parameters (args) based on validated configurations for each dataset.
    This function modifies the global `args` object from `Params.py`.
    """
    print(f"\n⚙️ Configuring parameters for {dataset_name.upper()} dataset...")

    # Set base dataset
    args.data = dataset_name.lower()

    # Dataset-specific validated configurations
    if dataset_name.lower() == 'yelp':
        # From yelp.sh - validated configuration
        args.lr = 1e-3
        args.reg = 1e-2
        args.temp = 0.1
        args.ssl_reg = 1e-7
        args.epoch = 150
        args.batch = 512
        args.sslNum = 40
        args.graphNum = 12
        args.gnn_layer = 3
        args.att_layer = 2
        args.testSize = 1000
        args.ssldim = 32
        args.sampNum = 40

    elif dataset_name.lower() == 'amazon':
        # From amazon.sh - validated configuration
        args.lr = 1e-3
        args.reg = 1e-2
        args.temp = 0.1
        args.ssl_reg = 1e-6
        args.epoch = 150
        args.batch = 512
        args.sslNum = 80
        args.graphNum = 5
        args.pred_num = 0
        args.gnn_layer = 3
        args.att_layer = 4
        args.testSize = 1000
        args.keepRate = 0.5
        args.sampNum = 40
        args.pos_length = 200

    elif dataset_name.lower() == 'gowalla':
        # From gowalla.sh - validated configuration
        args.lr = 2e-3
        args.reg = 1e-2
        args.temp = 0.1
        args.ssl_reg = 1e-6
        args.epoch = 150
        args.batch = 512
        args.graphNum = 3
        args.gnn_layer = 2
        args.att_layer = 1
        args.testSize = 1000
        args.sampNum = 40

    elif dataset_name.lower() == 'movielens':
        # From movielens.sh - validated configuration
        args.lr = 1e-3
        args.reg = 1e-2
        args.ssl_reg = 1e-6
        args.epoch = 150
        args.batch = 512
        args.sampNum = 40
        args.sslNum = 90
        args.graphNum = 6
        args.gnn_layer = 2
        args.att_layer = 3
        args.testSize = 1000
        args.ssldim = 48
        args.keepRate = 0.5
        args.pos_length = 200
        args.leaky = 0.5

    else:
        print(f"⚠️  Unknown dataset: {dataset_name}. Available datasets: yelp, amazon, gowalla, movielens")
        print("Using default parameters for unlisted datasets, which may not be optimized.")

    # Handle edge cases for hard negative sampling and contrastive loss
    if hard_neg_k == 0:
        # K=0: Disable hard negative sampling entirely
        args.use_hard_neg = False
        args.hard_neg_top_k = 0
        print("ℹ️ Hard negative sampling disabled (hard_neg_k = 0).")
    else:
        # K>0: Enable hard negative sampling
        args.use_hard_neg = True
        args.hard_neg_top_k = hard_neg_k
        print(f"ℹ️ Hard negative sampling enabled with top K = {hard_neg_k}.")

    # Set contrastive weight (λ=0 is handled in model during loss computation)
    args.contrastive_weight = contrastive_weight
    print(f"ℹ️ Contrastive loss weight (λ) set to {contrastive_weight}.")
    # Note: τ (temperature) is already set in args.temp = 0.1 during dataset-specific configuration

    # Evaluation configuration
    args.shoot = 20  # Set top-K evaluation metric to top-20
    print(f"ℹ️ Evaluation will be performed for top-{args.shoot}.")

    # Performance optimization for hard negative sampling
    args.cache_refresh_steps = 25  # Refresh embeddings every N training steps
    print(f"ℹ️ Embedding cache will refresh every {args.cache_refresh_steps} steps.")

    # Enable AMP (Automatic Mixed Precision) to be handled within the model
    # This flag will be read by HardGNN_model.py to enable mixed precision training on GPU.
    args.enable_amp = True
    print(f"ℹ️ Automatic Mixed Precision (AMP) enabled: {args.enable_amp}.")

    args.tstEpoch = 3  # Test every 3 epochs (can be adjusted if needed for full runs)
    print(f"ℹ️ Model will be tested every {args.tstEpoch} epochs.")

    # Set save path for models and results
    args.save_path = f'hardgnn_results/{dataset_name.lower()}_k{hard_neg_k}_lambda{contrastive_weight}'
    print(f"ℹ️ Results and model checkpoints will be saved to: {args.save_path}")

    return args

# --- Configure the dataset based on execution mode ---
if GRID_SEARCH_ENABLED:
    # Use first combination for initial setup to ensure args are populated
    configure_dataset(DATASET, HARD_NEG_SAMPLES_K[0], CONTRASTIVE_WEIGHTS[0])
    print(f"\n🔬 Grid Search Mode Enabled")
    print(f"   K values to test: {HARD_NEG_SAMPLES_K}")
    print(f"   λ values to test: {CONTRASTIVE_WEIGHTS}")
    print(f"   Epochs per experiment: {GRID_SEARCH_EPOCHS}")
    print(f"   Total experiments: {len(HARD_NEG_SAMPLES_K) * len(CONTRASTIVE_WEIGHTS)}")
    print(f"   Results will be saved to a subdirectory under: {DRIVE_RESULTS_PATH}")
    # Create base results directory for grid search
    os.makedirs(DRIVE_RESULTS_PATH, exist_ok=True)
    print(f"📁 Base grid search results directory created: {DRIVE_RESULTS_PATH}")
else:
    # Single experiment mode
    configure_dataset(DATASET, SINGLE_K, SINGLE_LAMBDA)
    args.epoch = SINGLE_EPOCHS # Override epochs for single run
    print(f"\n🎯 Single Experiment Mode Enabled")
    print(f"   Hard Negatives (K): {args.hard_neg_top_k}")
    print(f"   Contrastive Weight (λ): {args.contrastive_weight}")
    print(f"   Total Epochs: {args.epoch}")
    # Create specific save path directory for single experiment
    os.makedirs(os.path.dirname(args.save_path) if os.path.dirname(args.save_path) else '.', exist_ok=True)
    print(f"📁 Single experiment results directory created: {os.path.dirname(args.save_path) if os.path.dirname(args.save_path) else 'Current directory'}")


print("\n✅ HardGNN modules imported and configured successfully.")
print(f"\n📊 Final Configuration for {DATASET.upper()} Dataset:")
print(f"  Dataset: {args.data}")
print(f"  Learning Rate: {args.lr}")
print(f"  Regularization: {args.reg}")
print(f"  Temperature (τ): {args.temp}")
print(f"  SSL Regularization: {args.ssl_reg}")
print(f"  Batch Size: {args.batch}")
print(f"  Graph Number: {args.graphNum}")
print(f"  GNN Layers: {args.gnn_layer}")
print(f"  Attention Layers: {args.att_layer}")
print("🔥 Hard Negative Sampling Configuration:")
print(f"  Enabled: {args.use_hard_neg}")
if GRID_SEARCH_ENABLED:
    print(f"  Mode: Grid Search")
    print(f"  K values: {HARD_NEG_SAMPLES_K}")
    print(f"  λ values: {CONTRASTIVE_WEIGHTS}")
    print(f"  Epochs per experiment: {GRID_SEARCH_EPOCHS}")
else:
    print(f"  Mode: Single Experiment")
    print(f"  Hard Negatives (K): {args.hard_neg_top_k}")
    print(f"  Contrastive Weight (λ): {args.contrastive_weight}")
    print(f"  Total Epochs: {args.epoch}")

print(f"  Evaluation @ Top-K: {args.shoot}")
print(f"  Cache Refresh Steps: {args.cache_refresh_steps}")
print(f"  Automatic Mixed Precision (AMP) Enabled: {args.enable_amp}")
print(f"  Test Every Epochs: {args.tstEpoch}")
print(f"  Save Path: {args.save_path}")


