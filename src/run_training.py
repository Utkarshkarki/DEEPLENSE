"""
Entry point for training the Lensiformer model.

Usage:
    python src/run_training.py                        # train lensiformer (default)
    python src/run_training.py --model resnet         # train ResNet18 baseline
    python src/run_training.py --model physics_cnn    # train PhysicsCNN baseline
    python src/run_training.py --epochs 10            # quick test with fewer epochs
"""

import argparse
import sys
import os

# Make sure 'src' is on the path so pinn_model imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pinn_model.train import train_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Lensiformer / baselines")
    parser.add_argument(
        "--model",
        choices=["lensiformer", "resnet", "physics_cnn"],
        default="lensiformer",
        help="Model to train (default: lensiformer)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (default: value from config.py = 100)"
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of training data to use (default: 1.0)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name for checkpoint/result files (default: model type)"
    )
    args = parser.parse_args()

    model, history = train_model(
        model_type=args.model,
        num_epochs=args.epochs,
        data_fraction=args.fraction,
        experiment_name=args.name,
    )
