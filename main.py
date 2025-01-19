# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "9"
import argparse

import torch
import numpy as np
import json

from typing import Dict, Optional
import random

from trainer import Trainer


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Evaluate model performance"""
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)

    return {"mse": mse, "mae": mae, "rmse": rmse}


def run_experiment(
    lr,
    num_experiments: int = 5,
    model_name: str = "lstm",
    model_path: str = "best_model.pth",
    len_=96,
):
    """Run multiple experiments and collect results"""
    results = {"mse": [], "mae": [], "rmse": []}

    visualization_data = None
    print(f"\nRunning {num_experiments} experiments...")

    for i in range(num_experiments):
        print(f"\nStarting experiment {i + 1}...")

        # Set random seeds
        seed = i * 100 + 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        try:
            predictor = Trainer(
                model_name=model_name,
                lookback=96,
                prediction_horizon=len_,
                model_path=model_path,
            )

            # Prepare data
            train_loader = predictor.datasets["train"]

            # Train model
            predictor.train(train_loader, epochs=100, save_best=True, learning_rate=lr)

            # Generate predictions and evaluate
            pred, y, up, uy = predictor.eval(predictor.datasets["test"])
            metrics = evaluate_model(uy, up)

            # Save results
            for metric, value in metrics.items():
                results[metric].append(value)

            print(f"Experiment {i + 1} results:")
            print(f"MSE: {metrics['mse']:.2f}")
            print(f"MAE: {metrics['mae']:.2f}")
            print(f"RMSE: {metrics['rmse']:.2f}")

            # Save visualization data from last experiment
            if i == num_experiments - 1:
                visualization_data = {
                    "predictions": pred,
                    "actual": y,
                    "time_points": np.arange(len(pred)),
                }

        except Exception as e:
            print(f"Error in experiment {i + 1}: {str(e)}")
            raise e

    # Calculate statistics
    stats = {"mean": {}, "std": {}}

    if not results["mse"]:
        return {
            "mean": {"mse": np.nan, "mae": np.nan, "rmse": np.nan},
            "std": {"mse": np.nan, "mae": np.nan, "rmse": np.nan},
        }, None

    for metric in ["mse", "mae", "rmse"]:
        stats["mean"][metric] = np.mean(results[metric])
        stats["std"][metric] = np.std(results[metric])

    return stats, visualization_data


def save_results(
    stats: Dict[str, Dict[str, float]],
    visualization_data: Dict[str, np.ndarray],
    output_file: str = "experiment_results.json",
):
    """Save experiment results to JSON file"""
    results = {
        "statistics": {
            "mean": {k: float(v) for k, v in stats["mean"].items()},
            "std": {k: float(v) for k, v in stats["std"].items()},
        }
    }

    if visualization_data is not None:
        results["visualization"] = {
            "predictions": visualization_data["predictions"].tolist(),
            "actual": visualization_data["actual"].tolist(),
            "time_points": visualization_data["time_points"].tolist(),
        }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)


import matplotlib.pyplot as plt


def plot_prediction_comparison_from_testdata(
    actual: np.ndarray,
    predictions: np.ndarray,
    time_points: np.ndarray,
    output_file: Optional[str] = None,
):
    # 从测试数据中提取真实值（真实值是 'cnt' 列）
    plt.figure(figsize=(12, 6))
    plt.plot(time_points, actual, label="Ground Truth (Actual)", linewidth=2, alpha=0.8)
    plt.plot(
        time_points,
        predictions,
        label="Predictions",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
    )
    plt.xlabel("Time Points", fontsize=14)
    plt.ylabel("Bike Rentals (cnt)", fontsize=14)
    plt.title("Bike Rental Prediction vs Ground Truth", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    if output_file:
        plt.savefig(output_file, bbox_inches="tight")
        print(f"Plot saved to {output_file}")
    plt.show()


def main(args):
    """Main function to run the experiment"""
    try:
        # Run experiments
        stats, viz_data = run_experiment(
            model_name=args.model,
            num_experiments=5,
            model_path="best_model.pth",
            lr=args.lr,
            len_=args.len
        )

        # Print results
        print("\n=== Final Results ===")
        print(f"MSE - Mean: {stats['mean']['mse']:.2f}, Std: {stats['std']['mse']:.2f}")
        print(f"MAE - Mean: {stats['mean']['mae']:.2f}, Std: {stats['std']['mae']:.2f}")
        print(
            f"RMSE - Mean: {stats['mean']['rmse']:.2f}, Std: {stats['std']['rmse']:.2f}"
        )

        # Save experiment results
        save_results(stats, viz_data, f"{args.model}_experiment_results.json")
        print("\nResults saved to experiment_results.json")

        # Plot prediction comparison
        if viz_data:
            plot_prediction_comparison_from_testdata(
                actual=viz_data["actual"],
                predictions=viz_data["predictions"],
                time_points=viz_data["time_points"],
                output_file=f"{args.model}_prediction_comparison.png",
            )

    except Exception as e:
        print(f"Error in execution: {str(e)}")
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trainer")
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-l", "--lr", type=float, default=0.001)
    parser.add_argument("-ln", "--len", type=int, default=96)
    args = parser.parse_args()
    print(args)
    main(args)
