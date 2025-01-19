from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from trans.trainer import Trainer

predictor = Trainer(lookback=96, prediction_horizon=96, model_path="best_model.pth")
predictor.load_model("best_model.pth")
pred, y = predictor.eval(predictor.datasets['test'])
print(pred, y)


def plot_prediction_comparison_from_testdata(actual: np.ndarray, predictions: np.ndarray, time_points: np.ndarray,
                                             output_file: Optional[str] = None):
    # 从测试数据中提取真实值（真实值是 'cnt' 列）
    plt.figure(figsize=(12, 6))
    plt.plot(time_points, actual, label="Ground Truth (Actual)", linewidth=2, alpha=0.8)
    plt.plot(time_points, predictions, label="Predictions", linestyle='--', linewidth=2, alpha=0.8)
    plt.xlabel("Time Points", fontsize=14)
    plt.ylabel("Bike Rentals (cnt)", fontsize=14)
    plt.title("Bike Rental Prediction vs Ground Truth", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    if output_file:
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    plt.show()


plot_prediction_comparison_from_testdata(y, pred, np.arange(len(pred)))
