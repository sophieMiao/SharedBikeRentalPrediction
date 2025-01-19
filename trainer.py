import torch
import torch.nn as nn
import numpy as np

from numpy import ndarray
from torch.utils.data import DataLoader
from typing import Optional, Tuple
from pathlib import Path

from data import TimeSeriesDatasets
from models import get_model


class Trainer:
    """Bike rental prediction class with full training and evaluation capabilities"""

    def __init__(
        self,
        model_name: str,
        lookback=96,
        prediction_horizon=96,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.lookback = lookback
        self.prediction_horizon = prediction_horizon
        self.datasets = TimeSeriesDatasets(
            {
                "train": {"path": "./train_data.csv", "batch_size": 128},
                "test": {"path": "./test_data.csv", "batch_size": 128},
            },
            lookback,
            prediction_horizon,
        )
        self.device = device
        self.model = None
        self.best = None
        self.feature_columns = None
        self.model_path = f"{model_name}_{model_path}"
        self.input_dim = 12
        self.scaler = None

    def save_model(self, path: str):
        """Save model state and scaler"""
        if self.model is not None:
            model_state = {
                "model_state_dict": self.model.state_dict(),
                "scaler": self.scaler,
                "input_dim": self.input_dim,
            }
            torch.save(model_state, path)

    def load_model(self, path: str):
        """Load model state and scaler"""
        if Path(path).exists():
            checkpoint = torch.load(path, map_location=self.device)
            self.input_dim = checkpoint["input_dim"]
            self.scaler = checkpoint["scaler"]
            self.datasets.scaler = self.scaler
            self.model = get_model(
                self.model_name, self.input_dim, self.prediction_horizon
            ).to(self.device)

            self.model.load_state_dict(checkpoint["model_state_dict"])
            return True
        return False

    def inverse_transform(self, batch: ndarray) -> ndarray:
        # Inverse transform predictions
        # batch_size, seq_len, dim_data = batch.shape
        # dummy_data = np.zeros((batch_size * seq_len, self.scaler.n_features_in_))
        # dummy_data[:, -1] = batch[:, :, -1].flatten()
        # return self.scaler.inverse_transform(dummy_data)[:, -1].reshape(batch_size, seq_len)
        dummy_data = np.zeros((batch.shape[1], self.scaler.n_features_in_))
        dummy_data[:, -1] = batch[:, :]
        return self.scaler.inverse_transform(dummy_data)[:, -1]

    def train(
        self,
        train_loader: DataLoader,
        epochs: int = 50,
        learning_rate: float = 0.001,
        save_best: bool = True,
    ):
        """Train the model"""
        if self.model is None:
            self.model = get_model(
                self.model_name, self.input_dim, self.prediction_horizon
            ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        best_loss = float("inf")
        self.model.train()
        self.scaler = self.datasets.scaler

        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            scheduler.step(avg_loss)

            if avg_loss < best_loss and save_best and self.model_path:
                print("Save Model")
                best_loss = avg_loss
                self.save_model(self.model_path)

            # if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    def eval(self, test_loader: DataLoader) -> Tuple[ndarray, ndarray]:
        """Generate predictions"""
        self.load_model(self.model_path)
        self.model.eval()
        with torch.no_grad():
            pred, y = [], []
            uninvtrs_pred, uninvtrs_y = [], []
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                predictions = self.model(batch_x[:1]).cpu().numpy()
                uninvtrs_pred.append(predictions)
                uninvtrs_y.append(batch_y[:1])
                pred.append(self.inverse_transform(predictions))
                y.append(self.inverse_transform(batch_y[:1]))
                break

            pred = np.concatenate(pred, axis=None)
            y = np.concatenate(y, axis=None)
            uninvtrs_pred = np.concatenate(uninvtrs_pred, axis=None)
            uninvtrs_y = np.concatenate(uninvtrs_y, axis=None)
        return pred, y, uninvtrs_pred, uninvtrs_y
