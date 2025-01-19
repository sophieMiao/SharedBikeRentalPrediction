import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    """Time series dataset class"""

    def __init__(self, x, y):
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def prepare_single_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare single DataFrame with feature engineering"""
    numeric_features = [
        "season",
        "yr",
        "mnth",
        "hr",
        "holiday",
        "workingday",
        "weathersit",
        "temp",
        "atemp",
        "hum",
        "windspeed",
        "cnt",
    ]
    return df[numeric_features]


class TimeSeriesDatasets:
    def __init__(self, meta: dict, lookback, prediction_horizon):
        self.train_df = pd.read_csv(meta["train"]["path"])
        self.test_df = pd.read_csv(meta["test"]["path"])
        self.meta = meta
        self.lookback = lookback
        self.prediction_horizon = prediction_horizon
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

    def __getitem__(self, item):
        assert item in self.meta
        if item == "train":
            return self.prepare_data(self.train_df, True, self.meta[item]["batch_size"])
        return self.prepare_data(self.test_df, False, self.meta[item]["batch_size"])

    def prepare_data(
        self, df: pd.DataFrame, is_training: bool = False, batch_size: int = 32
    ) -> DataLoader:
        """Prepare data for training or prediction"""
        df = df.sort_values("instant")
        df_processed = prepare_single_df(df)

        if is_training:
            scaled_data = self.scaler.fit_transform(df_processed)
        else:
            scaled_data = self.scaler.transform(df_processed)

        x, y = [], []
        for i in range(len(scaled_data) - self.lookback - self.prediction_horizon + 1):
            x.append(scaled_data[i : (i + self.lookback), :])
            y.append(
                scaled_data[
                    i + self.lookback : i + self.lookback + self.prediction_horizon, -1
                ]
            )

        dataset = TimeSeriesDataset(np.array(x), np.array(y))
        return DataLoader(dataset, batch_size=batch_size, shuffle=is_training)
