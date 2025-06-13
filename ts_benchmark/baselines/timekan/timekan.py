import math
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from einops import rearrange

from ts_benchmark.baselines.timekan.utils.tools import (
    EarlyStopping,
    adjust_learning_rate,
)
from ts_benchmark.utils.data_processing import split_time
from typing import Type, Dict, Optional, Tuple
from torch import optim
import numpy as np
import pandas as pd
from ts_benchmark.baselines.utils import (
    forecasting_data_provider,
    train_val_split,
    get_time_mark,
)
from ts_benchmark.baselines.timekan.models.timekan_model import TimeKANModeL
from ...models.model_base import ModelBase, BatchMaker

DEFAULT_HYPER_PARAMS = {
    "lradj": "type1",
    "data": "custom",
    "label_len": 48,
    "freq": "h",
    "seq_len": 96,
    "top_k": 5,
    "num_kernels": 6,
    "enc_in": 7,
    "dec_in": 7,
    "c_out": 7,
    "d_model": 16,
    "n_heads": 4,
    "e_layers": 2,
    "d_layers": 1,
    "d_ff": 32,
    "moving_avg": 25,
    "factor": 1,
    "distil": True,
    "dropout": 0.1,
    "embed": "timeF",
    "activation": "gelu",
    "output_attention": False,
    "channel_independence": 1,
    "decomp_method": "moving_avg",
    "use_norm": 1,
    "down_sampling_layers": 2,
    "down_sampling_window": 2,
    "use_future_temporal_feature": 0,
    "begin_order": 1,
    "mask_rate": 0.25,
    "anomaly_ratio": 0.25,
    "num_workers": 10,
    "itr": 1,
    "num_epochs": 100,
    "batch_size": 16,
    "patience": 10,
    "lr": 0.001,
    "des": "test",
    "loss": "MSE",
    "pct_start": 0.2,
    "use_amp": False,
    "comment": "none",
    "use_gpu": True,
    "gpu": 0,
    "use_multi_gpu": False,
    "devices": "0,1",
    "p_hidden_dims": [128, 128],
    "p_hidden_layers": 2,
    "use_mlp": False,
}


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, output_size):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size1, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class TimeKANConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def pred_len(self):
        return self.horizon


class TimeKAN(ModelBase):
    def __init__(self, **kwargs):
        super(TimeKAN, self).__init__()
        self.config = TimeKANConfig(**kwargs)
        self.scaler1 = StandardScaler()
        self.scaler2 = StandardScaler()
        self.seq_len = self.config.seq_len
        self.win_size = self.config.seq_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def model_name(self):
        return "TimeKAN"

    @staticmethod
    def required_hyper_params() -> dict:
        """
        Return the hyperparameters required by model.

        :return: An empty dictionary indicating that model does not require additional hyperparameters.
        """
        return {
            "seq_len": "input_chunk_length",
            "horizon": "output_chunk_length",
            "norm": "norm",
        }

    def multi_forecasting_hyper_param_tune(self, train_data: np.ndarray):
        self.config.freq = "h"  # Simplified frequency setting
        if len(train_data.shape) == 3:
            column_num = train_data.shape[1]
        else:
            column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num

        if self.model_name == "MICN":
            setattr(self.config, "label_len", self.config.seq_len)
        else:
            setattr(self.config, "label_len", self.config.seq_len // 2)

    def single_forecasting_hyper_param_tune(self, train_data: np.ndarray):
        self.config.freq = "h"  # Simplified for consistency
        if len(train_data.shape) == 3:
            column_num = train_data.shape[1]
        else:
            column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num
        setattr(self.config, "label_len", self.config.horizon)

    def validate(
            self, valid_data_loader: DataLoader, series_dim: int, criterion: torch.nn.Module
    ) -> float:
        """
        Validates the model performance on the provided validation dataset.
        :param valid_data_loader: A PyTorch DataLoader for the validation dataset.
        :param series_dim : The number of series data's dimensions.
        :param criterion : The loss function to compute the loss between model predictions and ground truth.
        :returns:The mean loss computed over the validation dataset.
        """
        config = self.config
        total_loss = []
        self.model.eval()
        if self.MLP is not None:
            self.MLP.eval()

        for input, target in valid_data_loader:
            input, target = input.to(self.device), target.to(self.device)

            # decoder input
            dec_input = torch.zeros_like(target[:, -config.horizon:, :]).float()
            dec_input = (
                torch.cat([target[:, : config.label_len, :], dec_input], dim=1)
                .float()
                .to(self.device)
            )

            exog_future = target[:, -config.horizon:, series_dim:].to(self.device)
            output = self.model(input)

            if self.config.use_mlp and self.MLP is not None:
                transformer_output = output[:, -config.horizon:, :series_dim]
                output = self.MLP(torch.cat((transformer_output, exog_future), dim=-1))
            else:
                output = output[:, -config.horizon:, :series_dim]

            target = target[:, -config.horizon:, :series_dim]
            loss = criterion(output, target).detach().cpu().numpy()
            total_loss.append(loss)

        total_loss = np.mean(total_loss)
        self.model.train()
        if self.MLP is not None:
            self.MLP.train()
        return total_loss

    def forecast_fit(
            self,
            train_valid_data: np.ndarray,
            *,
            covariates: Optional[dict] = None,
            train_ratio_in_tv: float = 1.0,
            **kwargs,
    ) -> "ModelBase":
        """
        Train the model.

        :param train_valid_data: Time series data used for training and validation (numpy array).
        :param covariates: Additional external variables.
        :param train_ratio_in_tv: Represents the splitting ratio of the training set validation set. If it is equal to 1, it means that the validation set is not partitioned.
        :return: The fitted model object.
        """
        if covariates is None:
            covariates = {}

        # Handle both 2D (pandas compatibility) and 3D data
        if len(train_valid_data.shape) == 2:
            # Convert 2D to 3D: (length, channels) -> (length, channels, 1)
            train_valid_data = train_valid_data[:, :, np.newaxis]

        series_dim = train_valid_data.shape[-2]
        exog_data = covariates.get("exog", None)
        if exog_data is not None:
            if len(exog_data.shape) == 2:
                exog_data = exog_data[:, :, np.newaxis]
            train_valid_data = np.concatenate((train_valid_data, exog_data), axis=1)
            exog_dim = exog_data.shape[-2]
        else:
            exog_dim = 0

        if train_valid_data.shape[1] == 1:
            train_drop_last = False
            self.single_forecasting_hyper_param_tune(train_valid_data)
        else:
            train_drop_last = True
            self.multi_forecasting_hyper_param_tune(train_valid_data)

        setattr(self.config, "task_name", "long_term_forecast")
        self.model = TimeKANModeL(self.config)

        # Initialize MLP if needed
        if self.config.use_mlp:
            input_size = series_dim + exog_dim
            output_size = series_dim
            self.MLP = MLP(input_size=input_size, hidden_size1=2048, output_size=output_size)
            self.MLP.to(self.device)
        else:
            self.MLP = None

        print(
            "----------------------------------------------------------",
            self.model_name,
        )
        config = self.config
        train_data, valid_data = train_val_split(
            train_valid_data, train_ratio_in_tv, config.seq_len
        )

        train_data_l = train_data.shape[0]
        valid_data_l = valid_data.shape[0] if train_ratio_in_tv != 1 else 0

        # Fit scalers
        if exog_dim > 0:
            # Fit scaler1 for series data
            self.scaler1.fit(rearrange(train_data[:, :series_dim, :], 'l c n->(l n) c'))
            # Fit scaler2 for exog data
            self.scaler2.fit(rearrange(train_data[:, series_dim:, :], 'l c n->(l n) c'))

            if config.norm:
                # Scale series data
                scaled_series = self.scaler1.transform(rearrange(train_data[:, :series_dim, :], 'l c n->(l n) c'))
                train_series = rearrange(scaled_series, '(l n) c -> l c n', l=train_data_l)

                # Scale exog data
                scaled_exog = self.scaler2.transform(rearrange(train_data[:, series_dim:, :], 'l c n->(l n) c'))
                train_exog = rearrange(scaled_exog, '(l n) c -> l c n', l=train_data_l)

                # Concatenate scaled data
                train_data = np.concatenate([train_series, train_exog], axis=1)
        else:
            # Only series data, use scaler1
            self.scaler1.fit(rearrange(train_data, 'l c n->(l n) c'))
            if config.norm:
                scaled_data = self.scaler1.transform(rearrange(train_data, 'l c n->(l n) c'))
                train_data = rearrange(scaled_data, '(l n) c -> l c n', l=train_data_l)

        if train_ratio_in_tv != 1:
            if config.norm:
                if exog_dim > 0:
                    # Scale validation series data
                    scaled_series = self.scaler1.transform(rearrange(valid_data[:, :series_dim, :], 'l c n->(l n) c'))
                    valid_series = rearrange(scaled_series, '(l n) c -> l c n', l=valid_data_l)

                    # Scale validation exog data
                    scaled_exog = self.scaler2.transform(rearrange(valid_data[:, series_dim:, :], 'l c n->(l n) c'))
                    valid_exog = rearrange(scaled_exog, '(l n) c -> l c n', l=valid_data_l)

                    # Concatenate scaled data
                    valid_data = np.concatenate([valid_series, valid_exog], axis=1)
                else:
                    scaled_data = self.scaler1.transform(rearrange(valid_data, 'l c n->(l n) c'))
                    valid_data = rearrange(scaled_data, '(l n) c -> l c n', l=valid_data_l)

            valid_dataset, valid_data_loader = forecasting_data_provider(
                valid_data,
                config,
                timeenc=1,
                batch_size=config.batch_size,
                shuffle=True,
                drop_last=False,
            )

        train_dataset, train_data_loader = forecasting_data_provider(
            train_data,
            config,
            timeenc=1,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=train_drop_last,
        )

        # Define the loss function and optimizer
        if config.loss == "MSE":
            criterion = nn.MSELoss()
        elif config.loss == "MAE":
            criterion = nn.L1Loss()
        else:
            criterion = nn.HuberLoss(delta=0.5)

        # Mixed optimizer when using MLP
        if self.MLP is not None:
            optimizer = optim.Adam([
                {'params': self.model.parameters(), 'lr': config.lr},
                {'params': self.MLP.parameters(), 'lr': config.lr * 0.1}
            ])
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=config.lr)

        self.early_stopping = EarlyStopping(patience=config.patience)
        self.model.to(self.device)

        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        if self.MLP is not None:
            total_params += sum(p.numel() for p in self.MLP.parameters() if p.requires_grad)

        print(f"Total trainable parameters: {total_params}")

        for epoch in range(config.num_epochs):
            self.model.train()
            if self.MLP is not None:
                self.MLP.train()

            # Only unpack 2 values as per adapters_for_transformers.py
            for i, (input, target) in enumerate(train_data_loader):
                optimizer.zero_grad()
                input, target = input.to(self.device), target.to(self.device)

                # decoder input
                dec_input = torch.zeros_like(target[:, -config.horizon:, :]).float()
                dec_input = (
                    torch.cat([target[:, : config.label_len, :], dec_input], dim=1)
                    .float()
                    .to(self.device)
                )

                exog_future = target[:, -config.horizon:, series_dim:].to(self.device)
                output = self.model(input)

                if self.config.use_mlp and self.MLP is not None:
                    transformer_output = output[:, -config.horizon:, :series_dim]
                    output = self.MLP(torch.cat((transformer_output, exog_future), dim=-1))
                else:
                    output = output[:, -config.horizon:, :series_dim]

                target = target[:, -config.horizon:, :series_dim]
                loss = criterion(output, target)

                loss.backward()
                optimizer.step()

            if train_ratio_in_tv != 1:
                valid_loss = self.validate(valid_data_loader, series_dim, criterion)
                if self.MLP is not None:
                    self.early_stopping(valid_loss, {'transformer': self.model, 'mlp': self.MLP})
                else:
                    self.early_stopping(valid_loss, {'transformer': self.model})
                if self.early_stopping.early_stop:
                    break

            adjust_learning_rate(optimizer, epoch + 1, config)

    def forecast(
            self,
            horizon: int,
            series: np.ndarray,
            *,
            covariates: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Make predictions.
        :param horizon: The predicted length.
        :param series: Time series data used for prediction (numpy array).
        :param covariates: Additional external variables
        :return: An array of predicted results.
        """
        if covariates is None:
            covariates = {}

        # Handle both 2D and 3D input
        if len(series.shape) == 2:
            series = series[:, :, np.newaxis]

        series_dim = series.shape[-2]
        exog_data = covariates.get("exog", None)
        if exog_data is not None:
            if len(exog_data.shape) == 2:
                exog_data = exog_data[:, :, np.newaxis]
            series = np.concatenate([series, exog_data], axis=1)
            if (
                    hasattr(self.config, "output_chunk_length")
                    and horizon != self.config.output_chunk_length
            ):
                raise ValueError(
                    f"Error: 'exog' is enabled during training, but horizon ({horizon}) != output_chunk_length ({self.config.output_chunk_length}) during forecast."
                )

        if self.early_stopping.check_point is not None:
            if isinstance(self.early_stopping.check_point, dict):
                self.model.load_state_dict(self.early_stopping.check_point['transformer'])
                if self.MLP is not None and 'mlp' in self.early_stopping.check_point:
                    self.MLP.load_state_dict(self.early_stopping.check_point['mlp'])
            else:
                # Backward compatibility
                self.model.load_state_dict(self.early_stopping.check_point)

        if self.config.norm:
            series_l = series.shape[0]
            if exog_data is not None and series.shape[1] > series_dim:
                # Scale series data with scaler1
                series_data = series[:, :series_dim, :]
                scaled_series = self.scaler1.transform(rearrange(series_data, 'l c n->(l n) c'))
                scaled_series = rearrange(scaled_series, '(l n) c -> l c n', l=series_l)

                # Scale exog data with scaler2
                exog_data = series[:, series_dim:, :]
                scaled_exog = self.scaler2.transform(rearrange(exog_data, 'l c n->(l n) c'))
                scaled_exog = rearrange(scaled_exog, '(l n) c -> l c n', l=series_l)

                # Combine scaled data
                series = np.concatenate([scaled_series, scaled_exog], axis=1)
            else:
                scaled_data = self.scaler1.transform(rearrange(series, 'l c n->(l n) c'))
                series = rearrange(scaled_data, '(l n) c -> l c n', l=series_l)

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        config = self.config
        # Simplified rolling prediction for numpy arrays
        self.model.to(self.device)
        self.model.eval()
        if self.MLP is not None:
            self.MLP.to(self.device)
            self.MLP.eval()

        with torch.no_grad():
            predictions = []
            current_input = series[-config.seq_len:, :, :]

            # Calculate number of prediction steps needed
            num_steps = math.ceil(horizon / config.horizon)

            for step in range(num_steps):
                # Convert to tensor and predict
                input_tensor = torch.tensor(current_input, dtype=torch.float32).unsqueeze(0).to(self.device)

                # If 3D, need to handle batch dimension properly
                if len(current_input.shape) == 3:
                    # Reshape to (batch, seq_len, features)
                    input_tensor = rearrange(input_tensor, 'b l c n -> (b n) l c')

                output = self.model(input_tensor)

                if self.config.use_mlp and self.MLP is not None and exog_data is not None:
                    # Extract future exogenous if available
                    start_idx = series.shape[0] + step * config.horizon
                    end_idx = min(start_idx + config.horizon, series.shape[0] + horizon)

                    if covariates.get("exog_future") is not None:
                        exog_future = covariates["exog_future"][start_idx:end_idx, :]
                        if len(exog_future.shape) == 2:
                            exog_future = exog_future[:, :, np.newaxis]
                        exog_future_tensor = torch.tensor(exog_future, dtype=torch.float32).to(self.device)

                        transformer_output = output[:, -config.horizon:, :series_dim]
                        output = self.MLP(torch.cat((transformer_output, exog_future_tensor), dim=-1))
                    else:
                        output = output[:, -config.horizon:, :series_dim]
                else:
                    output = output[:, -config.horizon:, :series_dim]

                # Extract predictions
                pred = output.cpu().numpy()
                if len(pred.shape) == 3:
                    pred = pred[0]  # Remove batch dimension
                predictions.append(pred)

                # Update input for next step
                if step < num_steps - 1:
                    # Shift and append predictions
                    if len(current_input.shape) == 3:
                        current_input = np.concatenate([
                            current_input[config.horizon:, :, :],
                            pred[:, :, :]
                        ], axis=0)
                    else:
                        current_input = np.concatenate([
                            current_input[config.horizon:, :],
                            pred
                        ], axis=0)

            # Concatenate all predictions
            all_predictions = np.concatenate(predictions, axis=0)[:horizon, :]

            if self.config.norm:
                # Only inverse transform series data with scaler1
                pred_l = all_predictions.shape[0]
                if len(all_predictions.shape) == 3:
                    scaled_data = self.scaler1.inverse_transform(
                        rearrange(all_predictions[:, :series_dim, :], 'l c n->(l n) c')
                    )
                    all_predictions = rearrange(scaled_data, '(l n) c -> l c n', l=pred_l)
                else:
                    all_predictions = self.scaler1.inverse_transform(all_predictions)

            # Return only series dimensions
            if len(all_predictions.shape) == 3:
                return all_predictions[:, :series_dim, 0]  # Remove the added dimension
            else:
                return all_predictions[:, :series_dim]

    def batch_forecast(
            self, horizon: int, batch_maker: BatchMaker, exog_futures=None, i=0, **kwargs
    ) -> np.ndarray:
        """
        Make predictions by batch.

        :param horizon: The length of each prediction.
        :param batch_maker: Make batch data used for prediction.
        :param exog_futures: Future exogenous variables.
        :param i: Batch index.
        :return: An array of predicted results.
        """
        if self.early_stopping.check_point is not None:
            if isinstance(self.early_stopping.check_point, dict):
                self.model.load_state_dict(self.early_stopping.check_point['transformer'])
                if self.MLP is not None and 'mlp' in self.early_stopping.check_point:
                    self.MLP.load_state_dict(self.early_stopping.check_point['mlp'])
            else:
                # Backward compatibility
                self.model.load_state_dict(self.early_stopping.check_point)

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        self.model.to(self.device)
        self.model.eval()
        if self.MLP is not None:
            self.MLP.to(self.device)
            self.MLP.eval()

        input_data = batch_maker.make_batch(self.config.batch_size, self.config.seq_len)
        input_np = input_data["input"]

        # Handle 3D data (batch x length x channels x samples)
        if len(input_np.shape) == 4:
            real_batch_size = self.config.batch_size * input_np.shape[3]
            series_dim = input_np.shape[-2]

            if input_data["covariates"] is None:
                covariates = {}
            else:
                covariates = input_data["covariates"]
            exog_data = covariates.get("exog")

            if exog_data is not None:
                exog_dim = exog_data.shape[-2]
                input_np = np.concatenate((input_np, exog_data), axis=2)
                if (
                        hasattr(self.config, "output_chunk_length")
                        and horizon != self.config.output_chunk_length
                ):
                    raise ValueError(
                        f"Error: 'exog' is enabled during training, but horizon ({horizon}) != output_chunk_length ({self.config.output_chunk_length}) during forecast."
                    )
            else:
                exog_dim = 0

            # Reshape from 4D to 3D
            input_np = rearrange(input_np, 'b l c n -> (b n) l c')
            input_np_b = input_np.shape[0]

            if self.config.norm:
                if exog_dim > 0:
                    # Scale series data with scaler1
                    series_data = input_np[:, :, :series_dim]
                    scaled_series = self.scaler1.transform(rearrange(series_data, 'b l c->(b l) c'))
                    scaled_series = rearrange(scaled_series, '(b l) c -> b l c', b=input_np_b)

                    # Scale exog data with scaler2
                    exog_data = input_np[:, :, series_dim:]
                    scaled_exog = self.scaler2.transform(rearrange(exog_data, 'b l c->(b l) c'))
                    scaled_exog = rearrange(scaled_exog, '(b l) c -> b l c', b=input_np_b)

                    # Combine scaled data
                    input_np = np.concatenate([scaled_series, scaled_exog], axis=2)
                else:
                    scaled_data = self.scaler1.transform(rearrange(input_np, 'b l c->(b l) c'))
                    input_np = rearrange(scaled_data, '(b l) c -> b l c', b=input_np_b)

            if exog_futures is not None and exog_dim > 0:
                exog_future = torch.tensor(
                    exog_futures[i * real_batch_size: (i + 1) * real_batch_size, -horizon:, :]
                ).to(self.device)

                if self.config.norm:
                    exog_future_np = exog_future.cpu().numpy()
                    exog_future_b = exog_future_np.shape[0]
                    scaled_exog_future = self.scaler2.transform(rearrange(exog_future_np, 'b l c->(b l) c'))
                    scaled_exog_future = rearrange(scaled_exog_future, '(b l) c -> b l c', b=exog_future_b)
                    exog_future = torch.tensor(scaled_exog_future).to(self.device)
            else:
                exog_future = None

            answers = self._perform_rolling_predictions(horizon, input_np, exog_future, series_dim)
            answers = torch.tensor(answers)[:, -horizon:, :series_dim]

            if self.config.norm:
                # Only inverse transform series data with scaler1
                answers_b = answers.shape[0]
                scaled_data = self.scaler1.inverse_transform(
                    rearrange(answers.cpu().detach().numpy(), 'b l c->(b l) c')
                )
                answers = rearrange(scaled_data, '(b l) c -> b l c', b=answers_b)

            return answers
        else:
            # Handle 2D/3D data (backward compatibility)
            if len(input_np.shape) == 2:
                input_np = input_np[:, :, np.newaxis]

            series_dim = input_np.shape[-2] if len(input_np.shape) == 3 else input_np.shape[-1]

            if input_data["covariates"] is None:
                covariates = {}
            else:
                covariates = input_data["covariates"]
            exog_data = covariates.get("exog")

            if exog_data is not None:
                if len(exog_data.shape) == 2:
                    exog_data = exog_data[:, :, np.newaxis]
                input_np = np.concatenate((input_np, exog_data), axis=1)
                if (
                        hasattr(self.config, "output_chunk_length")
                        and horizon != self.config.output_chunk_length
                ):
                    raise ValueError(
                        f"Error: 'exog' is enabled during training, but horizon ({horizon}) != output_chunk_length ({self.config.output_chunk_length}) during forecast."
                    )

            if self.config.norm:
                batch_size = input_np.shape[0]
                length = input_np.shape[1]
                channels = input_np.shape[2]

                # Reshape for scaler
                flattened = rearrange(input_np, 'b l c -> (b l) c')
                scaled = self.scaler1.transform(flattened)
                input_np = rearrange(scaled, '(b l) c -> b l c', b=batch_size)

            # Simplified batch prediction
            with torch.no_grad():
                predictions = []

                for batch_idx in range(input_np.shape[0]):
                    batch_input = input_np[batch_idx:batch_idx + 1]
                    input_tensor = torch.tensor(batch_input, dtype=torch.float32).to(self.device)

                    output = self.model(input_tensor)

                    if self.config.use_mlp and self.MLP is not None and exog_futures is not None:
                        exog_future = exog_futures[batch_idx, -horizon:, :]
                        exog_future_tensor = torch.tensor(exog_future, dtype=torch.float32).unsqueeze(0).to(self.device)

                        transformer_output = output[:, -horizon:, :series_dim]
                        output = self.MLP(torch.cat((transformer_output, exog_future_tensor), dim=-1))
                    else:
                        output = output[:, -horizon:, :series_dim]

                    predictions.append(output.cpu().numpy())

                answers = np.concatenate(predictions, axis=0)[:, -horizon:, :]

                if self.config.norm:
                    # Reshape and inverse transform
                    batch_size = answers.shape[0]
                    flattened = rearrange(answers, 'b l c -> (b l) c')
                    scaled = self.scaler1.inverse_transform(flattened)
                    answers = rearrange(scaled, '(b l) c -> b l c', b=batch_size)

                # Return only series dimensions
                if len(answers.shape) == 3 and answers.shape[2] == 1:
                    return answers[:, :, 0]
                else:
                    return answers[:, :, :series_dim]

    def _perform_rolling_predictions(
            self, horizon: int, input_np: np.ndarray, exog_future: torch.Tensor, series_dim: int
    ) -> np.ndarray:
        """
        Perform rolling predictions for 3D data.
        """
        rolling_time = 0
        answers = []

        with torch.no_grad():
            while not answers or sum(a.shape[1] for a in answers) < horizon:
                input_tensor = torch.tensor(input_np, dtype=torch.float32).to(self.device)
                output = self.model(input_tensor)

                if self.config.use_mlp and self.MLP is not None and exog_future is not None:
                    output = torch.tensor(output[:, -horizon:, :series_dim]).to(self.device)
                    output = self.MLP(torch.cat((output.to(torch.float32), exog_future.to(torch.float32)), dim=-1))
                else:
                    output = output[:, -horizon:, :series_dim]

                column_num = output.shape[-1]
                real_batch_size = output.shape[0]
                answer = (
                    output.cpu()
                    .numpy()
                    .reshape(real_batch_size, -1, column_num)[:, -self.config.horizon:, :]
                )
                answers.append(answer)
                if sum(a.shape[1] for a in answers) >= horizon:
                    break
                rolling_time += 1
                output = output.cpu().numpy()[:, -self.config.horizon:, :]
                input_np = self._get_rolling_data_3d(input_np, output, rolling_time)

        answers = np.concatenate(answers, axis=1)
        return answers[:, -horizon:, :]

    def _get_rolling_data_3d(
            self,
            input_np: np.ndarray,
            output: Optional[np.ndarray],
            rolling_time: int,
    ) -> np.ndarray:
        """
        Prepare rolling data for 3D input.
        """
        if rolling_time > 0:
            input_np = np.concatenate((input_np, output), axis=1)
            input_np = input_np[:, -self.config.seq_len:, :]
        return input_np