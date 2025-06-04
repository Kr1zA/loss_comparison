#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

import pickle

import optuna
from optuna.trial import TrialState

import typing

import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim

import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio

from sklearn.preprocessing import StandardScaler
from copy import deepcopy
import tqdm
import time

from sklearn.preprocessing import MinMaxScaler

import plotly.io as pio

pio.orca.config.use_xvfb = True

pio.orca.config.executable = "path/orca"

import os

os.environ["WANDB_SILENT"] = "true"

from datetime import datetime, timedelta

torch.set_printoptions(precision=8)

import sys

current_dir = os.getcwd()

parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

sys.path.append(parent_dir)

import losses


# In[3]:
import argparse



# In[21]:
parser = argparse.ArgumentParser(description="Load parameters for the script.")

parser.add_argument("--cuda_device", type=int, default=0, help="CUDA device index.")
parser.add_argument("--trial_loss", type=str, default="angle_loss", help="Loss function to use.")
args = parser.parse_args()


# set WANDB to False, if you do not want to use wandb for testing purposes,
# or if you do not want to use wandb at all, you should change code to store results in a different way
WANDB = True
WANDB_PROJECT = "loss_security"
LAGS = 384
FUTURE = 1
CRITERION = "Port 445"
MODELS_PATH = "models/"
# create folder for your models if it does not exist
if not os.path.exists("models/"):
    os.makedirs("models/")
EPOCHS = 200
TRIALS = 200
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

CUDA_DEVICE = args.cuda_device
TRIAL_LOSS = args.trial_loss


# In[4]:


loss_functions = {
    "mae": losses.mae,
    "mse": losses.mse,
    "huber": losses.huber,
    "angle_loss": losses.angle_loss,
    "rmse": losses.rmse,
    "nrmse": losses.nrmse,
    "rrmse": losses.rrmse,
    "msle": losses.msle,
    "rmsle": losses.rmsle,
    "mase": losses.mase,
    "rmsse": losses.rmsse,
    "poisson": losses.poisson,
    "logCosh": losses.logCosh,
    "mape": losses.mape,
    "mbe": losses.mbe,
    "rae": losses.rae,
    "rse": losses.rse,
    "kernelMSE": losses.kernelMSE,
    "quantile25": losses.quantile25,
    "quantile75": losses.quantile75,
}


# In[5]:


device = torch.device(CUDA_DEVICE)
# device = torch.device("cpu")


# In[6]:


def create_lags_and_future(
    data: pd.DataFrame, lags: int = 1, future: int = 1
) -> pd.DataFrame:
    if lags < future:
        raise BaseException("Make lags bigger than future please")

    data_with_lags_and_future = data.copy()

    # insert one lag unscaled for MASE criterion
    data_with_lags_and_future.insert(
        1,
        f"{CRITERION}_lag_1",
        data_with_lags_and_future[[f"{CRITERION}"]].shift(1),
    )

    if future > 1:
        for fut in range(future, 1, -1):
            fut_column_name_scaled = f"{CRITERION}_scaled_fut_{fut}"
            lag_column_data_scaled = data_with_lags_and_future[
                [f"{CRITERION}_scaled"]
            ].shift(-(fut - 1))
            data_with_lags_and_future.insert(
                0, fut_column_name_scaled, lag_column_data_scaled
            )

            fut_column_name = f"{CRITERION}_fut_{fut}"
            lag_column_data = data_with_lags_and_future[[CRITERION]].shift(-(fut - 1))
            data_with_lags_and_future.insert(1, fut_column_name, lag_column_data)

    for lag in range(lags, 0, -1):
        lag_column_number = len(data_with_lags_and_future.columns)
        lag_column_name = f"{CRITERION}_scaled_lag_{lag}"
        lag_column_data = data_with_lags_and_future[[f"{CRITERION}_scaled"]].shift(
            lag
        )
        data_with_lags_and_future.insert(
            lag_column_number, lag_column_name, lag_column_data
        )

    # return data_with_lags_and_future.dropna()
    return data_with_lags_and_future


# In[7]:

# dataset will be published in the future and link added to repository
data = pd.read_csv("data/Port 445.csv.xz", index_col=0, compression="xz")
data.index = pd.to_datetime(data.index)




# In[8]:


sumed_data = pd.DataFrame(data.sum(axis=1, min_count=len(data.columns)), columns=[CRITERION])


# In[9]:


# change to 30 min time period
# this is the recommended way to change the time period
time = 30
sumed_data_30m = sumed_data.resample(rule=f'{time}min', label="left").sum(min_count=time)


# In[10]:


data = sumed_data_30m

# removed down peeks / almost zero data
data = data.where(data >= 50, np.nan)



# In[14]:


split_index = datetime.strptime("2018-11-21 21:00:00", DATE_FORMAT)

scaler = MinMaxScaler()
fut = timedelta(minutes=30 * FUTURE)
scaler.fit(data.loc[:str(split_index - fut)])

data.insert(len(data.columns), f"{CRITERION}_scaled", scaler.transform(data))


# In[16]:


data_lagged_and_futured = create_lags_and_future(data, LAGS, FUTURE)

futures = [CRITERION]
for fu in range(2, FUTURE + 1, 1):
    futures.append(f"{CRITERION}_fut_{fu}")



# In[17]:


train_df_lagged_and_futured, test_df_lagged_and_futured = (
    data_lagged_and_futured.dropna().loc[:str(split_index - fut)],
    data_lagged_and_futured.dropna().loc[str(split_index) :],
)





# In[23]:


X_train_columns = [f"{CRITERION}_scaled_lag_{i}" for i in range(LAGS, 0, -1)]

X_train = train_df_lagged_and_futured[X_train_columns]
y_train = train_df_lagged_and_futured[[f"{CRITERION}_scaled"]]
X_test = test_df_lagged_and_futured[X_train_columns]
y_test = test_df_lagged_and_futured[[f"{CRITERION}_scaled"]]



# In[29]:


X_train = torch.tensor(X_train.values.reshape((-1, LAGS, 1))).float()
y_train = torch.tensor(y_train.values.reshape((-1, 1))).float()
X_test = torch.tensor(X_test.values.reshape((-1, LAGS, 1))).float()
y_test = torch.tensor(y_test.values.reshape((-1, 1))).float()




# In[ ]:


# preparing values for computing some loss functions

avg_diff_scaled = (
    (
        train_df_lagged_and_futured[f"{CRITERION}_scaled"]
        - train_df_lagged_and_futured[f"{CRITERION}_scaled_lag_1"]
    )
    .abs()
    .mean()
)
avg_diff = (
    (
        train_df_lagged_and_futured[f"{CRITERION}"]
        - train_df_lagged_and_futured[f"{CRITERION}_lag_1"]
    )
    .abs()
    .mean()
)
avg_sq_train_diff_scaled = (
    (
        train_df_lagged_and_futured[f"{CRITERION}_scaled"]
        - train_df_lagged_and_futured[f"{CRITERION}_scaled_lag_1"]
    )
    ** 2
).mean()
mean_train_scaled = train_df_lagged_and_futured[f"{CRITERION}_scaled"].mean()
norm_factor_max_min_scaled = (
    train_df_lagged_and_futured[f"{CRITERION}_scaled"].max()
    - train_df_lagged_and_futured[f"{CRITERION}_scaled"].min()
)



# In[24]:


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)


# In[25]:


class LSTM(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_stacked_layers, output_size, dropout_rate
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_stacked_layers,
            batch_first=True,
            dropout=dropout_rate,
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# In[ ]:


def train_one_epoch(
    model, optimizer: torch.optim, loader: DataLoader, loss
) -> typing.Tuple[float, float]:
    model.train(True)
    epoch_loss = 0.0
    epoch_metrics = 0.0

    for batch_index, batch in enumerate(loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        output = model(x_batch)

        match loss:
            case losses.mae | losses.mse | losses.rmse | losses.rrmse | losses.msle | losses.rmsle | losses.poisson | losses.logCosh | losses.mape | losses.mbe | losses.kernelMSE:
                batch_loss = loss(output, y_batch)
            case losses.huber:
                batch_loss = loss(output, y_batch, delta=mean_train_scaled/2)
            case losses.angle_loss:
                batch_loss = loss(output, y_batch, x_batch[:, -1, :], avg_diff_scaled)
            case losses.nrmse:
                batch_loss = loss(output, y_batch, mean_train_scaled)
            case losses.mase:
                batch_loss = loss(output, y_batch, avg_diff_scaled)
            case losses.rmsse:
                batch_loss = loss(output, y_batch, avg_sq_train_diff_scaled)
            case losses.rae:
                batch_loss = loss(output, y_batch, mean_train_scaled)
            case losses.rse:
                batch_loss = loss(output, y_batch, mean_train_scaled)
            case losses.quantile25:
                batch_loss = loss(output, y_batch)
            case losses.quantile75:
                batch_loss = loss(output, y_batch)

        batch_metrics = nn.functional.l1_loss(output, y_batch)

        epoch_loss += batch_loss.item() * len(x_batch)
        epoch_metrics += batch_metrics.item() * len(x_batch)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    epoch_loss = epoch_loss / len(train_dataset)
    epoch_metrics = epoch_metrics / len(train_dataset)
    return epoch_loss, epoch_metrics


# In[ ]:


def validate_one_epoch(model, loader: DataLoader, loss) -> typing.Tuple[float, float]:
    model.train(False)
    running_loss = 0.0
    running_metrics = 0.0

    for batch_index, batch in enumerate(loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(x_batch)

            match loss:
                case losses.mae | losses.mse | losses.rmse | losses.rrmse | losses.msle | losses.rmsle | losses.poisson | losses.logCosh | losses.mape | losses.mbe | losses.kernelMSE:
                    batch_loss_function = loss(output, y_batch)
                case losses.huber:
                    batch_loss_function = loss(output, y_batch, delta=mean_train_scaled/2)
                case losses.angle_loss:
                    batch_loss_function = loss(output, y_batch, x_batch[:, -1, :], avg_diff_scaled)
                # case losses.nrmse:
                #     batch_loss_function = loss(output, y_batch, norm_factor_max_min_scaled)
                case losses.nrmse:
                    batch_loss_function = loss(output, y_batch, mean_train_scaled)
                case losses.mase:
                    batch_loss_function = loss(output, y_batch, avg_diff_scaled)
                case losses.rmsse:
                    batch_loss_function = loss(output, y_batch, avg_sq_train_diff_scaled)
                case losses.rae:
                    batch_loss_function = loss(output, y_batch, mean_train_scaled)
                case losses.rse:
                    batch_loss_function = loss(output, y_batch, mean_train_scaled)
                case losses.quantile25:
                    batch_loss_function = loss(output, y_batch)
                case losses.quantile75:
                    batch_loss_function = loss(output, y_batch)

            metrics = nn.functional.l1_loss(output, y_batch)

            running_loss += batch_loss_function.item() * len(x_batch)
            running_metrics += metrics.item() * len(x_batch)

    avg_loss_across_batches = running_loss / len(test_dataset)
    avg_metrics_across_batches = running_metrics / len(test_dataset)

    return avg_loss_across_batches, avg_metrics_across_batches


# In[28]:


def predict_values(model, loader: DataLoader) -> typing.Tuple[float]:
    model.train(False)

    output = torch.empty((0)).to(torch.device("cpu")).detach()
    for batch_index, batch in enumerate(loader):
        x_batch, _ = batch[0].to(device), batch[1].to(device)
        output = torch.cat((output, model(x_batch).to(torch.device("cpu")).detach()), 0)

    return output


# In[ ]:


def objective(trial):

    config = {
        "batch_size": trial.suggest_int("batch_size", 4, 256),
        "shuffle_dataset": trial.suggest_categorical("shuffle_dataset", [True, False]),
        "epochs": EPOCHS,
        "input_size": 1,
        "hidden_size": trial.suggest_int("hidden_size", 4, 1024),
        "num_stacked_layers": trial.suggest_int("num_stacked_layers", 1, 8),
        "output_size": 1,
        "dropout": trial.suggest_float("dropout", 0, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True),
        "loss": TRIAL_LOSS,
        "optimizer": trial.suggest_categorical(
            "optimizer",
            [
                "Adam",
                # "RMSprop",
                #  "SGD"
            ],
        ),
        "CUDA_DEVICE": CUDA_DEVICE,
    }
    config["architecture"] = (
        f"LSTM_{config['num_stacked_layers']}_{config['hidden_size']}"
    )

    if config["num_stacked_layers"] == 1:
        config["dropout"] = 0

    actual_model_name = (
        f"r_{config['loss']}_hs_{config['hidden_size']}_sl_{config['num_stacked_layers']}_sd_{config['shuffle_dataset']}_bs_{config['batch_size']}_lr_{config['learning_rate']}_dr_{config['dropout']}"
    )

    if WANDB:
        # Initialize wandb
        wandb.init(
            project=WANDB_PROJECT, name=actual_model_name, config=config
        )

    batch_size = config["batch_size"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=config["shuffle_dataset"],
          num_workers=4,
          prefetch_factor=3
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
         num_workers=4,
         prefetch_factor=3
    )

    model = LSTM(
        config["input_size"],
        config["hidden_size"],
        config["num_stacked_layers"],
        config["output_size"],
        config["dropout"],
    )
    model.to(device)

    optimizer = getattr(optim, config["optimizer"])(
        model.parameters(), lr=config["learning_rate"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5)

    best_model_state = deepcopy(model.state_dict())
    best_val_metrics = 999999999

    # custom early stopping
    best_epoch = 0

    for epoch in range(config["epochs"]):
        train_loss, train_metrics = train_one_epoch(
            model,
            optimizer=optimizer,
            loader=train_loader,
            loss=loss_functions[config["loss"]],
        )
        val_loss, val_metrics = validate_one_epoch(
            model, loader=test_loader, loss=loss_functions[config["loss"]]
        )

        # remmembering best model
        if val_metrics < best_val_metrics:
            best_epoch = epoch
            best_val_metrics = val_metrics
            best_model_state = deepcopy(model.state_dict())

        trial.report(val_metrics, epoch + 1)
        if WANDB:
            wandb.log(
                {
                    "train_metrics": train_metrics,
                    "train_loss": train_loss,
                    "val_metrics": val_metrics,
                    "val_loss": val_loss,
                }
            )

        # early stoping
        if epoch - best_epoch > 30:
            if WANDB:
                wandb.log({"early_stopped": True})
            break

        # if loss is nan
        if epoch > 9:
            if math.isnan(train_loss):
                if WANDB:
                    wandb.log({"loss_is_nan": True})
                break

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            if WANDB:
                wandb.log({"pruned": True})
            #     wandb.finish(quiet=True)
            # raise optuna.exceptions.TrialPruned()
            break

        scheduler.step(train_metrics)
    if WANDB:
        torch.save(best_model_state, f"{MODELS_PATH}{actual_model_name}_{wandb.run.id}.pt")
        model.load_state_dict(
            torch.load(
                f"{MODELS_PATH}{actual_model_name}_{wandb.run.id}.pt", weights_only=True
            )
        )
    else:
        torch.save(best_model_state, f"{MODELS_PATH}{actual_model_name}.pt")
        model.load_state_dict(
            torch.load(
                f"{MODELS_PATH}{actual_model_name}.pt", weights_only=True
            )
        )

    predicted_scaled = predict_values(model, test_loader)

    predicted_unscaled = scaler.inverse_transform(predicted_scaled)
    comparing_dataframe = test_df_lagged_and_futured[futures].copy(deep=True)
    comparing_dataframe.insert(0, f"{CRITERION}_predicted", predicted_unscaled)
    comparing_dataframe.insert(0, f"num_index", range(len(comparing_dataframe)))

    metrics = {}
    output = comparing_dataframe[[f"{CRITERION}_predicted"]].values
    target = comparing_dataframe[[CRITERION]].values
    metrics["MASE"] = loss_functions["mase"](
        torch.tensor(output), torch.tensor(target), avg_diff
    )
    metrics["MAE"] = loss_functions["mae"](torch.tensor(output), torch.tensor(target))

    if WANDB:
        wandb.log({"best_epoch": best_epoch})
        wandb.log(metrics)
        wandbTable = wandb.Table(dataframe=comparing_dataframe.reset_index())
        wandb.log({"comparing_dataframe": wandbTable})
        wandb.finish(quiet=True)

    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return val_metrics


# In[ ]:


study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.PatientPruner(
        optuna.pruners.MedianPruner(
            n_startup_trials=10, n_warmup_steps=10, n_min_trials=108
        ),
        patience=20,
    ),
)

batch_size_trials = [4, 16, 64]
shuffle_trials = [
    True,
    #   False
]
hidden_size_trials = [
    128,
    256,
    # 512
]
num_stacked_layers_trials = [1, 2, 4]
dropout_trials = [0, 0.25]
learning_rate_trials = [
    # 0.01,
    0.005,
    0.001,
    0.0005,
]
loss_trials = [TRIAL_LOSS]
optimizer_trials = [
    "Adam",
    # "RMSprop",
    # "SGD"
]

for bs in batch_size_trials:
    for sh in shuffle_trials:
        for hs in hidden_size_trials:
            for nsl in num_stacked_layers_trials:
                for dr in dropout_trials:
                    for lr in learning_rate_trials:
                        for lo in loss_trials:
                            for op in optimizer_trials:
                                study.enqueue_trial(
                                    {
                                        "batch_size": bs,
                                        "shuffle_dataset": sh,
                                        "hidden_size": hs,
                                        "num_stacked_layers": nsl,
                                        "dropout": dr,
                                        "learning_rate": lr,
                                        "loss": lo,
                                        "optimizer": op,
                                    }
                                )


# In[31]:


study.optimize(objective, n_trials=TRIALS)

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

