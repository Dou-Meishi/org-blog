import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_REPEATS = 5  # For objective function averaging
FAIL_THRESHOLD = 0.5    # Terminate early if test error too large


def get_dataloader(config):
    # Generate training data
    x_train = np.linspace(-4 * np.pi, 4 * np.pi, 800).reshape(-1, 1)
    y_train = np.sin(x_train)

    # Generate test data
    x_test = np.linspace(-4 * np.pi, 4 * np.pi, 199).reshape(-1, 1)
    y_test = np.sin(x_test)

    # Convert to PyTorch tensors
    train_data = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
    test_data = TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=199, shuffle=False)

    return train_loader, test_loader


def get_model_and_optimizer(config: dict):
    model = nn.Sequential(
        nn.Linear(1, config["hidden_size"]),
        nn.ReLU(),
        nn.Linear(config["hidden_size"], 1),
    ).to(device)

    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"]
    )
    return model, optimizer


def train_and_eval(model, optimizer, train_loader, test_loader, config):
    criterion = nn.MSELoss()
    model.train()

    for _ in range(config["num_epochs"]):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            if np.isnan(loss.item()):
                return np.nan

    # Evaluation
    model.eval()
    with torch.no_grad():
        x_test, y_test = next(iter(test_loader))
        x_test, y_test = x_test.to(device), y_test.to(device)
        predictions = model(x_test)
        test_error = criterion(predictions, y_test).item()

    return test_error


def objective(trial):
    config = {
        "batch_size": trial.suggest_int("batch_size", 16, 128, step=16),
        "hidden_size": trial.suggest_int("hidden_size", 64, 512, step=64),
        "lr": trial.suggest_float("lr", 5e-5, 5e-3, log=True),
        "momentum": trial.suggest_float("momentum", 0.8, 0.99),
        "num_epochs": trial.suggest_int("num_epochs", 500, 5000, step=500),
    }

    train_loader, test_loader = get_dataloader(config)
    total_error = 0.0

    for _ in range(NUM_REPEATS):
        model, optimizer = get_model_and_optimizer(config)
        test_error = train_and_eval(model, optimizer, train_loader, test_loader, config)
        total_error += test_error

        if test_error > FAIL_THRESHOLD:
            trial.set_user_attr("note", f"Fail due to too large error {test_error}")
            return np.nan
        elif np.isnan(test_error):
            trial.set_user_attr("note", f"Error is {test_error}")
            return np.nan

    return total_error / NUM_REPEATS
