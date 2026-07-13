import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data_processing import set_seed
from .models import make_model


def evaluate(model, x, y, scaler):
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x)).cpu().numpy()
    pred_real = scaler.inverse_transform(pred.reshape(-1, 1)).reshape(pred.shape)
    y_real = scaler.inverse_transform(y.reshape(-1, 1)).reshape(y.shape)
    mse = float(np.mean((pred_real - y_real) ** 2))
    mae = float(np.mean(np.abs(pred_real - y_real)))
    return mse, mae, pred_real, y_real


def train_one(model_name, horizon, seed, arrays, epochs, batch_size, lr):
    train_x, train_y, test_x, test_y, y_scaler, _, test_dates = arrays
    set_seed(seed)
    model = make_model(model_name, train_x.shape[-1], horizon)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_state = None
    best_loss = float("inf")
    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(ds)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    mse, mae, pred, truth = evaluate(model, test_x, test_y, y_scaler)
    return {"mse": mse, "mae": mae, "pred": pred, "truth": truth, "dates": test_dates}

