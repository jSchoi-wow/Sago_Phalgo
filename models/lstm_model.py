import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from config.settings import (
    LSTM_SEQ_LEN, LSTM_FEATURES, LSTM_EPOCHS, LSTM_BATCH_SIZE,
    LSTM_UNITS, LSTM_DROPOUT, LSTM_BUY_THRESHOLD, MODELS_DIR,
)

MODEL_PATH = str(MODELS_DIR / "lstm_model.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMNet(nn.Module):
    def __init__(self, n_features: int, hidden1: int, hidden2: int, dropout: float):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, hidden1, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc1   = nn.Linear(hidden2, 32)
        self.relu  = nn.ReLU()
        self.fc2   = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out[:, -1, :])  # 마지막 타임스텝
        out = self.relu(self.fc1(out))
        return self.fc2(out).squeeze(-1)


def build_model(n_features: int = len(LSTM_FEATURES)) -> LSTMNet:
    return LSTMNet(n_features, LSTM_UNITS[0], LSTM_UNITS[1], LSTM_DROPOUT).to(DEVICE)


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray | None = None,
    y_val:   np.ndarray | None = None,
    epochs:  int = LSTM_EPOCHS,
    batch_size: int = LSTM_BATCH_SIZE,
) -> LSTMNet:
    model = build_model(n_features=X_train.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    has_val = X_val is not None and len(X_val) > 0
    if has_val:
        Xv = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
        yv = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)

    best_val_loss = float("inf")
    patience_cnt  = 0
    best_state    = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(Xt)

        if has_val:
            model.eval()
            with torch.no_grad():
                val_pred = model(Xv)
                val_loss = criterion(val_pred, yv).item()
            scheduler.step(val_loss)
            print(f"Epoch {epoch:3d}/{epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_cnt  = 0
            else:
                patience_cnt += 1
                if patience_cnt >= 10:
                    print(f"Early stopping at epoch {epoch}")
                    break
        else:
            print(f"Epoch {epoch:3d}/{epochs}  train_loss={train_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def evaluate(model: LSTMNet, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    model.eval()
    Xt = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        y_pred = model(Xt).cpu().numpy()

    mse  = float(np.mean((y_pred - y_test) ** 2))
    mae  = float(np.mean(np.abs(y_pred - y_test)))
    dacc = float(np.mean(np.sign(y_pred) == np.sign(y_test)))
    print(f"LSTM  MSE: {mse:.6f} | MAE: {mae:.6f} | Directional Acc: {dacc:.4f}")
    return {"mse": mse, "mae": mae, "directional_acc": dacc, "y_pred": y_pred}


def save(model: LSTMNet) -> None:
    torch.save(model.state_dict(), MODEL_PATH)
    # 모델 구조 정보도 함께 저장
    torch.save({
        "state_dict": model.state_dict(),
        "n_features": model.lstm1.input_size,
        "hidden1": model.lstm1.hidden_size,
        "hidden2": model.lstm2.hidden_size,
        "dropout": LSTM_DROPOUT,
    }, MODEL_PATH)
    print(f"LSTM model saved -> {MODEL_PATH}")


def load() -> LSTMNet:
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = LSTMNet(
        n_features=ckpt["n_features"],
        hidden1=ckpt["hidden1"],
        hidden2=ckpt["hidden2"],
        dropout=ckpt["dropout"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def predict_return(model: LSTMNet, X: np.ndarray) -> np.ndarray:
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        return model(Xt).cpu().numpy()


def to_signal(predicted_returns: np.ndarray, threshold: float = LSTM_BUY_THRESHOLD) -> np.ndarray:
    signals = np.zeros(len(predicted_returns), dtype=int)
    signals[predicted_returns >  threshold] =  1
    signals[predicted_returns < -threshold] = -1
    return signals
