import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt

from config.settings import XGBOOST_PARAMS, MODELS_DIR


MODEL_PATH = MODELS_DIR / "xgboost_model.joblib"
SCALER_PATH = MODELS_DIR / "xgboost_scaler.joblib"


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    tune: bool = False,
) -> xgb.XGBClassifier:
    params = dict(XGBOOST_PARAMS)

    if tune:
        grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
        }
        base = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )
        gs = GridSearchCV(base, grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)
        print(f"Best params: {gs.best_params_}")
        model = gs.best_estimator_
    else:
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

    return model


def evaluate(
    model: xgb.XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    auc = roc_auc_score(y_test, y_prob)
    print(f"AUC-ROC: {auc:.4f}")

    if feature_names:
        plot_feature_importance(model, feature_names)

    return {"auc": auc, "y_pred": y_pred, "y_prob": y_prob}


def plot_feature_importance(
    model: xgb.XGBClassifier,
    feature_names: list[str],
    top_n: int = 20,
) -> None:
    importance = model.feature_importances_
    idx = np.argsort(importance)[-top_n:]
    plt.figure(figsize=(8, 6))
    plt.barh([feature_names[i] for i in idx], importance[idx])
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "xgb_feature_importance.png")
    plt.close()
    print(f"Feature importance saved → {MODELS_DIR / 'xgb_feature_importance.png'}")


def save(model: xgb.XGBClassifier, scaler=None) -> None:
    joblib.dump(model, MODEL_PATH)
    if scaler is not None:
        joblib.dump(scaler, SCALER_PATH)
    print(f"XGBoost model saved → {MODEL_PATH}")


def load() -> tuple[xgb.XGBClassifier, object | None]:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
    return model, scaler


def predict_proba(
    model: xgb.XGBClassifier,
    X: np.ndarray,
) -> np.ndarray:
    """상승 확률 반환 (0~1)."""
    return model.predict_proba(X)[:, 1]
