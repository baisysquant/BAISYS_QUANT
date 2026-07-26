from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

_RETRAIN_FREQ = 60
_RETRAIN_EVERY = 20
_TRAIN_WINDOW = 120  # 最多用最近 120 天训练
_VAL_SPLIT = 0.8     # 训练窗口内 80% 训练，20% 验证（时序前 train 后 val）
_EARLY_STOP_ROUNDS = 15

_FEATURE_CN = [
    "MACD趋势分", "金叉信号分", "柱状动能分",
    "DIF斜评分", "背离信号分", "量价配合分", "K线形态分",
]

_EXTRA_FEATURES = ["atr_log", "ret_5d"]

_FEATURE_ALL = _FEATURE_CN + _EXTRA_FEATURES

_TARGET = "fwd_5d"
_TARGET_CLIP_LOW = 0.01    # 去极值底部分位数
_TARGET_CLIP_HIGH = 0.99   # 去极值顶部分位数

_HAS_XGB = False
try:
    import xgboost as xgb

    _HAS_XGB = True
except ImportError:
    pass


def _cross_sectional_rank(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """逐日对指定列做横截面排名，转为 [0, 1] 百分位。"""
    df = df.copy()
    for col in cols:
        ranks = df.groupby("trade_date")[col].rank(pct=True)
        df[col] = ranks.fillna(0.5).astype(np.float64)
    return df


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df["atr_log"] = np.log(df.get("ATR", 0).clip(lower=1e-8))
    for sym, grp in df.groupby("symbol"):
        idx = grp.index
        close = grp["close"]
        df.loc[idx, "ret_5d"] = close / close.shift(5).fillna(close.iloc[:6].mean()) - 1
        df.loc[idx, _TARGET] = close.shift(-5) / close - 1
    # 横截面标准化所有特征
    df = _cross_sectional_rank(df, _FEATURE_ALL)
    # 标签去极值
    low = df[_TARGET].quantile(_TARGET_CLIP_LOW)
    high = df[_TARGET].quantile(_TARGET_CLIP_HIGH)
    df[_TARGET] = df[_TARGET].clip(low, high)
    return df


def _cross_sectional_z(
    s: pd.Series, group: pd.Series | None = None,
) -> pd.Series:
    def _z(x: pd.Series) -> pd.Series:
        std = x.std()
        return (x - x.mean()) / std if std > 1e-8 else pd.Series(0.0, index=x.index)
    if group is not None and group.nunique() > 1:
        result = s.groupby(group).transform(_z)
    else:
        result = _z(s)
    return result.fillna(0).clip(-3, 3)


class BaseSignalModel:
    """Abstract base — shared interface for Ridge and XGBoost variants."""

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> bool:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


class RidgeSignalModel(BaseSignalModel):
    def __init__(self, l2_alpha: float = 0.5):
        self.l2_alpha = l2_alpha
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> bool:
        n, p = X.shape
        if n < p + 10:
            return False
        try:
            XTX = X.T @ X + self.l2_alpha * np.eye(p)
            XTy = X.T @ y
            self.coef_ = np.linalg.solve(XTX, XTy)
            return True
        except np.linalg.LinAlgError:
            return False

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            return np.zeros(X.shape[0])
        return X @ self.coef_


class XGBSignalModel(BaseSignalModel):
    def __init__(self, n_estimators: int = 500, max_depth: int = 4, lr: float = 0.05):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.lr = lr
        self._model: xgb.Booster | None = None  # type: ignore[name-defined]
        self._importances: np.ndarray | None = None
        self._best_iteration: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> bool:
        if not _HAS_XGB:
            return False
        n, p = X.shape
        if n < p + 10:
            return False
        try:
            dtrain = xgb.DMatrix(X, label=y)  # type: ignore[attr-defined]
            params = {
                "objective": "reg:squarederror",
                "max_depth": self.max_depth,
                "eta": self.lr,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "alpha": 0.5,
                "lambda": 2.0,
                "min_child_weight": 5,
                "n_jobs": -1,
                "verbosity": 0,
            }
            evals = [(dtrain, "train")]
            if X_val is not None and y_val is not None and len(y_val) >= 20:
                dval = xgb.DMatrix(X_val, label=y_val)  # type: ignore[attr-defined]
                evals.append((dval, "eval"))
            self._model = xgb.train(
                params, dtrain, num_boost_round=self.n_estimators,
                evals=evals,
                early_stopping_rounds=_EARLY_STOP_ROUNDS if len(evals) > 1 else None,
                verbose_eval=False,
            )  # type: ignore[attr-defined]
            fs = self._model.get_score(importance_type="weight")
            self._importances = np.array([fs.get(f"f{i}", 0) for i in range(p)], dtype=float)
            if self._importances.sum() > 0:
                self._importances /= self._importances.sum()
            self._best_iteration = getattr(self._model, "best_iteration", self.n_estimators)
            return True
        except Exception:
            return False

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            return np.zeros(X.shape[0])
        d = xgb.DMatrix(X)  # type: ignore[attr-defined]
        return self._model.predict(d)

    def feature_importances(self) -> list[tuple[str, float]]:
        if self._importances is None:
            return []
        return sorted(zip(_FEATURE_ALL, self._importances), key=lambda x: -x[1])


def _pick_model() -> BaseSignalModel:
    if _HAS_XGB:
        logger.info("[ML] 使用 XGBoost 信号模型")
        return XGBSignalModel()
    logger.info("[ML] XGBoost 不可用，回退 Ridge 信号模型")
    return RidgeSignalModel()


def apply_ml_signal(merged: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward ML model to replace entry_score with ML prediction.

    Uses XGBoost when available, falls back to Ridge regression.
    Retrains every _RETRAIN_EVERY days on a rolling _TRAIN_WINDOW window.
    """
    required = [c for c in _FEATURE_CN if c in merged.columns]
    if len(required) < 3:
        logger.info("[ML] 不足 3 个特征列，跳过 ML 信号模型")
        return merged

    if "close" not in merged.columns:
        logger.info("[ML] 缺少 close 列，跳过 ML 信号模型")
        return merged

    df = _compute_features(merged)

    dates = sorted(df["trade_date"].unique())
    if len(dates) < _RETRAIN_FREQ + 10:
        logger.info(f"[ML] 交易日数 {len(dates)} < {_RETRAIN_FREQ + 10}，跳过 ML 信号模型")
        return merged

    model = _pick_model()
    last_train_date_idx: int | None = None
    total_predicted = 0

    for i, date in enumerate(dates):
        if i < _RETRAIN_FREQ:
            continue

        today_mask = df["trade_date"] == date

        should_retrain = (
            last_train_date_idx is None
            or i - last_train_date_idx >= _RETRAIN_EVERY
        )

        if should_retrain:
            cut_idx = max(0, i - _TRAIN_WINDOW)
            window_dates = dates[cut_idx:i]
            split = int(len(window_dates) * _VAL_SPLIT)
            train_dates = window_dates[:split]
            val_dates = window_dates[split:]
            train_mask = df["trade_date"].isin(train_dates)
            val_mask = df["trade_date"].isin(val_dates)
            train_X = df.loc[train_mask, _FEATURE_ALL].values.astype(np.float64)
            train_y_raw = df.loc[train_mask, _TARGET].values.astype(np.float64)
            train_ok = np.isfinite(train_y_raw) & np.all(np.isfinite(train_X), axis=1)
            val_X = df.loc[val_mask, _FEATURE_ALL].values.astype(np.float64) if len(val_dates) > 0 else None
            val_y = df.loc[val_mask, _TARGET].values.astype(np.float64) if val_X is not None else None
            val_ok = np.isfinite(val_y) & np.all(np.isfinite(val_X), axis=1) if val_X is not None else None
            if train_ok.sum() < 20:
                continue
            X_val = val_X[val_ok] if val_ok is not None and val_ok.sum() >= 20 else None
            y_val = val_y[val_ok] if val_ok is not None and val_ok.sum() >= 20 else None
            if not model.fit(train_X[train_ok], train_y_raw[train_ok], X_val, y_val):
                continue
            last_train_date_idx = i

        elif last_train_date_idx is None:
            continue

        today_X = df.loc[today_mask, _FEATURE_ALL].values.astype(np.float64)
        today_valid = np.all(np.isfinite(today_X), axis=1)
        if today_valid.sum() == 0:
            continue

        today_idx = df.index[today_mask]
        pred_raw = model.predict(today_X)
        pred_s = pd.Series(pred_raw, index=today_idx)
        pred_std = pred_s.std()
        if pred_std < 1e-6:
            continue
        pred_z = _cross_sectional_z(
            pred_s,
            df.loc[today_idx, "行业"] if "行业" in df.columns else None,
        )
        entry_raw = (pred_z * 16.67 + 50).clip(0, 100)
        df.loc[today_idx, "进场评分"] = entry_raw.values
        df.loc[today_idx, "退出评分"] = (100.0 - entry_raw).values
        total_predicted += today_valid.sum()

        if i % max(1, len(dates) // 10) == 0:
            logger.info(f"[ML] 进度 {date} — 已预测 {total_predicted} 行")

    if last_train_date_idx is None:
        logger.warning("[ML] 模型训练失败，使用原始 MACD 评分")
        return merged

    logger.info(f"[ML] Walk-forward 完成，共预测 {total_predicted} 行")
    if isinstance(model, XGBSignalModel):
        fi = model.feature_importances()
        fi_str = " | ".join(f"{n}: {v:.1%}" for n, v in fi[:5])
        logger.info(f"[ML] 特征重要性 Top5: {fi_str}")
        if model._best_iteration:
            logger.info(f"[ML] XGBoost 早停: best_iteration={model._best_iteration}")
    return df
