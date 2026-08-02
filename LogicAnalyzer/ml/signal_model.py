from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr

_RETRAIN_FREQ = 60
_RETRAIN_EVERY = 20
_TRAIN_WINDOW = 120  # 最多用最近 120 天训练
_VAL_SPLIT = 0.8     # 训练窗口内 80% 训练，20% 验证（时序前 train 后 val）
_PURGE_DAYS = 5      # train 末尾清洗天数，防 fwd_5d 数据泄露
_EARLY_STOP_ROUNDS = 15
_N_SHUFFLE = 40      # label shuffle 检验洗牌次数
_MIN_VAL_IC = 0.02   # 验证集 Rank IC 最低阈值，低于此值视为无信号
_ML_SIGNAL_ENABLED = True   # ML 信号主开关：True=启用 XGBoost 覆写（内置 label shuffle 检验，不达标自动回退）

_FEATURE_CN = [
    "MACD趋势分", "金叉信号分", "柱状动能分",
    "DIF斜评分", "背离信号分", "量价配合分", "K线形态分",
]

_FEATURE_ALL = [
    # 保留验证过的 CN 特征（4 个 IC 相对较高的）
    "MACD趋势分", "金叉信号分", "DIF斜评分", "量价配合分",
    # 收益率族（5 个不同周期）
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    # 波动率族（7 个）
    "atr_log", "atr_ratio", "hl_ratio", "gap_abs",
    "ret_1d_std_20d", "ret_5d_std",
    "max_drawdown_20d", "max_ret_20d",
    # 流动性族（4 个）
    "amt_20d", "amt_ratio_5_20", "volume_shock", "illiquidity_5d",
    # 微观结构族（6 个）
    "close_position", "vwap_dev", "price_pos_ma20", "price_pos_ma60",
    "up_ratio_20d", "consec_up_days",
    # 截面相对族（1 个，需全面板计算）
    "rel_strength_20d",
    # 市场状态特征（3 个，需全面板聚合）
    "regime_trend", "regime_vol", "regime_dispersion",
]

_TARGET = "fwd_5d"

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
    has_atr = "ATR" in df.columns
    if has_atr:
        df["atr_log"] = np.log(df["ATR"].clip(lower=1e-8))
        df["atr_ratio"] = df["ATR"] / (df["close"] + 1e-8)

    for sym, grp in df.groupby("symbol"):
        idx = grp.index
        c = grp["close"]
        h = grp["high"]
        l = grp["low"]
        v = grp["volume"]
        a = grp["amount"]

        # ── 收益率族 ──
        prev_close = c.shift(1)
        df.loc[idx, "ret_1d"] = c / prev_close - 1
        df.loc[idx, "ret_5d"] = c / c.shift(5) - 1
        df.loc[idx, "ret_10d"] = c / c.shift(10) - 1
        df.loc[idx, "ret_20d"] = c / c.shift(20) - 1
        df.loc[idx, "ret_60d"] = c / c.shift(60) - 1

        # ── 波动率族 ──
        hl = (h - l) / (c + 1e-8)
        df.loc[idx, "hl_ratio"] = hl
        df.loc[idx, "gap_abs"] = abs(c / prev_close - 1)
        ret_1d = c.pct_change()
        df.loc[idx, "ret_1d_std_20d"] = ret_1d.rolling(20).std()
        df.loc[idx, "ret_5d_std"] = ret_1d.rolling(5).std()
        df.loc[idx, "max_drawdown_20d"] = c / c.rolling(20).max() - 1
        df.loc[idx, "max_ret_20d"] = c.rolling(20).max() / c.shift(20) - 1

        # ── 流动性族 ──
        df.loc[idx, "amt_20d"] = a.rolling(20).mean()
        amt_5d = a.rolling(5).mean()
        df.loc[idx, "amt_ratio_5_20"] = amt_5d / df.loc[idx, "amt_20d"]
        df.loc[idx, "volume_shock"] = v / v.rolling(20).median().clip(lower=1)
        abs_ret_1d = abs(ret_1d)
        df.loc[idx, "illiquidity_5d"] = (abs_ret_1d / (a / 1e8 + 1e-8)).rolling(5).mean()

        # ── 微观结构族 ──
        df.loc[idx, "close_position"] = (c - l) / (h - l + 1e-8)
        vwap = a / (v + 1e-8)
        df.loc[idx, "vwap_dev"] = c / vwap - 1
        df.loc[idx, "price_pos_ma20"] = c / c.rolling(20).mean() - 1
        df.loc[idx, "price_pos_ma60"] = c / c.rolling(60).mean() - 1
        df.loc[idx, "up_ratio_20d"] = (c > prev_close).rolling(20).mean()

        cum_up = (c > prev_close).astype(int)
        consec = cum_up.groupby((cum_up != cum_up.shift(1)).cumsum()).cumsum()
        df.loc[idx, "consec_up_days"] = consec

        # ── 目标变量（原始值，后续转截面排名） ──
        df.loc[idx, "fwd_5d_raw"] = c.shift(-5) / c - 1

    # ── 截面相对 + 市场状态特征（需全面板） ──
    for date, grp in df.groupby("trade_date"):
        if len(grp) < 5:
            continue
        date_idx = grp.index
        mkt_ret_20d = grp["ret_20d"].median()
        df.loc[date_idx, "rel_strength_20d"] = grp["ret_20d"] - mkt_ret_20d
        df.loc[date_idx, "regime_trend"] = mkt_ret_20d
        df.loc[date_idx, "regime_vol"] = grp["ret_1d"].std()
        df.loc[date_idx, "regime_dispersion"] = grp["ret_20d"].std()

    # 横截面标准化所有特征
    df = _cross_sectional_rank(df, _FEATURE_ALL)

    # 目标变量：截面排名收益率（选股能力）
    df[_TARGET] = df.groupby("trade_date")["fwd_5d_raw"].rank(pct=True)
    df.drop(columns=["fwd_5d_raw"], inplace=True)
    return df


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


def _label_shuffle_test(
    model: BaseSignalModel, X_val: np.ndarray, y_val: np.ndarray,
    n_shuffle: int = _N_SHUFFLE,
) -> tuple[float, float]:
    """Label shuffle 检验：比较验证集预测 IC 与随机标签的 IC 分布。

    Returns:
        (real_ic, p_value) — p_value < 0.05 表示模型有统计显著预测能力。
    """
    pred = model.predict(X_val)
    real_ic = float(spearmanr(pred, y_val)[0])
    shuffled_ics: list[float] = []
    for _ in range(n_shuffle):
        y_shuff = np.random.permutation(y_val)
        shuffled_ics.append(float(spearmanr(pred, y_shuff)[0]))
    count = sum(1 for s in shuffled_ics if s >= real_ic)
    p_value = (count + 1) / (n_shuffle + 1)
    return real_ic, p_value


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
    if not _ML_SIGNAL_ENABLED:
        logger.info("[ML] ML 信号已禁用（_ML_SIGNAL_ENABLED=False），使用 MACD 原生评分")
        return merged

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
    model_valid = False   # label shuffle 检验通过才允许覆写评分，否则自动回退原生 MACD 评分
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
            purge = min(_PURGE_DAYS, split)
            train_dates = window_dates[:split - purge]
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
            # Label shuffle 检验：不显著（p>=0.05 或 IC 过低）自动回退原生评分
            if X_val is not None and len(y_val) >= 50:
                ic_val, p_val = _label_shuffle_test(model, X_val, y_val)
                model_valid = (p_val < 0.05) and (ic_val >= _MIN_VAL_IC)
                logger.info(f"[ML] 验证集 IC={ic_val:.4f}  label shuffle p={p_val:.4f}" +
                            (" ✓显著" if model_valid else " ✗不显著，回退原生评分"))
            else:
                model_valid = False
                logger.info("[ML] 验证集样本不足，回退原生评分")

        elif last_train_date_idx is None:
            continue

        if not model_valid:
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
        entry_rank = pred_s.rank(pct=True) * 99 + 1  # [1, 100]
        df.loc[today_idx, "进场评分"] = entry_rank.values
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
