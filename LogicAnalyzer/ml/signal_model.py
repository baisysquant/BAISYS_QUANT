"""
Walk-forward 机器学习信号模型（XGBoost / Ridge）

标签合规（1.1 标签前瞻性清除）：
  策略约定为尾盘下单（信号当日生成、当日收盘价成交），因此标签收益率
  的起点必须是明日收盘价：fwd_5d_raw = C_{t+5} / C_{t+1} - 1，
  禁止以今日收盘价及更早价格作为标签起点。
  训练/验证按 80/20 时序切分，训练窗口尾端 purge 5 天（= 标签持有期），
  确保训练与验证标签所引用的价格区间不重叠。

特征窗口合规（1.2 特征未来函数阻断）：
  特征在 T 行的取值只使用截止 T-1 收盘的信息（_apply_feature_closure
  统一后移一日）；运行期自检 run_feature_window_check 检出任何使用当日
  及以后数据的特征（扰动重构 + 全时段统计量检测）。

样本切分合规（1.3 样本切平时序合规）：
  训练/验证严格按时间轴线性切分（_split_train_val：先过去后未来），
  关闭任何随机 Shuffle；训练窗口尾端 purge 5 天（≥ 标签持有期）；
  运行期自检 validate_time_split 检出打乱/重叠/交错（未来样本进入训练集），
  循环末对全部历史折叠执行 check_walk_forward_folds 检出窗口回退。

数据重叠泄漏隔离（1.4 Data Overlap Purging & Embargo）：
  特征含最多 60 天滚动窗口（ret_60d / price_pos_ma60），标签为未来 5 天
  收益 ⇒ 训练与验证间必须留 ≥ 60 天的时间真空隔离带（_split_train_val
  在训练窗口尾端 purge max(60, 5) = 60 天）；运行期自检
  validate_purge_embargo 检出特征/标签隔离带不足（时间跨度重叠泄露）。

预处理信息隔离（1.5 Preprocessing Isolation）：
  特征变换为逐日横截面排名（各日独立、与训练和未来无关）与逐行 NaN 掩码，
  不启用任何含时间维分布参数的预处理（无 imputation/标准化/winsorize/PCA）。
  任何新增分布参数预处理必须登记进 _PREPROCESS_PARAMS（拟合区间只能在
  训练集内）；运行期自检 check_train_features_invariant 通过「训练行不变性
  重构」检出全局/含测试期分布拟合。

动态重训时间戳锚定（1.6 Anchor-Time Verification）：
  锁死模型可见数据边界——任何一次重训的 fit 输入流（特征 X 与标签 Y 的
  样本行）最大时间戳必须 ≤ 重训锚点 T（check_train_cutoff 运行期自检）；
  重训当天信号废弃：T 日信号只可由训练时点 < T 的上一代模型产出，当天
  训练完成的模型最早自 T+1（下一交易窗口）起信号（重训日先旧模型、后重训，
  check_first_signal_after_train 周期末复核）。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr

from LogicAnalyzer.ml.anchor_integrity import (
    check_first_signal_after_train,
    check_train_cutoff,
)
from LogicAnalyzer.ml.feature_window import run_feature_window_check
from LogicAnalyzer.ml.label_integrity import LabelConvention, validate_labels
from LogicAnalyzer.ml.overlap_integrity import (
    check_walk_forward_bands,
    validate_purge_embargo,
)
from LogicAnalyzer.ml.preprocess_isolation import (
    PreprocessParam,
    check_param_fit_within_train,
    check_train_features_invariant,
)
from LogicAnalyzer.ml.split_integrity import (
    check_walk_forward_folds,
    validate_time_split,
)

_RETRAIN_FREQ = 60
_RETRAIN_EVERY = 20
_TRAIN_WINDOW = 180  # 最多用最近 180 天训练（含 60 天隔离带后仍有约 84 天训练）
_VAL_SPLIT = 0.8     # 训练窗口内 80% 训练，20% 验证（时序前 train 后 val）
_FEATURE_WINDOW_DAYS = 60  # 特征最大滚动窗口（ret_60d / price_pos_ma60 为 60 天）
_PURGE_DAYS = 5      # 标签侧隔离（Embargo）：train 末尾清洗天数 ≥ 标签持有期
_ISOLATION_DAYS = max(_FEATURE_WINDOW_DAYS, _PURGE_DAYS)  # 特征/标签隔离带取大者 = 60
_LABEL_HORIZON = 5   # 标签持有期（天）
_EARLY_STOP_ROUNDS = 15
_N_SHUFFLE = 40      # label shuffle 检验洗牌次数
_MIN_VAL_IC = 0.02   # 验证集 Rank IC 最低阈值，低于此值视为无信号
_ML_SIGNAL_ENABLED = True   # ML 信号主开关：True=启用 XGBoost 覆写（内置 label shuffle 检验，不达标自动回退）

# 1.5 预处理信息隔离：分布参数预处理登记表（当前管线为逐日横截面排名 +
# 逐行掩码，不启用任何含时间维分布参数的预处理）。若后续加入 imputation/
# 标准化/winsorize/PCA，必须在此登记且拟合区间只能落在当前训练集内。
_PREPROCESS_PARAMS: list[PreprocessParam] = []

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
        if col not in df.columns:
            continue
        ranks = df.groupby("trade_date")[col].rank(pct=True)
        df[col] = ranks.fillna(0.5).astype(np.float64)
    return df


def _compute_features_raw(df: pd.DataFrame) -> pd.DataFrame:
    """计算全部原始特征（T 行含当日行情，未做 T-1 窗口闭合与横截面标准化）。

    供 _compute_feature_matrix 与特征窗口自检（1.2）复用。
    """
    df = df.copy()
    has_atr = "ATR" in df.columns
    if has_atr:
        df["atr_log"] = np.log(df["ATR"].clip(lower=1e-8))
        df["atr_ratio"] = df["ATR"] / (df["close"] + 1e-8)

    for sym, grp in df.groupby("symbol"):
        idx = grp.index
        c = grp["close"]
        h = grp["high"]
        lo = grp["low"]
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
        hl = (h - lo) / (c + 1e-8)
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
        df.loc[idx, "close_position"] = (c - lo) / (h - lo + 1e-8)
        vwap = a / (v + 1e-8)
        df.loc[idx, "vwap_dev"] = c / vwap - 1
        df.loc[idx, "price_pos_ma20"] = c / c.rolling(20).mean() - 1
        df.loc[idx, "price_pos_ma60"] = c / c.rolling(60).mean() - 1
        df.loc[idx, "up_ratio_20d"] = (c > prev_close).rolling(20).mean()

        cum_up = (c > prev_close).astype(int)
        consec = cum_up.groupby((cum_up != cum_up.shift(1)).cumsum()).cumsum()
        df.loc[idx, "consec_up_days"] = consec

        # ── 目标变量（原始值，后续转截面排名） ──
        # 合规约定（尾盘下单）：信号当日生成、当日收盘价成交 → 标签起点
        # 必须是明日收盘价（C_{t+h} / C_{t+1} - 1），禁止以今日收盘价为起点。
        df.loc[idx, "fwd_5d_raw"] = c.shift(-_LABEL_HORIZON) / c.shift(-1) - 1

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

    return df


def _apply_feature_closure(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    """1.2 特征窗口闭合：特征在 T 行只使用截止 T-1 收盘的信息。

    合规约定（尾盘下单）：信号 T 日生成并以 T 日收盘价成交，特征窗口必须
    截止 T-1 收盘——T 行不得引用当日收盘/最高/最低/量/额（即未来函数）。
    实现：将各特征列按 symbol 整体后移一日（T 行取值 = 以 T-1 及以前数据
    计算的原始值）。目标变量（fwd_5d_raw）不参与闭合。
    """
    for col in feature_cols:
        if col in df.columns:
            df[col] = df.groupby("symbol")[col].shift(1)
    return df


def _compute_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """特征矩阵（1.2 特征未来函数阻断）：原始特征 + T-1 窗口闭合。"""
    return _apply_feature_closure(_compute_features_raw(df), _FEATURE_ALL)


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """完整特征工程：T-1 窗口闭合 + 标签/特征合规自检 + 横截面标准化。"""
    raw_in = df
    df = _compute_feature_matrix(df)

    # ── 标签合规自检（1.1 标签前瞻性清除） ──
    try:
        report = validate_labels(
            df, "fwd_5d_raw", LabelConvention.TAIL_CLOSE, _LABEL_HORIZON
        )
        if not report.passed:
            logger.warning(
                f"[标签合规] fwd_5d_raw 存在前瞻性泄露："
                f"违规 {report.n_violations}/{report.n_checked} 行"
            )
    except Exception as e:
        logger.warning(f"[标签合规] 自检执行失败: {e}")

    # ── 特征窗口自检（1.2 特征未来函数阻断） ──
    try:
        result = run_feature_window_check(_compute_feature_matrix, raw_in, _FEATURE_ALL)
        if not result["passed"]:
            failed = "；".join(r.check_name for r in result["reports"] if not r.passed)
            logger.warning(f"[特征窗口] 自检未通过: {failed}（特征使用了当日及以后数据）")
    except Exception as e:
        logger.warning(f"[特征窗口] 自检执行失败: {e}")

    # ── 预处理信息隔离自检（1.5 预处理信息隔离） ──
    try:
        iso_report = check_train_features_invariant(
            _compute_feature_matrix, raw_in, _FEATURE_ALL
        )
        if not iso_report.passed:
            logger.warning(
                f"[预处理隔离] 训练行特征随测试行漂移（分布参数含测试/全局数据）: "
                f"{'；'.join(iso_report.details[:3])}"
            )
    except Exception as e:
        logger.warning(f"[预处理隔离] 重构自检执行失败: {e}")

    # 横截面标准化所有特征
    df = _cross_sectional_rank(df, _FEATURE_ALL)

    # 目标变量：截面排名收益率（选股能力）
    df[_TARGET] = df.groupby("trade_date")["fwd_5d_raw"].rank(pct=True)
    df.drop(columns=["fwd_5d_raw"], inplace=True)
    return df


def _split_train_val(window_dates: list[str]) -> tuple[list[str], list[str]]:
    """1.3/1.4：窗口内时序线性切分（先过去后未来）+ 时间真空隔离带。

    - 1.3 样本切分合规：严格按时间轴顺序 80% 训练 / 20% 验证，不做任何
      随机 Shuffle。
    - 1.4 数据重叠泄漏隔离：训练窗口尾端 purge _ISOLATION_DAYS 天
      （= max(特征滚动窗口 60 天, 标签持有期 5 天)），在训练与验证之间留出
      至少 60 个交易日的时间真空隔离带，消除特征滚动窗口与训练标签价格
      区间进入验证集造成的重叠泄露。
    """
    split = int(len(window_dates) * _VAL_SPLIT)
    purge = min(_ISOLATION_DAYS, split)
    train_dates = window_dates[:split - purge]
    val_dates = window_dates[split:]
    return train_dates, val_dates


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
    folds: list[tuple[list[str], list[str]]] = []   # 1.3 多折 Walk-Forward 折叠（周期末自检）
    model_anchor_idx: int | None = None            # 1.6 当前生效模型的训练锚点索引
    anchor_audits: list[dict[str, Any]] = []       # 1.6 锚点审计：{anchor_idx, first_idx}

    def _emit_prediction(i_day: int) -> None:
        """用当前生效模型输出第 i_day 个交易日的信号并覆写当日进场评分。"""
        nonlocal total_predicted
        day_mask = df["trade_date"] == dates[i_day]
        day_X = df.loc[day_mask, _FEATURE_ALL].values.astype(np.float64)
        day_valid = np.all(np.isfinite(day_X), axis=1)
        if day_valid.sum() == 0:
            return
        day_idx = df.index[day_mask]
        pred_raw = model.predict(day_X)
        pred_s = pd.Series(pred_raw, index=day_idx)
        if pred_s.std() < 1e-6:
            return
        df.loc[day_idx, "进场评分"] = (pred_s.rank(pct=True) * 99 + 1).values
        total_predicted += day_valid.sum()
        if (
            model_anchor_idx is not None
            and anchor_audits
            and anchor_audits[-1]["first_idx"] is None
            and anchor_audits[-1]["anchor_idx"] == model_anchor_idx
        ):
            anchor_audits[-1]["first_idx"] = i_day

    for i, date in enumerate(dates):
        if i < _RETRAIN_FREQ:
            continue

        should_retrain = (
            last_train_date_idx is None
            or i - last_train_date_idx >= _RETRAIN_EVERY
        )

        if should_retrain:
            # 1.6 重训当天信号废弃：T 日信号只可由训练时点 < T 的旧模型产出，
            # 先用旧模型完成 T 日信号，再重训；新模型最早自 T+1 交易日起生效。
            if model_valid and last_train_date_idx is not None:
                _emit_prediction(i)

            cut_idx = max(0, i - _TRAIN_WINDOW)
            window_dates = dates[cut_idx:i]
            train_dates, val_dates = _split_train_val(window_dates)

            # 1.6 动态训练集截止线自检：fit 输入流最大时间戳必须 ≤ 锚点 T
            try:
                cutoff_report = check_train_cutoff(train_dates, date)
                if not cutoff_report.passed:
                    logger.warning(
                        f"[锚定检查] 训练集含 > 锚点样本: "
                        f"{'；'.join(cutoff_report.details[:3])}"
                    )
            except Exception as e:
                logger.warning(f"[锚定检查] 截止线自检执行失败: {e}")

            # 1.3 样本切分自检（单折）：检出打乱/重叠/未来样本进入训练集/purge 不足
            if train_dates and val_dates:
                try:
                    split_report = validate_time_split(
                        train_dates, val_dates,
                        horizon=_LABEL_HORIZON, purge_days=_PURGE_DAYS,
                    )
                    if not split_report.passed:
                        logger.warning(
                            f"[切分合规] 重训折叠不合规: "
                            f"{'；'.join(split_report.details[:3])}"
                        )
                except Exception as e:
                    logger.warning(f"[切分合规] 单折自检执行失败: {e}")

                # 1.4 数据重叠泄漏隔离自检（单折）：特征/标签隔离带不足即时间跨度重叠泄露
                try:
                    band_report = validate_purge_embargo(
                        train_dates, val_dates,
                        feature_window_days=_FEATURE_WINDOW_DAYS,
                        horizon=_LABEL_HORIZON,
                    )
                    if not band_report.passed:
                        logger.warning(
                            f"[重叠检查] 隔离带不足: {band_report.details[0]}"
                            f"；{band_report.details[1] if len(band_report.details) > 1 else ''}"
                        )
                except Exception as e:
                    logger.warning(f"[重叠检查] 单折自检执行失败: {e}")

                # 1.5 预处理信息隔离自检（单折）：分布参数拟合基准只能来自当前训练集
                try:
                    param_report = check_param_fit_within_train(
                        _PREPROCESS_PARAMS, train_dates, test_dates=val_dates
                    )
                    if not param_report.passed:
                        logger.warning(
                            f"[预处理隔离] 参数来源违规: "
                            f"{'；'.join(param_report.details[:3])}"
                        )
                except Exception as e:
                    logger.warning(f"[预处理隔离] 参数来源自检执行失败: {e}")

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
            folds.append((train_dates, val_dates))
            last_train_date_idx = i
            model_anchor_idx = i            # 1.6 新模型锚点 = 重训日 T
            anchor_audits.append({"anchor_idx": i, "first_idx": None})
            # Label shuffle 检验：不显著（p>=0.05 或 IC 过低）自动回退原生评分
            if X_val is not None and len(y_val) >= 50:
                ic_val, p_val = _label_shuffle_test(model, X_val, y_val)
                model_valid = (p_val < 0.05) and (ic_val >= _MIN_VAL_IC)
                logger.info(f"[ML] 验证集 IC={ic_val:.4f}  label shuffle p={p_val:.4f}" +
                            (" ✓显著" if model_valid else " ✗不显著，回退原生评分"))
            else:
                model_valid = False
                logger.info("[ML] 验证集样本不足，回退原生评分")

            # 1.6 重训当天信号废弃：新模型从下一交易窗口（T+1）起生效，T 日不产信号
            continue

        elif last_train_date_idx is None:
            continue

        if not model_valid:
            continue

        _emit_prediction(i)

        if i % max(1, len(dates) // 10) == 0:
            logger.info(f"[ML] 进度 {date} — 已预测 {total_predicted} 行")

    if last_train_date_idx is None:
        logger.warning("[ML] 模型训练失败，使用原始 MACD 评分")
        return merged

    logger.info(f"[ML] Walk-forward 完成，共预测 {total_predicted} 行")

    # 1.3 样本切分自检（多折）：检出窗口回退/随机折叠/打乱残留
    if folds:
        try:
            wf_report = check_walk_forward_folds(
                folds, horizon=_LABEL_HORIZON, purge_days=_PURGE_DAYS
            )
            if not wf_report.passed:
                logger.warning(
                    f"[切分合规] Walk-Forward 折叠不合规（{wf_report.n_violations} 项）: "
                    f"{'；'.join(wf_report.details[:3])}"
                )
        except Exception as e:
            logger.warning(f"[切分合规] 多折自检执行失败: {e}")

    # 1.4 数据重叠泄漏隔离自检（多折）：全部历史折叠的隔离带合规
    if folds:
        try:
            band_report = check_walk_forward_bands(
                folds, feature_window_days=_FEATURE_WINDOW_DAYS, horizon=_LABEL_HORIZON
            )
            if not band_report.passed:
                logger.warning(
                    f"[重叠检查] Walk-Forward 隔离带不合规（{band_report.n_violations} 项）: "
                    f"{'；'.join(band_report.details[:3])}"
                )
        except Exception as e:
            logger.warning(f"[重叠检查] 多折自检执行失败: {e}")

    # 1.6 动态重训时间戳锚定自检（多轮）：每个模型的首个信号必须晚于其训练锚点
    for rec in anchor_audits:
        if rec["first_idx"] is None:
            continue
        try:
            sig_report = check_first_signal_after_train(
                dates[rec["anchor_idx"]], dates[rec["first_idx"]]
            )
            if not sig_report.passed:
                logger.warning(
                    f"[锚定检查] 重训当天信号违规: "
                    f"{'；'.join(sig_report.details[:3])}"
                )
        except Exception as e:
            logger.warning(f"[锚定检查] 信号时序自检执行失败: {e}")

    if isinstance(model, XGBSignalModel):
        fi = model.feature_importances()
        fi_str = " | ".join(f"{n}: {v:.1%}" for n, v in fi[:5])
        logger.info(f"[ML] 特征重要性 Top5: {fi_str}")
        if model._best_iteration:
            logger.info(f"[ML] XGBoost 早停: best_iteration={model._best_iteration}")
    return df
