"""
参数灵敏度分析

对目标参数做网格扫描，批量跑全量分析，输出每个参数组合下的：
  - A/B/C/D 分布
  - 7 维评分空值率
  - 综合得分均值/中位数/标准差

用于排除"怎么调都没区别"的冗余参数，缩小待优化参数集。

用法：
    python sensitivity_runner.py --scan first_pass           # 首次扫描（阈值+衰减+规则）
    python sensitivity_runner.py --scan weights              # 7 维权重扫描
    python sensitivity_runner.py --scan all                  # 全量扫描（组合较多，建议 overnight）
    python sensitivity_runner.py --dry-run --scan first_pass # 只打印参数组合
    python sensitivity_runner.py --resume --scan weights     # 断点续跑
"""

from __future__ import annotations

import configparser
import glob
import json
import os
import shutil
import sys
import tempfile
import time
from itertools import product

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from LogicAnalyzer.StockAnalysisCoordinator import StockAnalysisCoordinatorFactory

# ── 工具函数 ────────────────────────────────────────────────────────


def _make_temp_config(base_path: str, overrides: dict, label: str) -> str:
    """复制 base config.ini，应用 overrides，写入临时文件，返回路径。"""
    cp = configparser.ConfigParser(inline_comment_prefixes=("#", ";"), interpolation=None)
    cp.read(base_path, encoding="utf-8")

    for section, params in overrides.items():
        if section not in cp:
            cp[section] = {}
        for key, value in params.items():
            cp[section][key] = str(value)

    tmp_dir = tempfile.mkdtemp(prefix="sens_")
    tmp_path = os.path.join(tmp_dir, f"config_{label}.ini")
    with open(tmp_path, "w", encoding="utf-8") as f:
        cp.write(f)
    return tmp_path, tmp_dir


def _extract_stats(report_dir: str, label: str) -> dict:
    """从最新 Excel 提取统计指标。"""
    pattern = os.path.join(report_dir, "审计报告_*.xlsx")
    files = sorted(glob.glob(pattern), key=os.path.getctime, reverse=True)
    if not files:
        return {"label": label, "error": "no report found"}

    df = pd.read_excel(files[0], sheet_name="数据汇总")

    level_col = "综合级别"
    if level_col not in df.columns:
        return {"label": label, "error": f"column '{level_col}' not found"}

    total = len(df)
    level_dist = df[level_col].value_counts().to_dict()

    score_col = "综合分析评分"
    score_series = pd.to_numeric(df[score_col], errors="coerce")
    score_stats = {
        "mean": round(float(score_series.mean()), 2),
        "median": round(float(score_series.median()), 2),
        "std": round(float(score_series.std()), 2),
        "min": round(float(score_series.min()), 2),
        "max": round(float(score_series.max()), 2),
        "null_count": int(score_series.isna().sum()),
    }

    dim_cols = ["MACD趋势", "金叉信号", "柱状动能", "DIF斜率", "背离信号", "量价配合", "K线形态"]
    null_rates = {}
    for col in dim_cols:
        if col in df.columns:
            empty = int(df[col].astype(str).str.strip().eq("").sum())
            null_rates[col] = round(empty / total * 100, 1) if total > 0 else 0
        else:
            null_rates[col] = None

    return {
        "label": label,
        "total_stocks": total,
        "level_distribution": {k: {"count": v, "pct": round(v / total * 100, 1)} for k, v in level_dist.items()},
        "score_stats": score_stats,
        "dim_null_rates": null_rates,
    }


def _cleanup_temp(tmp_dir: str) -> None:
    """清理临时目录。"""
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass


# ── 网格定义 ────────────────────────────────────────────────────────

SCANS = {}

# 首次扫描：阈值 + 衰减 + 规则参数
SCANS["first_pass"] = {
    "SCORING_PARAMS": {
        "cross_decay_days": [20, 30, 60],
        "cross_decay_min": [0.2, 0.3, 0.5],
        "kline_decay_days": [5, 10, 20],
        "kline_decay_min": [0.1, 0.2, 0.3],
    },
    "RULE_THRESHOLDS": {
        "divergence": [0.2, 0.3, 0.4],
        "winner_rate_high": [70, 80, 90],
        "liq_veto_ratio": [0.03, 0.05, 0.08],
    },
    "TECHNICAL_CONSTANTS": {
        "atr_length": [10, 14, 20],
        "rsi_length": [10, 14, 20],
    },
}

# 7 维权重扫描
SCANS["weights"] = {
    "FULL_BULL_WEIGHTS": [
        {"MACD趋势": 25, "金叉信号": 10, "柱状动能": 15, "DIF斜率": 10, "背离信号": 10, "量价配合": 10, "K线形态": 10},
        {"MACD趋势": 20, "金叉信号": 20, "柱状动能": 15, "DIF斜率": 10, "背离信号": 10, "量价配合": 10, "K线形态": 5},
        {"MACD趋势": 15, "金叉信号": 15, "柱状动能": 20, "DIF斜率": 15, "背离信号": 10, "量价配合": 10, "K线形态": 5},
        {"MACD趋势": 20, "金叉信号": 15, "柱状动能": 15, "DIF斜率": 10, "背离信号": 15, "量价配合": 10, "K线形态": 5},
        {"MACD趋势": 20, "金叉信号": 15, "柱状动能": 15, "DIF斜率": 10, "背离信号": 10, "量价配合": 15, "K线形态": 5},
        {"MACD趋势": 30, "金叉信号": 15, "柱状动能": 15, "DIF斜率": 5, "背离信号": 5, "量价配合": 5, "K线形态": 5},
    ],
}

# 全量扫描（组合较多）
SCANS["all"] = {}
SCANS["all"].update(SCANS["first_pass"])
for section, params_list in SCANS["weights"].items():
    SCANS["all"][section] = params_list
# 只保留 first_pass 中的 FULL_BULL_WEIGHTS 条目，用 weights 的替代
# 手动合并
full_bull_weight_list = SCANS["weights"]["FULL_BULL_WEIGHTS"]
scoring_params = SCANS["first_pass"]["SCORING_PARAMS"]
rule_thresholds = SCANS["first_pass"]["RULE_THRESHOLDS"]
tech_constants = SCANS["first_pass"]["TECHNICAL_CONSTANTS"]
SCANS["all"] = {
    "FULL_BULL_WEIGHTS": full_bull_weight_list,
    "SCORING_PARAMS": scoring_params,
    "RULE_THRESHOLDS": rule_thresholds,
    "TECHNICAL_CONSTANTS": tech_constants,
}

# ── Config overrides 映射 ───────────────────────────────────────────


def _build_overrides(combo: dict) -> dict:
    """将 combo dict 转为 config.ini section/key/value 格式。

    combo 键名使用语义名（如 divergence, cross_decay_days），
    方法内部映射到 config.ini 实际的节/键名。

    映射表：
      SCORING_PARAMS.{key}      → [SCORING_PARAMS].{UPPER_KEY}
      RULE_THRESHOLDS.{key}     → [FULL_BULL_SCORING].{UPPER_MAPPED_KEY}
      TECHNICAL_CONSTANTS.{key} → [TECHNICAL_CONSTANTS].{UPPER_KEY}
      FULL_BULL_WEIGHTS.{cn}    → [FULL_BULL_SCORING].{WEIGHT_*}
    """
    # SCORING_PARAMS 键名映射（网格键 → ini键名，ConfigParser 有大小写混用，保持原样）
    SCORING_MAP = {
        "cross_decay_days": "cross_decay_days",
        "cross_decay_min": "cross_decay_min",
        "kline_decay_days": "kline_decay_days",
        "kline_decay_min": "kline_decay_min",
    }
    # RULE_THRESHOLDS → ini 映射（注意来源不同，target 格式为 (section, key)）
    RULE_MAP = {
        "divergence": ("FULL_BULL_SCORING", "RULE_DIVERGENCE_THRESHOLD"),
        "winner_rate_high": ("FULL_BULL_SCORING", "RULE_WINNER_RATE_HIGH"),
        "winner_rate_low": ("FULL_BULL_SCORING", "RULE_WINNER_RATE_LOW"),
        "liq_veto_ratio": ("BACKTEST_CALIBRATED", "liq_veto_ratio"),
    }
    # TECHNICAL_CONSTANTS 键名映射
    TECH_MAP = {
        "atr_length": "ATR_LENGTH",
        "rsi_length": "RSI_LENGTH",
        "adx_length": "ADX_LENGTH",
        "boll_length": "BOLL_LENGTH",
        "boll_std": "BOLL_STD",
        "stoch_k": "STOCH_K",
        "stoch_d": "STOCH_D",
    }
    # 权重键名映射（网格中文key → ini大写键名）
    WEIGHT_MAP = {
        "MACD趋势": "WEIGHT_ZERO_AXIS",
        "金叉信号": "WEIGHT_STRATEGY_GOLDEN",
        "柱状动能": "WEIGHT_MOMENTUM",
        "DIF斜率": "WEIGHT_DIF_SLOPE",
        "背离信号": "WEIGHT_DIVERGENCE",
        "量价配合": "WEIGHT_VOLUME_PRICE",
        "K线形态": "WEIGHT_KLINE_PATTERN",
    }

    overrides = {}
    for section, params in combo.items():
        if section == "FULL_BULL_WEIGHTS":
            ini_section = "FULL_BULL_SCORING"
            overrides[ini_section] = {}
            for cn_key, ini_key in WEIGHT_MAP.items():
                if cn_key in params:
                    overrides[ini_section][ini_key] = str(params[cn_key])
        elif section == "SCORING_PARAMS":
            overrides[section] = {}
            for k, v in params.items():
                ini_key = SCORING_MAP.get(k, k.upper())
                overrides[section][ini_key] = str(v)
        elif section == "RULE_THRESHOLDS":
            for k, v in params.items():
                target = RULE_MAP.get(k)
                if target is None:
                    continue
                ini_section, ini_key = target
                if ini_section not in overrides:
                    overrides[ini_section] = {}
                overrides[ini_section][ini_key] = str(v)
        elif section == "TECHNICAL_CONSTANTS":
            overrides[section] = {}
            for k, v in params.items():
                ini_key = TECH_MAP.get(k, k.upper())
                overrides[section][ini_key] = str(v)
        else:
            overrides[section] = {k: str(v) for k, v in params.items()}
    return overrides


# ── 参数组合展开器 ───────────────────────────────────────────────────


def _flatten_grid(grid: dict) -> list[dict]:
    """展平网格为参数组合列表。

    grid 支持两种格式：
      - {section: {key: [values...]}}  → 笛卡尔积
      - {section: [dict, dict, ...]}    → 列举
    """
    combos = []
    for section, params in grid.items():
        if isinstance(params, list):
            # 列举式（如权重组合）
            for item in params:
                combos.append({section: item})
        elif isinstance(params, dict):
            keys = list(params.keys())
            values = list(params.values())
            for combo in product(*values):
                combos.append({section: dict(zip(keys, combo))})
    return combos


# ── 组合标签 ────────────────────────────────────────────────────────


def _make_label(combo: dict) -> str:
    parts = []
    for section, params in combo.items():
        if isinstance(params, dict):
            for k, v in params.items():
                parts.append(f"{section}.{k}={v}")
        else:
            parts.append(f"{params}")
    return "_".join(parts).replace(" ", "")


# ── 主扫描函数 ───────────────────────────────────────────────────────


def run_scan(grid: dict, base_config_path: str = "config.ini",
             dry_run: bool = False, resume: bool = False) -> pd.DataFrame | None:
    """执行网格扫描。"""
    param_combos = _flatten_grid(grid)
    print(f"\n参数组合数: {len(param_combos)}")
    if dry_run:
        for i, combo in enumerate(param_combos):
            print(f"  [{i+1}] {_make_label(combo)}")
        return

    results_dir = os.path.join(os.path.dirname(os.path.abspath(base_config_path)),
                               "sensitivity_results")
    os.makedirs(results_dir, exist_ok=True)
    results_log = os.path.join(results_dir, "results.jsonl")

    # 断点续跑：读取已完成标签
    completed_labels = set()
    if resume and os.path.exists(results_log):
        with open(results_log, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    completed_labels.add(entry.get("label", ""))
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"断点续跑: 跳过 {len(completed_labels)} 个已完成组合")

    # 基线：不修改任何参数
    print(f"\n[0/{len(param_combos)}] BASELINE (默认参数)")
    try:
        coordinator = StockAnalysisCoordinatorFactory.create(config_file=base_config_path)
        coordinator.run()
        baseline_stats = _extract_stats(coordinator.config.TEMP_DATA_DIRECTORY, "BASELINE")
        baseline_stats["elapsed_seconds"] = 0
        baseline_stats["combo"] = {}
        with open(results_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(baseline_stats, ensure_ascii=False) + "\n")
        print(f"  A/B/C/D: {baseline_stats.get('level_distribution', {})}")
    except Exception as e:
        print(f"  [BASELINE ERROR] {e}")

    for i, combo in enumerate(param_combos):
        label = _make_label(combo)

        if label in completed_labels:
            print(f"[{i+1}/{len(param_combos)}] SKIP {label} (已完成)")
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(param_combos)}] {label}")

        overrides = _build_overrides(combo)
        tmp_config_path, tmp_dir = _make_temp_config(base_config_path, overrides, label)

        t0 = time.time()
        try:
            coordinator = StockAnalysisCoordinatorFactory.create(config_file=tmp_config_path)
            coordinator.run()
            elapsed = time.time() - t0

            # 临时 config 使用原始 TEMP_DATA_DIRECTORY（不修改路径）
            stats = _extract_stats(coordinator.config.TEMP_DATA_DIRECTORY, label)
            stats["elapsed_seconds"] = round(elapsed, 1)
            stats["combo"] = combo

            ld = stats.get("level_distribution", {})
            ss = stats.get("score_stats", {})
            print(f"  A/B/C/D: {ld}")
            print(f"  得分: mean={ss.get('mean','?')} median={ss.get('median','?')} std={ss.get('std','?')}")
            print(f"  耗时: {elapsed:.0f}s")

            with open(results_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(stats, ensure_ascii=False) + "\n")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [ERROR] {e}")
            with open(results_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "label": label, "error": str(e), "combo": combo,
                    "elapsed_seconds": round(elapsed, 1),
                }, ensure_ascii=False) + "\n")
        finally:
            _cleanup_temp(tmp_dir)

    # ── 总结 ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"扫描完成。结果保存在: {results_log}")

    if not os.path.exists(results_log):
        return

    results = []
    with open(results_log, "r", encoding="utf-8") as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    rows = []
    for r in results:
        if "error" in r:
            continue
        row = {"label": r["label"]}
        ld = r.get("level_distribution", {})
        for level in ["A", "B", "C", "D"]:
            row[f"{level}_pct"] = ld.get(level, {}).get("pct", 0)
        ss = r.get("score_stats", {})
        row["score_mean"] = ss.get("mean")
        row["score_median"] = ss.get("median")
        row["score_std"] = ss.get("std")
        row["score_null"] = ss.get("null_count", 0)
        for dim, rate in r.get("dim_null_rates", {}).items():
            row[f"null_{dim}"] = rate
        row["elapsed_s"] = r.get("elapsed_seconds", 0)
        rows.append(row)

    summary = pd.DataFrame(rows)
    csv_path = os.path.join(results_dir, "summary.csv")
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"总结表: {csv_path}")

    if len(summary) > 1:
        print("\n--- 灵敏度初步诊断 ---")
        metrics = ["score_mean", "score_median", "score_std", "A_pct", "B_pct", "C_pct", "D_pct"]
        for col in metrics:
            if col not in summary.columns:
                continue
            vals = summary[col].dropna()
            if len(vals) <= 1:
                continue
            lo, hi = vals.min(), vals.max()
            spread = hi - lo
            mean_val = vals.mean()
            cv = spread / mean_val if abs(mean_val) > 1e-6 else 0
            marker = " [低灵敏度]" if cv < 0.05 and spread < 5 else ""
            print(f"  {col}: [{lo:.1f}, {hi:.1f}] 极差={spread:.1f} CV={cv:.2f}{marker}")
        print("(低灵敏度 = CV<0.05 且极差<5，参数对这个指标几乎没有影响)")

    return summary


# ── CLI ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="参数灵敏度分析")
    parser.add_argument("--scan", default="first_pass",
                        choices=list(SCANS.keys()) + ["list"],
                        help="扫描方案（list 查看所有方案）")
    parser.add_argument("--config", default="config.ini", help="基础配置文件")
    parser.add_argument("--dry-run", action="store_true", help="只打印组合，不执行")
    parser.add_argument("--resume", action="store_true", help="断点续跑")
    args = parser.parse_args()

    if args.scan == "list":
        print("可用扫描方案:")
        for name in SCANS:
            grid = SCANS[name]
            count = len(_flatten_grid(grid))
            print(f"  {name}: {count} 个参数组合")
        sys.exit(0)

    grid = SCANS.get(args.scan)
    if not grid:
        print(f"未知扫描方案: {args.scan}")
        sys.exit(1)

    run_scan(grid, base_config_path=args.config,
             dry_run=args.dry_run, resume=args.resume)
