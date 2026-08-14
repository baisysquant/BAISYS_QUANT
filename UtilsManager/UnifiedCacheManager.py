"""
统一缓存管理器

特性：
- 统一的缓存目录
- 自动缓存验证
- 灵活的失效策略
- 缓存监控和统计
- CSV 文件缓存 API（load_cache / save_cache / cache_exists / get_cache_path）
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger


class CacheStrategy:
    """缓存策略枚举"""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    MANUAL = "manual"
    NEVER = "never"
    CUSTOM = "custom"


class CacheConfig:
    """缓存配置类"""

    def __init__(
        self,
        strategy: str = CacheStrategy.DAILY,
        ttl_seconds: int | None = None,
        max_size_mb: float = 100.0,
        compress: bool = False,
        validate_on_load: bool = True,
    ) -> None:
        self.strategy = strategy
        self.ttl_seconds = ttl_seconds
        self.max_size_mb = max_size_mb
        self.compress = compress
        self.validate_on_load = validate_on_load


class CacheEntry:
    """缓存条目元数据"""

    def __init__(self, key: str, created_at: float, size_bytes: int, metadata: dict | None = None) -> None:
        self.key = key
        self.created_at = created_at
        self.size_bytes = size_bytes
        self.metadata = metadata or {}

    def is_expired(self, strategy: CacheConfig) -> bool:
        if strategy.strategy == CacheStrategy.NEVER:
            return False
        if strategy.strategy == CacheStrategy.MANUAL:
            return False

        now = time.time()
        age_seconds = now - self.created_at

        if strategy.strategy == CacheStrategy.CUSTOM:
            if strategy.ttl_seconds is None:
                raise ValueError("CUSTOM策略必须指定ttl_seconds")
            return age_seconds > strategy.ttl_seconds

        if strategy.strategy == CacheStrategy.DAILY:
            created_date = datetime.fromtimestamp(self.created_at).date()
            today = datetime.now().date()
            return created_date != today

        elif strategy.strategy == CacheStrategy.WEEKLY:
            created_date = datetime.fromtimestamp(self.created_at).date()
            today = datetime.now().date()
            created_week = created_date.isocalendar()[1]
            today_week = today.isocalendar()[1]
            return created_week != today_week

        elif strategy.strategy == CacheStrategy.MONTHLY:
            created_date = datetime.fromtimestamp(self.created_at).date()
            today = datetime.now().date()
            return created_date.year != today.year or created_date.month != today.month

        return False


class UnifiedCacheManager:
    """
    统一缓存管理器

    特性：
    - 统一的缓存目录和命名规范
    - 多种失效策略
    - 自动缓存验证
    - 缓存统计和监控
    - 兼容 CSV 文件缓存（load_cache/save_cache/cache_exists/get_cache_path）
    """

    def __init__(
        self,
        cache_dir: str,
        default_strategy: str = CacheStrategy.DAILY,
        auto_cleanup: bool = True,
        today_str: str | None = None,
    ) -> None:
        """
        Args:
            cache_dir: 缓存根目录
            default_strategy: 默认缓存策略
            auto_cleanup: 是否自动清理过期缓存
            today_str: 业务日期字符串（YYYYMMDD 或 YYYY-MM-DD），
                       用于 CSV 文件缓存命名的日期后缀。
                       不传时自动取当前日期。
        """
        self.cache_dir = Path(cache_dir).expanduser()
        self.default_strategy = default_strategy
        self.auto_cleanup = auto_cleanup
        self._today_str = today_str

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.stats = {"hits": 0, "misses": 0, "writes": 0, "cleanups": 0}

        if auto_cleanup:
            self.cleanup_expired()

    # ── CSV 文件缓存方法 ──────────────────────────────────────────────────

    def _resolve_today(self) -> str:
        """获取用于文件命名的日期字符串（纯数字 YYYYMMDD，交易日优先）"""
        if self._today_str:
            return self._today_str.replace("-", "")
        try:
            from DataCollection.CalendarManager import TradingCalendarAnalyzer
            return TradingCalendarAnalyzer().get_last_trading_day().replace("-", "")
        except Exception:
            return datetime.now().strftime("%Y%m%d")

    def _compat_file_path(self, base_name: str, cleaned: bool = False, suffix: str = ".txt") -> str:
        """
        生成缓存文件路径。
        格式: {base_name}{_经清洗}_{todayYYYYMMDD}{suffix}
        """
        clean_suffix = "_经清洗" if cleaned else ""
        file_name = f"{base_name}{clean_suffix}_{self._resolve_today()}{suffix}"
        return os.path.join(str(self.cache_dir), file_name)

    def load_cache(
        self,
        base_name: str,
        cleaned: bool = True,
        sep: str = "|",
        encoding: str = "utf-8",
        dtype_mapping: dict | None = None,
    ) -> pd.DataFrame:
        """
        从 CSV 缓存文件加载数据

        Args:
            base_name: 文件基础名称
            cleaned: 是否加载清洗后的缓存
            sep: CSV分隔符
            encoding: 文件编码
            dtype_mapping: 列数据类型映射

        Returns:
            pd.DataFrame: 加载的数据，不存在或失败返回空 DataFrame
        """
        file_path = self._compat_file_path(base_name, cleaned=cleaned)

        if not os.path.exists(file_path):
            return pd.DataFrame()

        try:
            if dtype_mapping is None:
                dtype_mapping = {"股票代码": str, "symbol": str}

            df = pd.read_csv(file_path, sep=sep, encoding=encoding, dtype=dtype_mapping)

            if "symbol" in df.columns and "股票代码" not in df.columns:
                df.rename(columns={"symbol": "股票代码"}, inplace=True)

            logger.info(f"  - 发现缓存，加载: {os.path.basename(file_path)}")
            return df

        except Exception as e:
            logger.warning(f"[WARN] 加载缓存 {os.path.basename(file_path)} 失败: {e}，将重新获取。")
            return pd.DataFrame()

    def save_cache(
        self, df: pd.DataFrame, base_name: str, cleaned: bool = True, sep: str = "|", encoding: str = "utf-8"
    ) -> bool:
        """
        保存数据到 CSV 缓存文件

        Args:
            df: 要保存的 DataFrame
            base_name: 文件基础名称
            cleaned: 是否保存为清洗后的数据
            sep: CSV分隔符，默认 '|'
            encoding: 文件编码，默认 'utf-8'

        Returns:
            bool: 保存是否成功
        """
        if df is None or df.empty:
            return False

        file_path = self._compat_file_path(base_name, cleaned=cleaned)

        try:
            df.to_csv(file_path, sep=sep, index=False, encoding=encoding)
            logger.info(f"  - 保存数据至缓存: {os.path.basename(file_path)}")
            return True

        except Exception as e:
            logger.error(f"[ERROR] 保存数据到缓存 {os.path.basename(file_path)} 失败: {e}")
            return False


    def _generate_cache_key(self, name: str, params: dict = None) -> str:
        """
        生成标准化的缓存键名

        Args:
            name: 缓存名称
            params: 可选的参数（用于区分不同配置的缓存）

        Returns:
            标准化的缓存文件名
        """
        if params:
            # 将参数字典转换为排序后的JSON字符串，然后哈希
            params_str = json.dumps(params, sort_keys=True)
            params_hash = hashlib.sha256(params_str.encode()).hexdigest()[:8]
            return f"{name}_{params_hash}"
        return name

    def _get_cache_path(self, key: str, extension: str = ".csv") -> Path:
        """获取缓存文件的完整路径"""
        filename = f"{key}{extension}"
        return self.cache_dir / filename

    def _get_metadata_path(self, key: str) -> Path:
        """获取元数据文件路径"""
        return self.cache_dir / f"{key}.meta.json"

    def _save_metadata(self, key: str, entry: CacheEntry) -> None:
        """保存缓存元数据"""
        meta_path = self._get_metadata_path(key)
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "key": entry.key,
                        "created_at": entry.created_at,
                        "size_bytes": entry.size_bytes,
                        "metadata": entry.metadata,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as e:
            logger.warning(f"保存缓存元数据失败: {e}")

    def _load_metadata(self, key: str) -> CacheEntry | None:
        """加载缓存元数据"""
        meta_path = self._get_metadata_path(key)
        if not meta_path.exists():
            return None

        try:
            with open(meta_path, encoding="utf-8") as f:
                data = json.load(f)
                return CacheEntry(
                    key=data["key"],
                    created_at=data["created_at"],
                    size_bytes=data["size_bytes"],
                    metadata=data.get("metadata", {}),
                )
        except Exception as e:
            logger.warning(f"加载缓存元数据失败: {e}")
            return None


    def cleanup_expired(self) -> None:
        """清理所有过期的缓存"""
        config = CacheConfig(strategy=self.default_strategy)
        cleaned_count = 0

        for meta_file in self.cache_dir.glob("*.meta.json"):
            try:
                key = meta_file.stem
                entry = self._load_metadata(key)

                if entry and entry.is_expired(config):
                    self._remove_cache(key)
                    cleaned_count += 1

            except Exception as e:
                logger.warning(f"清理缓存失败 [{meta_file.name}]: {e}")

        if cleaned_count > 0:
            self.stats["cleanups"] += 1
            logger.info(f"已清理 {cleaned_count} 个过期缓存")

    def _remove_cache(self, key: str) -> None:
        """删除缓存文件及其元数据"""
        cache_path = self._get_cache_path(key)
        meta_path = self._get_metadata_path(key)

        for path in [cache_path, meta_path]:
            if path.exists():
                try:
                    path.unlink()
                except Exception as e:
                    logger.warning(f"删除缓存文件失败 [{path.name}]: {e}")
