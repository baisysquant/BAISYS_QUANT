"""
统一缓存管理器

提供一致的缓存读写接口，支持：
- 统一的缓存命名规范
- 自动缓存验证
- 灵活的失效策略
- 缓存监控和统计
"""

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger


class CacheStrategy:
    """缓存策略枚举"""

    # 按时间失效
    DAILY = "daily"  # 每日失效（交易日）
    WEEKLY = "weekly"  # 每周失效
    MONTHLY = "monthly"  # 每月失效

    # 按事件失效
    MANUAL = "manual"  # 手动清除
    NEVER = "never"  # 永不过期（谨慎使用）

    # 自定义
    CUSTOM = "custom"  # 自定义失效逻辑


class CacheConfig:
    """缓存配置类"""

    def __init__(
        self,
        strategy: str = CacheStrategy.DAILY,
        ttl_seconds: int | None = None,
        max_size_mb: float = 100.0,
        compress: bool = False,
        validate_on_load: bool = True,
    ):
        """
        Args:
            strategy: 缓存失效策略
            ttl_seconds: 自定义TTL（秒），仅在strategy=CUSTOM时有效
            max_size_mb: 单个缓存文件最大大小（MB）
            compress: 是否压缩存储
            validate_on_load: 加载时是否验证数据完整性
        """
        self.strategy = strategy
        self.ttl_seconds = ttl_seconds
        self.max_size_mb = max_size_mb
        self.compress = compress
        self.validate_on_load = validate_on_load


class CacheEntry:
    """缓存条目元数据"""

    def __init__(self, key: str, created_at: float, size_bytes: int, metadata: dict = None):
        self.key = key
        self.created_at = created_at
        self.size_bytes = size_bytes
        self.metadata = metadata or {}

    def is_expired(self, strategy: CacheConfig) -> bool:
        """检查缓存是否过期"""
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

        # 按时间策略
        if strategy.strategy == CacheStrategy.DAILY:
            # 检查是否是同一天（考虑交易日）
            created_date = datetime.fromtimestamp(self.created_at).date()
            today = datetime.now().date()
            return created_date != today

        elif strategy.strategy == CacheStrategy.WEEKLY:
            # 检查是否是同一周
            created_date = datetime.fromtimestamp(self.created_at).date()
            today = datetime.now().date()
            created_week = created_date.isocalendar()[1]
            today_week = today.isocalendar()[1]
            return created_week != today_week

        elif strategy.strategy == CacheStrategy.MONTHLY:
            # 检查是否是同一月
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
    """

    def __init__(self, cache_dir: str, default_strategy: str = CacheStrategy.DAILY, auto_cleanup: bool = True):
        """
        Args:
            cache_dir: 缓存根目录
            default_strategy: 默认缓存策略
            auto_cleanup: 是否自动清理过期缓存
        """
        self.cache_dir = Path(cache_dir).expanduser()
        self.default_strategy = default_strategy
        self.auto_cleanup = auto_cleanup

        # 确保缓存目录存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 缓存统计
        self.stats = {"hits": 0, "misses": 0, "writes": 0, "cleanups": 0}

        # 启动时清理过期缓存
        if auto_cleanup:
            self.cleanup_expired()

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
            params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
            return f"{name}_{params_hash}"
        return name

    def _get_cache_path(self, key: str, extension: str = ".csv") -> Path:
        """获取缓存文件的完整路径"""
        filename = f"{key}{extension}"
        return self.cache_dir / filename

    def _get_metadata_path(self, key: str) -> Path:
        """获取元数据文件路径"""
        return self.cache_dir / f"{key}.meta.json"

    def _save_metadata(self, key: str, entry: CacheEntry):
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

    def save_dataframe(
        self, df: pd.DataFrame, name: str, config: CacheConfig = None, params: dict = None, metadata: dict = None
    ) -> bool:
        """
        保存DataFrame到缓存

        Args:
            df: 要缓存的DataFrame
            name: 缓存名称
            config: 缓存配置（可选，使用默认配置）
            params: 参数（用于生成唯一键）
            metadata: 额外的元数据

        Returns:
            是否保存成功
        """
        if df is None or df.empty:
            logger.warning(f"跳过空DataFrame缓存: {name}")
            return False

        config = config or CacheConfig(strategy=self.default_strategy)
        key = self._generate_cache_key(name, params)
        cache_path = self._get_cache_path(key)

        try:
            # 检查文件大小限制
            estimated_size = len(df) * len(df.columns) * 8  # 粗略估算
            if estimated_size > config.max_size_mb * 1024 * 1024:
                logger.warning(f"缓存文件超过大小限制 ({config.max_size_mb}MB): {name}")
                return False

            # 保存数据
            df.to_csv(cache_path, index=False, encoding="utf-8")

            # 保存元数据
            file_size = cache_path.stat().st_size
            entry = CacheEntry(key=key, created_at=time.time(), size_bytes=file_size, metadata=metadata or {})
            self._save_metadata(key, entry)

            self.stats["writes"] += 1
            logger.debug(f"缓存已保存: {name} ({len(df)}行, {file_size / 1024:.1f}KB)")
            return True

        except Exception as e:
            logger.error(f"保存缓存失败 [{name}]: {e}")
            return False

    def load_dataframe(
        self, name: str, config: CacheConfig = None, params: dict = None, validate_func=None
    ) -> pd.DataFrame | None:
        """
        从缓存加载DataFrame

        Args:
            name: 缓存名称
            config: 缓存配置（可选）
            params: 参数（用于生成唯一键）
            validate_func: 可选的验证函数 func(df) -> bool

        Returns:
            缓存的DataFrame，如果不存在或已过期则返回None
        """
        config = config or CacheConfig(strategy=self.default_strategy)
        key = self._generate_cache_key(name, params)
        cache_path = self._get_cache_path(key)

        # 检查文件是否存在
        if not cache_path.exists():
            self.stats["misses"] += 1
            return None

        # 检查元数据和过期状态
        entry = self._load_metadata(key)
        if entry and entry.is_expired(config):
            logger.debug(f"缓存已过期，删除: {name}")
            self._remove_cache(key)
            self.stats["misses"] += 1
            return None

        # 加载数据
        try:
            df = pd.read_csv(cache_path, encoding="utf-8")

            # 验证数据完整性
            if config.validate_on_load:
                if df.empty:
                    logger.warning(f"缓存数据为空: {name}")
                    self._remove_cache(key)
                    self.stats["misses"] += 1
                    return None

                if validate_func and not validate_func(df):
                    logger.warning(f"缓存数据验证失败: {name}")
                    self._remove_cache(key)
                    self.stats["misses"] += 1
                    return None

            self.stats["hits"] += 1
            logger.debug(f"缓存命中: {name} ({len(df)}行)")
            return df

        except Exception as e:
            logger.warning(f"加载缓存失败 [{name}]: {e}，将重新获取")
            self._remove_cache(key)
            self.stats["misses"] += 1
            return None

    def save_text(self, content: str, name: str, config: CacheConfig = None, params: dict = None) -> bool:
        """
        保存文本内容到缓存

        Args:
            content: 文本内容
            name: 缓存名称
            config: 缓存配置
            params: 参数

        Returns:
            是否保存成功
        """
        config = config or CacheConfig(strategy=self.default_strategy)
        key = self._generate_cache_key(name, params)
        cache_path = self._get_cache_path(key, extension=".txt")

        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(content)

            file_size = cache_path.stat().st_size
            entry = CacheEntry(key=key, created_at=time.time(), size_bytes=file_size)
            self._save_metadata(key, entry)

            self.stats["writes"] += 1
            return True

        except Exception as e:
            logger.error(f"保存文本缓存失败 [{name}]: {e}")
            return False

    def load_text(self, name: str, config: CacheConfig = None, params: dict = None) -> str | None:
        """
        从缓存加载文本内容

        Args:
            name: 缓存名称
            config: 缓存配置
            params: 参数

        Returns:
            缓存的文本内容，如果不存在或已过期则返回None
        """
        config = config or CacheConfig(strategy=self.default_strategy)
        key = self._generate_cache_key(name, params)
        cache_path = self._get_cache_path(key, extension=".txt")

        if not cache_path.exists():
            self.stats["misses"] += 1
            return None

        entry = self._load_metadata(key)
        if entry and entry.is_expired(config):
            self._remove_cache(key)
            self.stats["misses"] += 1
            return None

        try:
            with open(cache_path, encoding="utf-8") as f:
                content = f.read()

            self.stats["hits"] += 1
            return content

        except Exception as e:
            logger.warning(f"加载文本缓存失败 [{name}]: {e}")
            self._remove_cache(key)
            self.stats["misses"] += 1
            return None

    def invalidate(self, name: str, params: dict = None):
        """
        手动使缓存失效

        Args:
            name: 缓存名称
            params: 参数
        """
        key = self._generate_cache_key(name, params)
        self._remove_cache(key)
        logger.info(f"缓存已手动清除: {name}")

    def clear_all(self):
        """清除所有缓存"""
        for file in self.cache_dir.glob("*"):
            if file.is_file():
                file.unlink()

        self.stats["cleanups"] += 1
        logger.info(f"所有缓存已清除: {self.cache_dir}")

    def cleanup_expired(self):
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

    def _remove_cache(self, key: str):
        """删除缓存文件及其元数据"""
        cache_path = self._get_cache_path(key)
        meta_path = self._get_metadata_path(key)

        for path in [cache_path, meta_path]:
            if path.exists():
                try:
                    path.unlink()
                except Exception as e:
                    logger.warning(f"删除缓存文件失败 [{path.name}]: {e}")

    def get_stats(self) -> dict:
        """获取缓存统计信息"""
        total_files = len(list(self.cache_dir.glob("*")))
        cache_files = len(list(self.cache_dir.glob("*.csv"))) + len(list(self.cache_dir.glob("*.txt")))

        return {
            **self.stats,
            "total_files": total_files,
            "cache_files": cache_files,
            "hit_rate": (
                self.stats["hits"] / (self.stats["hits"] + self.stats["misses"])
                if (self.stats["hits"] + self.stats["misses"]) > 0
                else 0
            ),
        }

    def print_stats(self):
        """打印缓存统计信息"""
        stats = self.get_stats()
        print("\n" + "=" * 60)
        print(" 缓存统计信息")
        print("=" * 60)
        print(f"  缓存命中次数: {stats['hits']}")
        print(f"  缓存未命中次数: {stats['misses']}")
        print(f"  缓存写入次数: {stats['writes']}")
        print(f"  缓存清理次数: {stats['cleanups']}")
        print(f"  命中率: {stats['hit_rate']:.2%}")
        print(f"  缓存文件总数: {stats['cache_files']}")
        print(f"  总文件数: {stats['total_files']}")
        print("=" * 60 + "\n")
