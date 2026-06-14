import io
import urllib.parse

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError, IntegrityError, OperationalError, ProgrammingError

from UtilsManager.Exceptions import DatabaseError


class QuantDBManager:
    def __init__(
        self,
        user: str | None = None,
        password: str | None = None,
        host: str | None = None,
        port: int | None = None,
        db_name: str | None = None,
        engine: Engine | None = None,
    ) -> None:
        if engine is not None:
            self.engine = engine
            self.conn_str = str(engine.url)
            return

        safe_password = urllib.parse.quote_plus(str(password))
        self.conn_str = f"postgresql+psycopg2://{user}:{safe_password}@{host}:{port}/{db_name}"
        self.engine = create_engine(
            self.conn_str,
            connect_args={
                "connect_timeout": 30,
                "client_encoding": "utf8",
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5,
            },
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            pool_timeout=60,
            pool_recycle=3600,
        )

    def safe_insert_data(
        self,
        df: pd.DataFrame,
        table_name: str,
        date_column: str,
        today_str: str,
        max_retries: int = 3,
    ) -> None:
        """
        幂等写入：先删除今天的数据，再使用快速 COPY 插入
        支持自动重试机制

        Args:
            df: DataFrame数据
            table_name: 表名
            date_column: 日期列名
            today_str: 业务日期字符串
            max_retries: 最大重试次数，默认3次
        """
        if df is None or df.empty:
            logger.info(f"  - 表 {table_name} 无有效数据，跳过写入。")
            return

        if date_column in df.columns:
            df = df.copy()
            df[date_column] = today_str  # 强制覆盖为业务日期
        else:
            # 如果没有，则添加
            df = df.assign(**{date_column: today_str})

        # 重试逻辑
        for attempt in range(1, max_retries + 1):
            try:
                # 确保每次写入前都执行清理操作，防止主键冲突
                with self.engine.connect() as conn:
                    trans = conn.begin()
                    try:
                        # --- 修改点 2: 使用传入的 today_str 进行删除 ---
                        delete_query = text(f"DELETE FROM {table_name} WHERE {date_column} = :today")
                        result = conn.execute(delete_query, {"today": today_str})
                        logger.info(f" - {table_name} 清理旧记录: {result.rowcount} 条 (日期: {today_str})")
                        trans.commit()

                    except (DBAPIError, OperationalError, IntegrityError) as e:
                        trans.rollback()
                        logger.error(f" - {table_name} 清理失败: {e}")
                        raise DatabaseError(f"清理旧数据 {table_name}", str(e)) from e

                try:
                    self._fast_pg_copy(df, table_name)
                    logger.info(f" - {table_name} 成功插入新数据: {len(df)} 条 (日期: {today_str})")
                    return  # 成功则返回
                except (DBAPIError, OperationalError) as e:
                    logger.error(f" - {table_name} COPY 写入失败: {e}")
                    if attempt < max_retries:
                        with self.engine.connect() as conn:
                            conn.execute(
                                text(f"DELETE FROM {table_name} WHERE {date_column} = :today"), {"today": today_str}
                            )
                            conn.commit()
                    raise DatabaseError(f"COPY写入 {table_name}", str(e)) from e

            except (DBAPIError, OperationalError, IntegrityError) as e:
                if attempt < max_retries:
                    import time

                    wait_time = 2**attempt
                    logger.warning(f"  - {table_name} 写入失败 (尝试 {attempt}/{max_retries}): {e}")
                    logger.warning(f"  - {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"  - {table_name} 写入失败，已达到最大重试次数 ({max_retries})")
                    raise DatabaseError(f"写入 {table_name} (已达最大重试)", str(e)) from e

    def truncate_and_insert(self, df: pd.DataFrame, table_name: str) -> None:
        """
        清表覆盖写入模式：先清空全表，再写入新数据
        适用于基础信息表等只需要保留一份最新数据的场景

        Args:
            df: DataFrame数据
            table_name: 表名
        """
        if df is None or df.empty:
            logger.info(f"  - 表 {table_name} 无有效数据，跳过写入。")
            return

        # 步骤1: 清空全表
        with self.engine.connect() as conn:
            trans = conn.begin()
            try:
                conn.execute(text(f"DELETE FROM {table_name}"))
                trans.commit()
                logger.info(f"  - {table_name} 表已清空。")
            except (DBAPIError, OperationalError) as e:
                trans.rollback()
                logger.error(f"  - {table_name} 清空失败: {e}")
                raise DatabaseError("清空表", str(e)) from e

        # 步骤2: 使用 COPY 协议快速写入
        self._fast_pg_copy(df, table_name)
        logger.info(f"  - {table_name} 成功插入新数据: {len(df)} 条。")

    def _fast_pg_copy(self, df: pd.DataFrame, table_name: str, batch_size: int = 500) -> None:
        """
        内部方法：利用 PostgreSQL 的 COPY 协议实现秒级入库
        支持分批写入，避免大数据量导致连接断开

        Args:
            df: DataFrame数据
            table_name: 表名
            batch_size: 每批写入的记录数，默认500条（A股约5000只股票）
        """
        total_rows = len(df)
        logger.info(f"  - 开始分批写入 {table_name}，共 {total_rows} 条记录，每批 {batch_size} 条")

        # 如果数据量小于batch_size，直接写入
        if total_rows <= batch_size:
            self._write_batch(df, table_name)
            return

        # 分批写入
        for i in range(0, total_rows, batch_size):
            batch_df = df.iloc[i : i + batch_size]
            try:
                self._write_batch(batch_df, table_name)
                logger.info(
                    f"  - {table_name} 批次 {i // batch_size + 1}/{(total_rows - 1) // batch_size + 1} 写入成功 ({len(batch_df)} 条)"
                )
            except (DBAPIError, OperationalError) as e:
                logger.error(f"  - {table_name} 批次 {i // batch_size + 1} 写入失败: {e}")
                raise DatabaseError(f"批次写入 {table_name}", str(e)) from e

    def _write_batch(self, df: pd.DataFrame, table_name: str) -> None:
        """
        写入单个批次的数据
        """
        output = io.StringIO()
        df.to_csv(output, sep="\t", header=False, index=False, encoding="utf-8")
        output.seek(0)

        raw_conn = self.engine.raw_connection()
        try:
            cursor = raw_conn.cursor()

            # 必须显式指定列名，确保 DataFrame 的列顺序与 SQL 语句中的列顺序完全一致
            columns = [f'"{col}"' for col in df.columns]
            copy_sql = f"COPY {table_name} ({', '.join(columns)}) FROM STDIN WITH CSV DELIMITER '\t'"

            # 使用 copy_expert 执行内存流拷贝
            cursor.copy_expert(copy_sql, output)
            raw_conn.commit()
        except (DBAPIError, OperationalError) as e:
            raw_conn.rollback()
            raise DatabaseError("COPY写入", str(e)) from e
        finally:
            cursor.close()
            raw_conn.close()

    def execute_query(self, query: str, params: dict | None = None) -> list:
        """执行查询语句"""
        with self.engine.connect() as conn:
            if params:
                result = conn.execute(text(query), params)
            else:
                result = conn.execute(text(query))
            return result.fetchall()

    def execute_update(self, query: str, params: dict | None = None) -> int:
        """执行更新语句"""
        with self.engine.connect() as conn:
            trans = conn.begin()
            try:
                if params:
                    result = conn.execute(text(query), params)
                else:
                    result = conn.execute(text(query))
                trans.commit()
                return result.rowcount
            except (DBAPIError, OperationalError, ProgrammingError) as e:
                trans.rollback()
                raise DatabaseError("更新", str(e)) from e

    def execute_many(self, query: str, values_list: list[dict]) -> int:
        """批量执行SQL语句"""
        with self.engine.connect() as conn:
            trans = conn.begin()
            try:
                result = conn.execute(text(query), values_list)
                trans.commit()
                return result.rowcount
            except (DBAPIError, OperationalError, ProgrammingError) as e:
                trans.rollback()
                raise DatabaseError("批量执行", str(e)) from e

    def get_table_count(self, table_name: str) -> int:
        """获取表记录数"""
        query = f"SELECT COUNT(*) FROM {table_name}"
        result = self.execute_query(query)
        return result[0][0] if result else 0

    def get_latest_record_date(self, table_name: str, date_column: str) -> str:
        """获取表中指定日期列的最新日期值"""
        query = f"""
        SELECT TO_CHAR({date_column}, 'YYYYMMDD') as latest_date 
        FROM {table_name} 
        ORDER BY {date_column} DESC 
        LIMIT 1
        """
        result = self.execute_query(query)
        if result and len(result) > 0:
            return str(result[0][0]) if result[0][0] else ""
        else:
            return ""

    def check_table_exists(self, table_name: str) -> bool:
        """检查表是否存在"""
        query = """
        SELECT EXISTS (
           SELECT FROM information_schema.tables 
           WHERE table_schema = 'public' 
           AND table_name = :table_name
        );
        """
        result = self.execute_query(query, {"table_name": table_name})
        return result[0][0] if result else False
    
    def get_distinct_count(self, table_name: str, column_name: str) -> int:
        """
        获取指定表中某列的去重记录数
        用于缓存校验时比对数据维度是否发生变化
        
        Args:
            table_name: 表名
            column_name: 需要去重统计的列名
            
        Returns:
            int: 去重后的记录数量，如果查询失败或无数据则返回0
        """
        # 使用参数化查询防止SQL注入，但表名和列名无法直接参数化
        # 这里假设传入的table_name和column_name是受信任的内部变量
        query = f"SELECT COUNT(DISTINCT {column_name}) FROM {table_name}"
        try:
            result = self.execute_query(query)
            return result[0][0] if result and result[0][0] is not None else 0
        except (DBAPIError, OperationalError) as e:
            logger.error(f"  - 获取 {table_name}.{column_name} 去重计数失败: {e}")
            return 0

    def get_table_columns(self, table_name: str) -> list:
        """获取表的所有列名"""
        query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = :table_name
        ORDER BY ordinal_position;
        """
        result = self.execute_query(query, {"table_name": table_name})
        return [row[0] for row in result] if result else []

    def truncate_table(self, table_name: str) -> int:
        """清空表数据"""
        query = f"TRUNCATE TABLE {table_name};"
        return self.execute_update(query)
