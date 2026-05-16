import urllib.parse
import pandas as pd
from sqlalchemy import create_engine, text
import io


class QuantDBManager:

    def __init__(self, user, password, host, port, db_name):
        # 1. 关键：对密码进行 URL 转义
        # 如果密码里有特殊字符（如 ! @ #），不转义会导致连接串解析失败触发 GBK 报错
        safe_password = urllib.parse.quote_plus(str(password))

        # 2. 构建连接字符串
        self.conn_str = f"postgresql+psycopg2://{user}:{safe_password}@{host}:{port}/{db_name}"

        # 3. 核心修复：强制客户端编码为 UTF8，并增加连接超时
        self.engine = create_engine(
            self.conn_str,
            connect_args={
                'connect_timeout': 30,  # 增加连接超时到30秒
                'client_encoding': 'utf8',
                'keepalives': 1,  # 启用TCP keepalive
                'keepalives_idle': 30,  # 空闲30秒后发送keepalive
                'keepalives_interval': 10,  # keepalive间隔10秒
                'keepalives_count': 5  # keepalive重试次数
            },
            pool_pre_ping=True,  # 每次使用前检查连接是否有效
            pool_size=5,  # 连接池大小
            max_overflow=10,  # 最大溢出连接数
            pool_timeout=60,  # 获取连接超时时间
            pool_recycle=3600  # 连接回收时间（秒）
        )

    def safe_insert_data(self, df, table_name, date_column, today_str, max_retries=3):
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
            print(f"  - [数据库] 表 {table_name} 无有效数据，跳过写入。")
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
                with self.engine.connect() as conn:
                    trans = conn.begin()
                    try:
                        # --- 修改点 2: 使用传入的 today_str 进行删除 ---
                        # 这里的逻辑是正确的，因为我们传入的是交易日
                        delete_query = text(f"DELETE FROM {table_name} WHERE {date_column} = :today")
                        result = conn.execute(delete_query, {"today": today_str})
                        trans.commit()
                        print(f" - [数据库] {table_name} 清理旧记录: {result.rowcount} 条 (日期: {today_str})")

                    except Exception as e:
                        trans.rollback()
                        print(f" - [数据库错误] {table_name} 清理失败: {e}")
                        raise

                try:
                    self._fast_pg_copy(df, table_name)
                    print(f" - [数据库] {table_name} 成功插入新数据: {len(df)} 条 (日期: {today_str})")
                    return  # 成功则返回
                except Exception as e:
                    print(f" - [数据库错误] {table_name} COPY 写入失败: {e}")
                    raise
                    
            except Exception as e:
                if attempt < max_retries:
                    import time
                    wait_time = 2 ** attempt  # 指数退避：2s, 4s, 8s
                    print(f"  - [警告] {table_name} 写入失败 (尝试 {attempt}/{max_retries}): {e}")
                    print(f"  - [警告] {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"  - [错误] {table_name} 写入失败，已达到最大重试次数 ({max_retries})")
                    raise

    def _fast_pg_copy(self, df, table_name, batch_size=5000):
        """
        内部方法：利用 PostgreSQL 的 COPY 协议实现秒级入库
        支持分批写入，避免大数据量导致连接断开
        
        Args:
            df: DataFrame数据
            table_name: 表名
            batch_size: 每批写入的记录数，默认5000条
        """
        total_rows = len(df)
        print(f"  - [数据库] 开始分批写入 {table_name}，共 {total_rows} 条记录，每批 {batch_size} 条")
        
        # 如果数据量小于batch_size，直接写入
        if total_rows <= batch_size:
            self._write_batch(df, table_name)
            return
        
        # 分批写入
        for i in range(0, total_rows, batch_size):
            batch_df = df.iloc[i:i+batch_size]
            try:
                self._write_batch(batch_df, table_name)
                print(f"  - [数据库] {table_name} 批次 {i//batch_size + 1}/{(total_rows-1)//batch_size + 1} 写入成功 ({len(batch_df)} 条)")
            except Exception as e:
                print(f"  - [数据库错误] {table_name} 批次 {i//batch_size + 1} 写入失败: {e}")
                raise e
    
    def _write_batch(self, df, table_name):
        """
        写入单个批次的数据
        """
        output = io.StringIO()
        df.to_csv(output, sep='\t', header=False, index=False, encoding='utf-8')
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
        except Exception as e:
            raw_conn.rollback()
            raise e
        finally:
            cursor.close()
            raw_conn.close()

    def close(self):
        """释放连接池"""
        if self.engine:
            self.engine.dispose()
            print("  - [数据库] 连接池已释放。")

    def execute_query(self, query: str, params=None):
        """执行查询语句"""
        with self.engine.connect() as conn:
            if params:
                result = conn.execute(text(query), params)
            else:
                result = conn.execute(text(query))
            return result.fetchall()

    def execute_update(self, query: str, params=None):
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
            except Exception as e:
                trans.rollback()
                raise e

    def execute_many(self, query: str, values_list):
        """批量执行SQL语句"""
        with self.engine.connect() as conn:
            trans = conn.begin()
            try:
                result = conn.execute(text(query), values_list)
                trans.commit()
                return result.rowcount
            except Exception as e:
                trans.rollback()
                raise e

    def get_table_count(self, table_name: str):
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

    def truncate_table(self, table_name: str):
        """清空表数据"""
        query = f"TRUNCATE TABLE {table_name};"
        return self.execute_update(query)
