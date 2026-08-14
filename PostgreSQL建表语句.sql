--stock_daily_kline definition
CREATE TABLE public.stock_daily_kline ( trade_date text NULL, symbol text NULL, "open" float8 NULL, "close" float8 NULL, high float8 NULL, low float8 NULL, amount float8 NULL, close_normal float8 NULL, volume float8 NULL, adj_ratio float8 NULL);

-- app_stock_strategy_report definition
CREATE TABLE public.app_stock_strategy_report (
    archive_date date NOT NULL,
    stock_code varchar(20) NOT NULL,
    stock_name varchar(50) NULL,
    industry varchar(50) NULL,
    close_price numeric(12,2) NULL,
    is_strong_stock varchar(10) NULL,
    is_vol_price_rise varchar(10) NULL,
    consecutive_up_days int4 DEFAULT 0 NULL,
    high_vol_days int4 DEFAULT 0 NULL,
    is_top10_industry varchar(10) NULL,
    is_full_bullish varchar(10) NULL,
    macd_12269_signal varchar(50) NULL,
    macd_12269_momentum varchar(50) NULL,
    macd_12269_dif numeric(12,4) NULL,
    macd_second_signal varchar(50) NULL,
    macd_second_momentum varchar(50) NULL,
    macd_second_dif numeric(12,4) NULL,
    -- 多因子评分列
    macd_cross varchar(50) NULL,
    macd_momentum varchar(50) NULL,
    dif_slope varchar(50) NULL,
    divergence_signal varchar(100) NULL,
    volume_price_score varchar(50) NULL,
    comprehensive_conclusion text NULL,
    comprehensive_score numeric(12,4) NULL,
    comprehensive_level varchar(20) NULL,
    risk_level varchar(20) NULL,
    macd_trend_type varchar(50) NULL,
    -- 独立技术指标
    kdj_signal text NULL,
    cci_signal varchar(100) NULL,
    rsi_signal varchar(100) NULL,
    boll_signal varchar(50) NULL,
    report_buy_count int4 DEFAULT 0 NULL,
    fund_flow_trend numeric(18,2) NULL,
    fund_inflow_5d numeric(18,2) NULL,
    fund_inflow_10d numeric(18,2) NULL,
    fund_inflow_20d numeric(18,2) NULL,
    stock_link text NULL,
    -- 行业中性化
    industry_signal_tag varchar(50) NULL,
    industry_pctile numeric(12,4) NULL,
    industry_signal_score numeric(12,4) NULL,
    industry_deviation numeric(12,4) NULL,
    -- 背离
    divergence_days int4 NULL,
    divergence_price numeric(12,4) NULL,
    -- 退出策略
    stop_loss numeric(12,4) NULL,
    t1_target numeric(12,4) NULL,
    t2_target numeric(12,4) NULL,
    trailing_stop numeric(12,4) NULL,
    exit_rrr numeric(12,4) NULL,
    created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
    macd_second_period_name varchar(20) NULL,
    CONSTRAINT app_stock_strategy_report_pkey PRIMARY KEY (archive_date, stock_code)
);
CREATE INDEX idx_strategy_report_code ON public.app_stock_strategy_report USING btree (stock_code);
CREATE INDEX idx_strategy_report_date ON public.app_stock_strategy_report USING btree (archive_date);


-- ods_ak_industry_analysis definition
CREATE TABLE ods_ak_industry_analysis ( id serial4 NOT NULL, archive_date date NOT NULL, industry_name varchar(100) NULL, industry_index numeric(12, 2) NULL, change_pct_now numeric(10, 4) NULL, net_inflow_now numeric(20, 2) NULL, total_inflow_money numeric(20, 2) NULL, leading_stock varchar(100) NULL, leading_stock_pct numeric(10, 4) NULL, net_inflow_3d numeric(20, 2) NULL, change_pct_3d numeric(10, 4) NULL, net_inflow_5d numeric(20, 2) NULL, change_pct_5d numeric(10, 4) NULL, net_inflow_10d numeric(20, 2) NULL, change_pct_10d numeric(10, 4) NULL, net_inflow_20d numeric(20, 2) NULL, change_pct_20d numeric(10, 4) NULL, turnover_rate numeric(10, 4) NULL, big_order_confirm varchar(50) NULL, score_fund numeric(10, 4) NULL, score_price numeric(10, 4) NULL, score_turnover numeric(10, 4) NULL, score_trend numeric(10, 2) NULL, industry_signal varchar(50) NULL, created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL, CONSTRAINT ods_ak_industry_analysis_pkey PRIMARY KEY (id));
CREATE INDEX idx_ind_date ON ods_ak_industry_analysis USING btree (archive_date);


-- ods_ak_ranking_stocks definition
CREATE TABLE ods_ak_ranking_stocks ( id int4 NOT NULL, archive_date date NOT NULL, strategy_type varchar(50) NOT NULL, stock_code varchar(20) NOT NULL, stock_name varchar(50) NULL, feature_value numeric(10, 2) NULL, description text NULL, created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL, CONSTRAINT ods_ak_strategy_stocks_combined_pkey PRIMARY KEY (archive_date, strategy_type, stock_code));
CREATE INDEX idx_stock_code_lookup ON ods_ak_ranking_stocks USING btree (stock_code);


-- public.stock_basic_info_sw definition
CREATE TABLE public.stock_basic_info_sw ( id serial4 NOT NULL, industry_code varchar(20) NOT NULL, industry_name varchar(50) NOT NULL, stock_code varchar(20) NOT NULL, stock_name varchar(50) NOT NULL, weight float4 DEFAULT 0.0 NULL, record_date date NOT NULL, CONSTRAINT stock_basic_info_sw_pkey PRIMARY KEY (id), CONSTRAINT uk_ind_stock_date UNIQUE (industry_code, stock_code, record_date));
CREATE INDEX idx_sbi_industry_name ON public.stock_basic_info_sw USING btree (industry_name);
CREATE INDEX idx_sbi_record_date ON public.stock_basic_info_sw USING btree (record_date);
CREATE INDEX idx_sbi_stock_code ON public.stock_basic_info_sw USING btree (stock_code);

-- ── v1.1.0 migration: 复权因子列 ──────────────────────────────
ALTER TABLE public.stock_daily_kline ADD COLUMN IF NOT EXISTS adj_factor float8 DEFAULT 1.0;
UPDATE public.stock_daily_kline SET adj_factor = COALESCE(adj_ratio, 1.0) WHERE adj_factor = 1.0 AND adj_ratio IS NOT NULL;


-- backtest_calibration_log 定义
CREATE TABLE IF NOT EXISTS public.backtest_calibration_log (
    id              SERIAL PRIMARY KEY,
    run_time        TIMESTAMP   NOT NULL DEFAULT NOW(),
    frequency       VARCHAR(16) NOT NULL,
    lookback_days   INT         NOT NULL,
    out_of_sample_days INT      NOT NULL,
    initial_cash    NUMERIC(14,2) NOT NULL,
    params          JSONB       NOT NULL DEFAULT '{}'::jsonb,
    sharpe          NUMERIC(8,4),
    total_return    NUMERIC(8,4),
    max_drawdown    NUMERIC(8,4),
    status          VARCHAR(16) NOT NULL DEFAULT 'success'
);
CREATE INDEX IF NOT EXISTS idx_backtest_calibration_log_run_time
    ON public.backtest_calibration_log (run_time DESC);

-- 因子 IC 历史（衰减监控）
CREATE TABLE IF NOT EXISTS public.ods_factor_ic_history (
    id SERIAL PRIMARY KEY,
    factor_name VARCHAR(20) NOT NULL,
    check_date DATE NOT NULL,
    rolling_ic_mean FLOAT,
    rolling_ic_std FLOAT,
    icir FLOAT,
    rank_ic FLOAT,
    normal_ic FLOAT,
    decile_spread FLOAT,
    decile_monotonicity FLOAT,
    is_decayed BOOLEAN DEFAULT FALSE,
    current_weight FLOAT,
    suggested_weight FLOAT
);
CREATE INDEX IF NOT EXISTS idx_ic_factor_date
    ON public.ods_factor_ic_history (factor_name, check_date);

-- 基准指数日线（如上证综指 000001.SH）
CREATE TABLE IF NOT EXISTS public.ods_index_daily (
    index_code VARCHAR(20) NOT NULL,
    trade_date DATE NOT NULL,
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    volume FLOAT,
    amount FLOAT,
    PRIMARY KEY (index_code, trade_date)
);

-- 质量/成长因子（季度）
-- P0-8①：新增 disclosure_date（披露日/公告日）——record_date 是报告期末，披露日才是数据可得性时点，
-- as-of 查询须按 披露日 <= 查询日 过滤，否则历史复盘会用到尚未披露的财报（前视偏差）。
-- 披露日默认按监管披露截止日（公告日最晚可能值，保守无前视）回填，可被真实公告日覆盖。
CREATE TABLE IF NOT EXISTS public.ods_financial_quality (
    symbol VARCHAR(20) NOT NULL,
    record_date DATE NOT NULL,
    disclosure_date DATE,
    roe FLOAT,
    gross_profit_margin FLOAT,
    net_profit_margin FLOAT,
    revenue_growth_rate FLOAT,
    net_profit_growth_rate FLOAT,
    PRIMARY KEY (symbol, record_date)
);

-- v2.1 migration: 已有部署补 disclosure_date 列
ALTER TABLE public.ods_financial_quality ADD COLUMN IF NOT EXISTS disclosure_date DATE;
CREATE INDEX IF NOT EXISTS idx_fq_disclosure ON public.ods_financial_quality (disclosure_date, record_date);

-- ── v2.0 migration: 数据质量日志 ────────────────────────────────
CREATE TABLE IF NOT EXISTS public.dash_quality_log (
    id SERIAL PRIMARY KEY,
    trade_date VARCHAR(16) NOT NULL,
    step_name VARCHAR(64) NOT NULL,
    check_name VARCHAR(64) NOT NULL,
    status VARCHAR(16) NOT NULL DEFAULT 'pass',
    metric FLOAT,
    threshold FLOAT,
    detail TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_ql_date ON public.dash_quality_log (trade_date);
CREATE INDEX IF NOT EXISTS idx_ql_step ON public.dash_quality_log (step_name);

-- ── v2.0 migration: DW 层因子日宽表 ─────────────────────────────
CREATE TABLE IF NOT EXISTS public.dwd_factor_daily (
    trade_date DATE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    industry VARCHAR(50),
    composite_score FLOAT,
    composite_rank INT,
    factors JSONB DEFAULT '{}'::jsonb,       -- {"momentum": 0.5, "quality": -0.3, ...}
    factor_z JSONB DEFAULT '{}'::jsonb,      -- 行业内 Z-Score
    factor_raw JSONB DEFAULT '{}'::jsonb,    -- 原始值 {"momentum_raw": 0.05, "roe": 0.15}
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (trade_date, symbol)
);

-- ── v2.0 migration: 实验版本管理 ───────────────────────────────
CREATE TABLE IF NOT EXISTS public.dash_run_log (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(64) NOT NULL UNIQUE,
    trade_date VARCHAR(16) NOT NULL,
    pipeline_name VARCHAR(64) NOT NULL DEFAULT '',
    status VARCHAR(16) NOT NULL DEFAULT 'running',
    started_at TIMESTAMP NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMP,
    duration_seconds FLOAT,
    config_hash VARCHAR(64),
    stock_pool_hash VARCHAR(64),
    stock_count INT DEFAULT 0,
    score_summary JSONB DEFAULT '{}'::jsonb,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_runlog_date ON public.dash_run_log (trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_runlog_status ON public.dash_run_log (status);

-- ── v2.0 migration: DAG pipeline checkpoint ─────────────────────
CREATE TABLE IF NOT EXISTS public.dash_pipeline_checkpoint (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(64) NOT NULL,
    pipeline_name VARCHAR(64) NOT NULL,
    trade_date VARCHAR(16) NOT NULL,
    step_name VARCHAR(64) NOT NULL,
    status VARCHAR(16) NOT NULL DEFAULT 'pending',
    started_at TIMESTAMP,
    finished_at TIMESTAMP,
    duration_seconds FLOAT,
    error_message TEXT,
    ctx_json JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (run_id, step_name)
);
CREATE INDEX IF NOT EXISTS idx_ckpt_run ON public.dash_pipeline_checkpoint (run_id);
CREATE INDEX IF NOT EXISTS idx_ckpt_date ON public.dash_pipeline_checkpoint (trade_date);
CREATE INDEX IF NOT EXISTS idx_ckpt_pname_date ON public.dash_pipeline_checkpoint (pipeline_name, trade_date);

-- 估值/市值因子（日频）
CREATE TABLE IF NOT EXISTS public.ods_financial_valuation (
    symbol VARCHAR(20) NOT NULL,
    trade_date DATE NOT NULL,
    pe FLOAT,
    pe_ttm FLOAT,
    pb FLOAT,
    total_mv FLOAT,
    circ_mv FLOAT,
    PRIMARY KEY (symbol, trade_date)
);
CREATE INDEX IF NOT EXISTS idx_fv_date ON public.ods_financial_valuation (trade_date);

-- ── v2.1 migration: 申万一级行业映射表（P0-7 ① 行业一级中性化） ────────
-- 独立映射表（不修改 stock_basic_info_sw 的申万二级行业语义）：
-- 每只股票 → 申万一级行业（l1_name 与 MacroFactorFetcher._SW1_MACRO_CLASS
-- 的键一致，供宏观 tilt 映射与行业一级中性化使用）。
-- 由 DataManager/SwIndustrySync.py 同步填充（AkShare 申万一级成分股）。
CREATE TABLE IF NOT EXISTS public.stock_basic_info_sw_l1 (
    id serial4 NOT NULL,
    l1_code varchar(20) NOT NULL,
    l1_name varchar(50) NOT NULL,
    stock_code varchar(20) NOT NULL,
    stock_name varchar(50) NULL,
    record_date date NOT NULL,
    CONSTRAINT stock_basic_info_sw_l1_pkey PRIMARY KEY (id),
    CONSTRAINT uk_l1_ind_stock_date UNIQUE (l1_code, stock_code, record_date)
);
CREATE INDEX IF NOT EXISTS idx_sbi_l1_name ON public.stock_basic_info_sw_l1 USING btree (l1_name);
CREATE INDEX IF NOT EXISTS idx_sbi_l1_stock_code ON public.stock_basic_info_sw_l1 USING btree (stock_code);
CREATE INDEX IF NOT EXISTS idx_sbi_l1_record_date ON public.stock_basic_info_sw_l1 USING btree (record_date);