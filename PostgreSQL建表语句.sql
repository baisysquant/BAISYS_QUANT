--stock_daily_kline definition
CREATE TABLE public.stock_daily_kline ( trade_date text NULL, symbol text NULL, "open" float8 NULL, "close" float8 NULL, high float8 NULL, low float8 NULL, amount float8 NULL, close_normal float8 NULL, volume float8 NULL, adj_ratio float8 NULL);

-- app_stock_strategy_report definition
CREATE TABLE public.app_stock_strategy_report ( archive_date date NOT NULL, stock_code varchar(20) NOT NULL, stock_name varchar(50) NULL, industry varchar(50) NULL, close_price numeric(12, 2) NULL, is_strong_stock varchar(10) NULL, is_vol_price_rise varchar(10) NULL, consecutive_up_days int4 DEFAULT 0 NULL, high_vol_days int4 DEFAULT 0 NULL, is_top10_industry varchar(10) NULL, is_full_bullish varchar(10) NULL, macd_12269_signal varchar(50) NULL, macd_12269_momentum varchar(50) NULL, macd_12269_dif numeric(12, 4) NULL, macd_second_signal varchar(50) NULL, macd_second_momentum varchar(50) NULL, macd_second_dif numeric(12, 4) NULL, kdj_signal text NULL, cci_signal varchar(100) NULL, rsi_signal varchar(100) NULL, boll_signal varchar(50) NULL, report_buy_count int4 DEFAULT 0 NULL, fund_flow_trend numeric(18, 2) NULL, fund_inflow_5d numeric(18, 2) NULL, fund_inflow_10d numeric(18, 2) NULL, fund_inflow_20d numeric(18, 2) NULL, stock_link text NULL, created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL, macd_second_period_name varchar(20) NULL, CONSTRAINT app_stock_strategy_report_pkey PRIMARY KEY (archive_date, stock_code));
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

-- backtest_kline 定义
CREATE TABLE IF NOT EXISTS public.backtest_kline (
    symbol       VARCHAR(16)    NOT NULL,
    trade_date   DATE           NOT NULL,
    open         NUMERIC(12,2)  NOT NULL,
    high         NUMERIC(12,2)  NOT NULL,
    low          NUMERIC(12,2)  NOT NULL,
    close        NUMERIC(12,2)  NOT NULL,
    volume       NUMERIC(20,0)  NOT NULL,
    amount       NUMERIC(20,2)  NOT NULL,
    adj_factor   NUMERIC(10,6),
    PRIMARY KEY (symbol, trade_date)
);
CREATE INDEX IF NOT EXISTS idx_backtest_kline_date    ON public.backtest_kline (trade_date);
CREATE INDEX IF NOT EXISTS idx_backtest_kline_symbol  ON public.backtest_kline (symbol);

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