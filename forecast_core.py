# -*- coding: utf-8 -*-
"""
通用金融时间序列预测核心模块：ADF → auto_arima → GARCH

支持 A 股、美股、场外基金、商品/期货及本地 CSV。
供命令行脚本 silver_price_arima_garch.py 与 Streamlit app.py 共用。
"""

from __future__ import annotations

import io
import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

LogFn = Callable[[str], None]

# 市场类型常量
MARKET_CN_STOCK = "cn_stock"
MARKET_US_STOCK = "us_stock"
MARKET_CN_FUND = "cn_fund"
MARKET_COMMODITY = "commodity"
MARKET_CSV = "csv"

MARKET_LABELS = {
    MARKET_CN_STOCK: "A股 / CN Stock",
    MARKET_US_STOCK: "美股 / US Stock",
    MARKET_CN_FUND: "场外基金 / CN Fund",
    MARKET_COMMODITY: "商品期货 / Commodity",
    MARKET_CSV: "本地 CSV / Upload CSV",
}


@dataclass
class ForecastConfig:
    """预测任务配置。"""

    asset_name: str = "Asset"
    ticker: str = "600519"
    market: str = MARKET_CN_STOCK
    data_source: str = "auto"
    start_date: str = "2015-01-01"
    forecast_days: int = 30
    plot_start_date: str | None = None  # None = 自动取最近 6 个月
    output_dir: str = "output/default"
    date_column: str = "Date"
    price_column: str = "Close"
    csv_path: str | None = None
    csv_content: bytes | None = None
    currency_label: str = "CNY"
    yahoo_max_retries: int = 3
    yahoo_retry_wait: int = 10
    auto_arima_max_p: int = 5
    auto_arima_max_q: int = 5
    auto_arima_max_d: int = 2
    auto_arima_ic: str = "aic"
    auto_arima_stepwise: bool = True
    stooq_tickers: list[str] = field(default_factory=list)


@dataclass
class ForecastResult:
    """管道运行结果，供 Streamlit 或脚本展示。"""

    config: ForecastConfig
    prices: pd.Series
    log_returns: pd.Series
    arima_order: tuple[int, int, int]
    aic: float
    forecast_mean: pd.Series
    forecast_ci: pd.DataFrame
    fitted_values: pd.Series
    volatility_forecast: np.ndarray
    adf_suggested_d: int
    adf_price_result: dict
    adf_return_result: dict
    output_dir: str
    chart_paths: list[str] = field(default_factory=list)
    csv_paths: dict[str, str] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)


def _default_log(msg: str) -> None:
    print(msg)


def _make_logger(logs: list[str]) -> LogFn:
    def log(msg: str) -> None:
        logs.append(msg)
        print(msg)

    return log


def _start_date_compact(start_date: str) -> str:
    return start_date.replace("-", "")


def _filter_prices_by_start_date(prices: pd.Series, start_date: str) -> pd.Series:
    return prices[prices.index >= pd.Timestamp(start_date)]


def normalize_price_index(prices: pd.Series) -> pd.Series:
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index)
    if getattr(prices.index, "tz", None) is not None:
        prices.index = prices.index.tz_localize(None)
    return prices.sort_index()


def _load_from_csv_content(
    content: bytes | str,
    date_column: str,
    price_column: str,
    log: LogFn,
) -> pd.Series:
    if isinstance(content, bytes):
        df = pd.read_csv(io.BytesIO(content), parse_dates=[date_column])
    else:
        df = pd.read_csv(content, parse_dates=[date_column])
    df = df.sort_values(date_column).set_index(date_column)
    prices = df[price_column].astype(float).dropna()
    log(f"[Data] Loaded CSV: {len(prices)} rows")
    return prices


def _load_cn_stock(ticker: str, start_date: str, log: LogFn) -> pd.Series | None:
    try:
        import akshare as ak
    except ImportError:
        log("[AKShare] Not installed. Run: pip install akshare")
        return None

    log(f"[Data] AKShare A-share: {ticker}")
    try:
        df = ak.stock_zh_a_hist(
            symbol=ticker,
            period="daily",
            start_date=_start_date_compact(start_date),
            adjust="qfq",
        )
        if df is None or df.empty:
            return None
        date_col = "日期" if "日期" in df.columns else "date"
        close_col = "收盘" if "收盘" in df.columns else "close"
        df[date_col] = pd.to_datetime(df[date_col])
        prices = df.sort_values(date_col).set_index(date_col)[close_col].astype(float).dropna()
        log(f"[Data] A-share OK: {len(prices)} rows")
        return prices
    except Exception as exc:
        log(f"[Data] A-share failed: {type(exc).__name__}: {exc}")
        return None


def _load_cn_fund(ticker: str, start_date: str, log: LogFn) -> pd.Series | None:
    try:
        import akshare as ak
    except ImportError:
        log("[AKShare] Not installed. Run: pip install akshare")
        return None

    log(f"[Data] AKShare CN fund: {ticker}")
    try:
        df = ak.fund_open_fund_info_em(symbol=ticker, indicator="单位净值走势")
        if df is None or df.empty:
            return None
        date_col = "净值日期" if "净值日期" in df.columns else df.columns[0]
        value_col = "单位净值" if "单位净值" in df.columns else df.columns[1]
        df[date_col] = pd.to_datetime(df[date_col])
        prices = df.sort_values(date_col).set_index(date_col)[value_col].astype(float).dropna()
        prices = _filter_prices_by_start_date(prices, start_date)
        log(f"[Data] CN fund OK: {len(prices)} rows")
        return prices
    except Exception as exc:
        log(f"[Data] CN fund failed: {type(exc).__name__}: {exc}")
        return None


def _load_yahoo(ticker: str, start_date: str, config: ForecastConfig, log: LogFn) -> pd.Series | None:
    log(f"[Data] Yahoo Finance: {ticker} from {start_date}")
    for attempt in range(1, config.yahoo_max_retries + 1):
        try:
            yf_ticker = yf.Ticker(ticker)
            df = yf_ticker.history(start=start_date, auto_adjust=True)
            if df.empty:
                df = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
            if df.empty:
                log(f"  [Yahoo] attempt {attempt}: empty")
            else:
                if isinstance(df.columns, pd.MultiIndex):
                    prices = df["Close"].iloc[:, 0]
                else:
                    prices = df["Close"]
                prices = prices.dropna()
                prices.index = pd.to_datetime(prices.index).tz_localize(None)
                log(f"[Data] Yahoo OK: {len(prices)} rows")
                return prices
        except Exception as exc:
            log(f"  [Yahoo] attempt {attempt}: {type(exc).__name__}: {exc}")
        if attempt < config.yahoo_max_retries:
            time.sleep(config.yahoo_retry_wait)
    return None


def _download_stooq_csv(url: str) -> pd.DataFrame | None:
    request = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"},
    )
    with urlopen(request, timeout=30) as response:
        content = response.read().decode("utf-8", errors="ignore")
    if "Date" not in content or "Close" not in content:
        return None
    return pd.read_csv(io.StringIO(content))


def _load_stooq(tickers: list[str], start_date: str, log: LogFn) -> pd.Series | None:
    for ticker in tickers:
        url = f"https://stooq.com/q/d/l/?s={ticker}&i=d"
        log(f"[Data] Stooq: {ticker}")
        try:
            df = _download_stooq_csv(url)
        except Exception as exc:
            log(f"  [Stooq] {ticker} failed: {exc}")
            continue
        if df is None or df.empty:
            continue
        df["Date"] = pd.to_datetime(df["Date"])
        prices = df.sort_values("Date").set_index("Date")["Close"].astype(float).dropna()
        prices = _filter_prices_by_start_date(prices, start_date)
        if not prices.empty:
            log(f"[Data] Stooq OK: {len(prices)} rows")
            return prices
    return None


def _load_commodity_akshare(ticker: str, start_date: str, log: LogFn) -> pd.Series | None:
    try:
        import akshare as ak
    except ImportError:
        return None

    symbol_map = {"SI=F": "SI", "SI": "SI", "SLV": None}
    sina_sym = symbol_map.get(ticker.upper(), ticker.replace("=F", ""))

    if sina_sym:
        log(f"[Data] AKShare futures (Sina): {sina_sym}")
        try:
            df = ak.futures_foreign_hist(symbol=sina_sym)
            if df is not None and not df.empty and "close" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                prices = df.sort_values("date").set_index("date")["close"].astype(float).dropna()
                prices = _filter_prices_by_start_date(prices, start_date)
                if not prices.empty:
                    log(f"[Data] Futures OK: {len(prices)} rows")
                    return prices
        except Exception as exc:
            log(f"  [Futures] failed: {exc}")

    em_sym = "SI00Y" if "SI" in ticker.upper() else None
    if em_sym:
        log(f"[Data] AKShare futures (Eastmoney): {em_sym}")
        try:
            df = ak.futures_global_hist_em(symbol=em_sym)
            if df is not None and not df.empty and "日期" in df.columns:
                close_col = "最新价" if "最新价" in df.columns else "收盘"
                df["日期"] = pd.to_datetime(df["日期"])
                prices = df.sort_values("日期").set_index("日期")[close_col].astype(float).dropna()
                prices = _filter_prices_by_start_date(prices, start_date)
                if not prices.empty:
                    return prices
        except Exception as exc:
            log(f"  [EM futures] failed: {exc}")
    return None


def _stooq_candidates(ticker: str, custom: list[str]) -> list[str]:
    if custom:
        return custom
    t = ticker.upper()
    if t in ("SLV",):
        return ["slv.us"]
    if t in ("SI=F", "SI", "XAGUSD"):
        return ["xagusd", "slv.us"]
    if "." not in ticker:
        return [f"{ticker.lower()}.us", ticker.lower()]
    return [ticker.lower()]


def load_prices(config: ForecastConfig, log: LogFn = _default_log) -> pd.Series:
    """按市场类型加载价格序列。"""
    prices = None

    if config.market == MARKET_CSV:
        if config.csv_content:
            prices = _load_from_csv_content(
                config.csv_content, config.date_column, config.price_column, log
            )
        elif config.csv_path:
            prices = _load_from_csv_content(
                config.csv_path, config.date_column, config.price_column, log
            )
        else:
            raise ValueError("CSV market requires csv_path or uploaded file")

    elif config.market == MARKET_CN_STOCK:
        prices = _load_cn_stock(config.ticker, config.start_date, log)

    elif config.market == MARKET_CN_FUND:
        prices = _load_cn_fund(config.ticker, config.start_date, log)

    elif config.market == MARKET_US_STOCK:
        prices = _load_yahoo(config.ticker, config.start_date, config, log)

    elif config.market == MARKET_COMMODITY:
        if config.data_source in ("auto", "akshare"):
            prices = _load_commodity_akshare(config.ticker, config.start_date, log)
        if prices is None and config.data_source in ("auto", "stooq"):
            stooq = _stooq_candidates(config.ticker, config.stooq_tickers)
            prices = _load_stooq(stooq, config.start_date, log)
        if prices is None and config.data_source in ("auto", "yahoo"):
            prices = _load_yahoo(config.ticker, config.start_date, config, log)

    if prices is None or prices.empty:
        raise ValueError(
            f"Failed to load data for {config.asset_name} ({config.ticker}). "
            "Try another market/data source or upload a CSV."
        )

    prices.name = config.ticker
    return normalize_price_index(prices)


def safe_to_csv(df: pd.DataFrame, filepath: str, **kwargs) -> str:
    folder = os.path.dirname(filepath)
    if folder:
        os.makedirs(folder, exist_ok=True)
    try:
        df.to_csv(filepath, **kwargs)
        return filepath
    except PermissionError:
        base, ext = os.path.splitext(filepath)
        alt = f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        df.to_csv(alt, **kwargs)
        return alt


def build_future_trading_dates(last_date: pd.Timestamp, n_steps: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_steps)


def align_forecast_dates(
    prices: pd.Series, forecast_mean: pd.Series, forecast_ci: pd.DataFrame
) -> tuple[pd.Series, pd.DataFrame]:
    future_dates = build_future_trading_dates(pd.Timestamp(prices.index[-1]), len(forecast_mean))
    aligned_mean = pd.Series(forecast_mean.values, index=future_dates, name="forecast")
    aligned_ci = forecast_ci.copy()
    aligned_ci.index = future_dates
    return aligned_mean, aligned_ci


def compute_returns(prices: pd.Series) -> pd.Series:
    r = np.log(prices / prices.shift(1)).dropna()
    r.name = "log_return"
    return r


def compute_forecast_log_returns(prices: pd.Series, forecast_prices: pd.Series) -> pd.Series:
    path = pd.concat([prices.iloc[[-1]], forecast_prices])
    fr = np.log(path / path.shift(1)).iloc[1:]
    fr.index = forecast_prices.index
    fr.name = "forecast_log_return"
    return fr


def run_adf_test(series: pd.Series, series_name: str, log: LogFn = _default_log) -> dict:
    result = adfuller(series.dropna(), autolag="AIC")
    out = {
        "name": series_name,
        "adf_stat": result[0],
        "p_value": result[1],
        "lags": result[2],
        "nobs": result[3],
        "crit_1pct": result[4]["1%"],
        "crit_5pct": result[4]["5%"],
        "is_stationary": result[1] < 0.05,
    }
    log(f"[ADF] {series_name}: p={out['p_value']:.6f}, stationary={out['is_stationary']}")
    return out


def determine_differencing_order(prices: pd.Series, max_d: int, log: LogFn) -> int:
    log("[Step 1] ADF unit root test")
    run_adf_test(prices, "Price Level", log)
    current = prices.copy()
    for d in range(1, max_d + 1):
        current = current.diff().dropna()
        adf = run_adf_test(current, f"Differenced (d={d})", log)
        if adf["is_stationary"]:
            log(f"[ADF] Suggested d={d}")
            return d
    return max_d


def fit_auto_arima_and_forecast(
    prices: pd.Series, config: ForecastConfig, log: LogFn
) -> tuple:
    from pmdarima import auto_arima

    log("[Step 2] auto_arima model selection")
    auto_model = auto_arima(
        prices,
        start_p=0,
        start_q=0,
        max_p=config.auto_arima_max_p,
        max_d=config.auto_arima_max_d,
        max_q=config.auto_arima_max_q,
        seasonal=False,
        stepwise=config.auto_arima_stepwise,
        information_criterion=config.auto_arima_ic,
        suppress_warnings=True,
        error_action="ignore",
        trace=False,
    )
    order = auto_model.order
    aic = float(auto_model.aic())
    log(f"[ARIMA] Best order: ARIMA{order}, AIC={aic:.4f}")

    fitted_values = auto_model.arima_res_.fittedvalues
    sm_fc = auto_model.arima_res_.get_forecast(steps=config.forecast_days)
    forecast_mean, forecast_ci = align_forecast_dates(
        prices, sm_fc.predicted_mean, sm_fc.conf_int()
    )
    forecast_mean = forecast_mean.astype(float)
    forecast_ci = forecast_ci.astype(float)
    forecast_ci.columns = ["lower", "upper"]
    log(
        f"[Forecast] {config.forecast_days} days: "
        f"{forecast_mean.min():.4f} ~ {forecast_mean.max():.4f}"
    )
    return auto_model, forecast_mean, forecast_ci, fitted_values, order, aic


def fit_garch(residuals: pd.Series, horizon: int, log: LogFn) -> np.ndarray:
    log("[Step 3] GARCH(1,1) volatility model")
    scaled = residuals.dropna() * 100
    garch = arch_model(scaled, mean="Zero", vol="Garch", p=1, q=1, dist="normal")
    result = garch.fit(disp="off")
    var_fc = result.forecast(horizon=horizon).variance.iloc[-1].values
    return np.sqrt(var_fc) / 100


def resolve_plot_start_date(config: ForecastConfig, prices: pd.Series) -> str | None:
    if config.plot_start_date:
        return config.plot_start_date
    last = pd.Timestamp(prices.index[-1])
    return (last - pd.DateOffset(months=6)).strftime("%Y-%m-%d")


def setup_plot_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def prepend_forecast_anchor(history: pd.Series, forecast: pd.Series) -> pd.Series:
    return pd.concat([history.iloc[[-1]], forecast])


def plot_results(
    prices: pd.Series,
    fitted_values: pd.Series,
    forecast_mean: pd.Series,
    forecast_ci: pd.DataFrame,
    log_returns: pd.Series,
    volatility_forecast: np.ndarray,
    config: ForecastConfig,
    log: LogFn = _default_log,
) -> list[str]:
    setup_plot_style()
    out_dir = config.output_dir
    os.makedirs(out_dir, exist_ok=True)
    name = config.asset_name
    curr = config.currency_label
    plot_start = resolve_plot_start_date(config, prices)
    paths: list[str] = []

    def zoom(ax):
        if plot_start:
            ax.set_xlim(
                left=pd.Timestamp(plot_start),
                right=forecast_mean.index[-1] + pd.Timedelta(days=5),
            )

    split_date = prices.index[-1]
    fc_price = prepend_forecast_anchor(prices, forecast_mean)
    fc_returns = compute_forecast_log_returns(prices, forecast_mean)
    fc_ret_line = prepend_forecast_anchor(log_returns, fc_returns)

    charts = [
        (
            "01_arima_forecast.png",
            lambda ax: (
                prices.plot(ax=ax, label=f"Historical {name}", color="steelblue", lw=1.2),
                fitted_values.plot(ax=ax, label="In-Sample Fit", color="orange", alpha=0.8),
                ax.plot(fc_price.index, fc_price.values, color="crimson", lw=2, label="Forecast", zorder=5),
                ax.fill_between(
                    forecast_ci.index,
                    forecast_ci["lower"],
                    forecast_ci["upper"],
                    color="crimson",
                    alpha=0.15,
                    label="95% CI",
                ),
                ax.set_title(f"{name} ARIMA Forecast"),
                ax.set_ylabel(f"Price ({curr})"),
                zoom(ax),
            ),
        ),
        (
            "02_log_returns.png",
            lambda ax: (
                log_returns.plot(ax=ax, color="darkgreen", lw=0.8),
                ax.set_title(f"{name} Log Returns"),
            ),
        ),
        (
            "05_price_history_vs_forecast.png",
            lambda ax: (
                prices.plot(ax=ax, color="orange", lw=1.2, label="Historical"),
                ax.plot(fc_price.index, fc_price.values, color="red", lw=2, label="Forecast", zorder=5),
                ax.axvline(split_date, color="gray", ls="--", lw=1, alpha=0.7),
                ax.set_title(f"{name}: Historical vs Forecast"),
                ax.set_ylabel(f"Price ({curr})"),
                zoom(ax),
            ),
        ),
        (
            "06_returns_history_vs_forecast.png",
            lambda ax: (
                log_returns.plot(ax=ax, color="orange", lw=0.9, label="Historical Return"),
                ax.plot(fc_ret_line.index, fc_ret_line.values, color="red", lw=1.5, label="Forecast Return", zorder=5),
                ax.axvline(split_date, color="gray", ls="--", lw=1, alpha=0.7),
                ax.axhline(0, color="black", lw=0.6, alpha=0.4),
                ax.set_title(f"{name} Returns: Historical vs Forecast"),
                zoom(ax),
            ),
        ),
    ]

    for fname, draw in charts:
        fig, ax = plt.subplots(figsize=(12, 4 if "return" in fname else 5))
        draw(ax)
        ax.set_xlabel("Date")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
        log(f"[Chart] Saved: {path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(log_returns.dropna(), lags=40, ax=axes[0])
    axes[0].set_title("ACF")
    plot_pacf(log_returns.dropna(), lags=40, ax=axes[1], method="ywm")
    axes[1].set_title("PACF")
    fig.tight_layout()
    p3 = os.path.join(out_dir, "03_acf_pacf.png")
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    paths.append(p3)

    fig, ax = plt.subplots(figsize=(10, 4))
    days = np.arange(1, len(volatility_forecast) + 1)
    ax.plot(days, volatility_forecast, marker="o", color="purple", lw=1.5)
    ax.set_title("GARCH Volatility Forecast")
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("Conditional Volatility")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p4 = os.path.join(out_dir, "04_garch_volatility.png")
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    paths.append(p4)

    return paths


def run_forecast_pipeline(config: ForecastConfig) -> ForecastResult:
    """执行完整 ADF → auto_arima → GARCH 流程。"""
    logs: list[str] = []
    log = _make_logger(logs)
    os.makedirs(config.output_dir, exist_ok=True)

    log(f"=== Forecast: {config.asset_name} ({config.ticker}) ===")

    prices = load_prices(config, log)
    df_prices = prices.reset_index()
    df_prices.columns = [config.date_column, config.price_column]
    prices_csv = safe_to_csv(
        df_prices, os.path.join(config.output_dir, "prices.csv"), index=False, encoding="utf-8-sig"
    )
    log(f"[Output] Prices CSV: {prices_csv}")

    log_returns = compute_returns(prices)
    adf_return = run_adf_test(log_returns, "Log Returns", log)
    adf_d = determine_differencing_order(prices, config.auto_arima_max_d, log)
    adf_price = run_adf_test(prices, "Price (reference)", log)

    model, fc_mean, fc_ci, fitted, order, aic = fit_auto_arima_and_forecast(prices, config, log)
    residuals = model.arima_res_.resid
    acorr_ljungbox(residuals.dropna(), lags=[10], return_df=True)
    vol_fc = fit_garch(residuals, config.forecast_days, log)

    chart_paths = plot_results(
        prices, fitted, fc_mean, fc_ci, log_returns, vol_fc, config, log
    )

    fc_returns = compute_forecast_log_returns(prices, fc_mean)
    forecast_df = pd.DataFrame({
        "forecast_price": fc_mean,
        "lower_95": fc_ci["lower"],
        "upper_95": fc_ci["upper"],
        "forecast_return": fc_returns,
    })
    arima_path = safe_to_csv(
        forecast_df,
        os.path.join(config.output_dir, "arima_forecast.csv"),
        encoding="utf-8-sig",
    )
    vol_df = pd.DataFrame({
        "day": np.arange(1, len(vol_fc) + 1),
        "conditional_volatility": vol_fc,
    })
    garch_path = safe_to_csv(
        vol_df,
        os.path.join(config.output_dir, "garch_volatility_forecast.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    log("[Done] Pipeline complete.")

    return ForecastResult(
        config=config,
        prices=prices,
        log_returns=log_returns,
        arima_order=order,
        aic=aic,
        forecast_mean=fc_mean,
        forecast_ci=fc_ci,
        fitted_values=fitted,
        volatility_forecast=vol_fc,
        adf_suggested_d=adf_d,
        adf_price_result=adf_price,
        adf_return_result=adf_return,
        output_dir=config.output_dir,
        chart_paths=chart_paths,
        csv_paths={"prices": prices_csv, "arima": arima_path, "garch": garch_path},
        logs=logs,
    )


def silver_default_config() -> ForecastConfig:
    """原银价脚本默认配置。"""
    return ForecastConfig(
        asset_name="Silver",
        ticker="SI=F",
        market=MARKET_COMMODITY,
        data_source="auto",
        start_date="2015-01-01",
        forecast_days=30,
        plot_start_date="2026-01-01",
        output_dir="output",
        currency_label="USD",
        stooq_tickers=["xagusd", "slv.us"],
    )
