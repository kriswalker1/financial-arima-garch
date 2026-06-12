# -*- coding: utf-8 -*-
"""
金融时间序列预测 — Streamlit 网页界面

启动: streamlit run app.py
"""

from __future__ import annotations

import os
import re

import pandas as pd
import streamlit as st

from forecast_core import (
    MARKET_CN_FUND,
    MARKET_CN_STOCK,
    MARKET_COMMODITY,
    MARKET_CSV,
    MARKET_LABELS,
    MARKET_US_STOCK,
    ForecastConfig,
    run_forecast_pipeline,
)

st.set_page_config(
    page_title="ARIMA+GARCH Forecast",
    page_icon="📈",
    layout="wide",
)

TICKER_HINTS = {
    MARKET_CN_STOCK: ("600519", "000001", "510300"),
    MARKET_US_STOCK: ("AAPL", "TSLA", "MSFT"),
    MARKET_CN_FUND: ("161725", "110022", "005827"),
    MARKET_COMMODITY: ("SI=F", "SLV", "GC=F"),
    MARKET_CSV: ("upload CSV", "", ""),
}

CURRENCY_MAP = {
    MARKET_CN_STOCK: "CNY",
    MARKET_CN_FUND: "CNY",
    MARKET_US_STOCK: "USD",
    MARKET_COMMODITY: "USD",
    MARKET_CSV: "CNY",
}


def safe_dir_name(ticker: str) -> str:
    return re.sub(r"[^\w\-]", "_", ticker) or "asset"


def main() -> None:
    st.title("📈 金融时间序列预测 / Financial Time Series Forecast")
    st.caption("ADF → auto_arima → GARCH | 仅供学习研究 / For education only")

    with st.sidebar:
        st.header("⚙️ 参数 / Settings")

        market = st.selectbox(
            "市场类型 / Market",
            options=list(MARKET_LABELS.keys()),
            format_func=lambda k: MARKET_LABELS[k],
            index=0,
        )

        hints = TICKER_HINTS.get(market, ("",))
        ticker = st.text_input(
            "代码 / Ticker",
            value=hints[0] if hints else "",
            help=f"示例 / Examples: {', '.join(h for h in hints if h)}",
        ).strip()

        asset_name = st.text_input(
            "名称 / Asset Name",
            value=ticker or "My Asset",
        ).strip()

        start_date = st.date_input(
            "历史起始日 / History Start",
            value=pd.Timestamp("2015-01-01"),
        ).strftime("%Y-%m-%d")

        forecast_days = st.slider(
            "预测天数 / Forecast Days",
            min_value=5,
            max_value=90,
            value=30,
        )

        plot_mode = st.radio(
            "图表时间轴 / Chart X-axis",
            ["最近6个月 / Last 6 months", "自定义 / Custom date", "全部 / Full history"],
            index=0,
        )
        plot_start_date = None
        if plot_mode == "自定义 / Custom date":
            plot_start_date = st.date_input(
                "图表起始日 / Plot from",
                value=pd.Timestamp("2026-01-01"),
            ).strftime("%Y-%m-%d")
        elif plot_mode == "全部 / Full history":
            plot_start_date = None
        else:
            plot_start_date = None  # auto 6 months in core

        currency = st.text_input(
            "价格单位 / Currency",
            value=CURRENCY_MAP.get(market, "CNY"),
        )

        uploaded = None
        if market == MARKET_CSV:
            uploaded = st.file_uploader(
                "上传 CSV / Upload CSV",
                type=["csv"],
                help="需含 Date 与 Close 列 / Requires Date & Close columns",
            )
            st.text_input("日期列名 / Date column", value="Date", key="date_col")
            st.text_input("价格列名 / Price column", value="Close", key="price_col")

        run_btn = st.button("🚀 开始分析 / Run Analysis", type="primary", use_container_width=True)

    st.markdown("---")
    st.warning(
        "⚠️ **免责声明**：本工具仅供学习研究，不构成投资建议。"
        " / **Disclaimer**: For education only, not investment advice."
    )

    if not run_btn:
        st.info("👈 在左侧配置参数后点击「开始分析」/ Configure settings and click Run.")
        _show_examples()
        return

    if market != MARKET_CSV and not ticker:
        st.error("请输入代码 / Please enter a ticker.")
        return
    if market == MARKET_CSV and uploaded is None:
        st.error("请上传 CSV 文件 / Please upload a CSV file.")
        return

    out_dir = os.path.join("output", safe_dir_name(ticker or "csv_upload"))

    config = ForecastConfig(
        asset_name=asset_name or ticker,
        ticker=ticker or "CSV",
        market=market,
        start_date=start_date,
        forecast_days=forecast_days,
        plot_start_date=plot_start_date,
        output_dir=out_dir,
        currency_label=currency,
        csv_content=uploaded.getvalue() if uploaded else None,
        date_column=st.session_state.get("date_col", "Date"),
        price_column=st.session_state.get("price_col", "Close"),
    )

    with st.spinner("运行中… / Running pipeline…"):
        try:
            result = run_forecast_pipeline(config)
        except Exception as exc:
            st.error(f"运行失败 / Failed: {exc}")
            st.exception(exc)
            return

    st.success(f"✅ 完成！/ Done!  ARIMA{result.arima_order}  AIC={result.aic:.2f}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ARIMA 阶数 / Order", str(result.arima_order))
    c2.metric("AIC", f"{result.aic:.2f}")
    c3.metric("ADF 建议 d", str(result.adf_suggested_d))
    c4.metric("数据点数 / Rows", str(len(result.prices)))

    st.subheader("📋 预测预览 / Forecast Preview")
    preview = pd.DataFrame({
        "Date": result.forecast_mean.index,
        "Forecast": result.forecast_mean.values,
        "Lower 95%": result.forecast_ci["lower"].values,
        "Upper 95%": result.forecast_ci["upper"].values,
    }).head(10)
    st.dataframe(preview, use_container_width=True)

    st.subheader("📊 图表 / Charts")
    cols = st.columns(2)
    chart_files = [
        ("01_arima_forecast.png", "ARIMA Forecast"),
        ("05_price_history_vs_forecast.png", "Price: Hist vs Forecast"),
        ("06_returns_history_vs_forecast.png", "Returns: Hist vs Forecast"),
        ("04_garch_volatility.png", "GARCH Volatility"),
        ("03_acf_pacf.png", "ACF / PACF"),
        ("02_log_returns.png", "Log Returns"),
    ]
    for i, (fname, caption) in enumerate(chart_files):
        path = os.path.join(result.output_dir, fname)
        if os.path.exists(path):
            with cols[i % 2]:
                st.image(path, caption=caption, use_container_width=True)

    st.subheader("📥 下载 / Downloads")
    dl1, dl2, dl3 = st.columns(3)
    for col, key, label in [
        (dl1, "prices", "prices.csv"),
        (dl2, "arima", "arima_forecast.csv"),
        (dl3, "garch", "garch_volatility_forecast.csv"),
    ]:
        path = result.csv_paths.get(key)
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                col.download_button(
                    label=f"⬇️ {label}",
                    data=f.read(),
                    file_name=label,
                    mime="text/csv",
                    use_container_width=True,
                )

    with st.expander("📜 运行日志 / Logs"):
        st.code("\n".join(result.logs[-80:]))


def _show_examples() -> None:
    st.subheader("💡 代码示例 / Ticker Examples")
    examples = pd.DataFrame([
        {"市场 / Market": "A股", "代码 / Ticker": "600519", "名称 / Name": "贵州茅台"},
        {"市场 / Market": "A股 ETF", "代码 / Ticker": "510300", "名称 / Name": "沪深300ETF"},
        {"市场 / Market": "美股", "代码 / Ticker": "AAPL", "名称 / Name": "Apple"},
        {"市场 / Market": "基金", "代码 / Ticker": "161725", "名称 / Name": "示例基金"},
        {"市场 / Market": "商品", "代码 / Ticker": "SI=F", "名称 / Name": "COMEX Silver"},
        {"市场 / Market": "ETF", "代码 / Ticker": "SLV", "名称 / Name": "Silver ETF"},
    ])
    st.dataframe(examples, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
