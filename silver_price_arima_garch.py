# -*- coding: utf-8 -*-
"""
银价预测脚本（命令行版）

流程：ADF 检验 → auto_arima → GARCH
与 Streamlit 网页版共用 forecast_core 模块。

运行: python silver_price_arima_garch.py
网页: streamlit run app.py
"""

from forecast_core import run_forecast_pipeline, silver_default_config


def main() -> None:
    config = silver_default_config()
    result = run_forecast_pipeline(config)
    print(f"\nARIMA{result.arima_order}  |  Output: {result.output_dir}")


if __name__ == "__main__":
    main()
