# 金融时间序列预测工具 / Financial Time Series Forecast

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web_UI-red)](https://streamlit.io/)

基于 **ADF → auto_arima → GARCH** 的入门级金融预测工具，支持股票、基金、ETF、商品等价格序列的分析与短期预测。

An educational tool for **ADF → auto_arima → GARCH** forecasting on stocks, funds, ETFs, and commodities.

> ⚠️ **免责声明 / Disclaimer**  
> 本项目仅供学习与研究，**不构成任何投资建议**。金融市场有风险，请谨慎决策。  
> For **education and research only**. Not investment advice. Use at your own risk.

---

## 功能特点 / Features

| 中文 | English |
|------|---------|
| 🌐 Streamlit 网页界面，无需改代码 | 🌐 Streamlit web UI — no code editing |
| 📈 支持 A 股、美股、场外基金、商品 | 📈 CN/US stocks, funds, commodities |
| 🤖 `auto_arima` 自动选择 ARIMA(p,d,q) | 🤖 Automatic order selection via `auto_arima` |
| 📊 6 张图表 + 3 个 CSV 输出 | 📊 6 charts + 3 CSV exports |
| 🇨🇳 国内优先 AKShare 数据源 | 🇨🇳 AKShare-first for users in China |

---

## 环境要求 / Requirements

- Python **3.10+**
- Windows / macOS / Linux（含 WSL）

---

## 快速开始 / Quick Start

### 1. 克隆仓库 / Clone

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### 2. 安装依赖 / Install dependencies

```bash
pip install -r requirements.txt
```

### 3. 启动网页 / Launch web app

```bash
streamlit run app.py
```

浏览器访问 / Open: **http://localhost:8501**

### 4. 命令行（银价示例）/ CLI (silver example)

```bash
python silver_price_arima_garch.py
```

---

## 使用说明 / Usage

### 网页操作 / Web UI Steps

1. 左侧选择 **市场类型 / Market**（A 股、美股、基金、商品、CSV）
2. 输入 **代码 / Ticker** 和 **名称 / Asset Name**
3. 设置历史起始日、预测天数、图表时间轴
4. 点击 **「开始分析 / Run Analysis」**
5. 查看图表，或下载 CSV

### 代码示例 / Ticker Examples

| 市场 Market | 代码 Ticker | 名称 Name |
|-------------|-------------|-----------|
| A 股 CN Stock | `600519` | 贵州茅台 |
| A 股 ETF | `510300` | 沪深300ETF |
| 美股 US Stock | `AAPL` | Apple |
| 场外基金 CN Fund | `161725` | 示例基金 |
| 商品 Commodity | `SI=F` | COMEX 白银 |
| 白银 ETF | `SLV` | iShares Silver |

### 本地 CSV / Upload CSV

CSV 至少包含两列 / Minimum columns:

```csv
Date,Close
2020-01-02,100.5
2020-01-03,101.2
```

在界面选择「本地 CSV」，上传文件即可。  
Select **Upload CSV** market and upload your file.

---

## 项目结构 / Project Structure

```
Project/
├── app.py                      # Streamlit 网页入口 / Web entry
├── forecast_core.py            # 通用建模核心 / Core pipeline
├── silver_price_arima_garch.py # 命令行银价示例 / CLI silver demo
├── requirements.txt
├── README.md
├── .gitignore
└── output/                     # 运行结果（勿提交 Git）/ Generated output
    └── <ticker>/
        ├── prices.csv
        ├── arima_forecast.csv
        ├── garch_volatility_forecast.csv
        └── *.png
```

---

## 建模流程 / Pipeline

```
数据下载 → ADF 检验 → auto_arima 选阶 → ARIMA 预测 → GARCH 波动率 → 图表 & CSV
Download → ADF test → auto_arima → ARIMA forecast → GARCH → Charts & CSV
```

| 步骤 Step | 说明 Description |
|-----------|------------------|
| **ADF** | 检验序列是否平稳 / Test stationarity |
| **auto_arima** | 自动搜索最优 (p, d, q) / Auto-select ARIMA order |
| **GARCH** | 对 ARIMA 残差建模波动率 / Model volatility on residuals |

---

## 输出文件 / Output Files

| 文件 File | 内容 Content |
|-----------|--------------|
| `prices.csv` | 历史价格 / Historical prices |
| `arima_forecast.csv` | 价格预测 + 95% 区间 + 预测收益率 / Forecast + CI + returns |
| `garch_volatility_forecast.csv` | GARCH 条件波动率 / Conditional volatility |
| `01_arima_forecast.png` | 历史 + 拟合 + 预测总览 / Overview chart |
| `05_price_history_vs_forecast.png` | 橙=历史，红=预测 / Historical vs forecast price |
| `06_returns_history_vs_forecast.png` | 收益率对比 / Returns comparison |
| `03_acf_pacf.png` | 自相关图 / ACF & PACF |
| `04_garch_volatility.png` | 波动率预测 / Volatility forecast |

---

## 常见问题 / FAQ

### 雅虎限流 / Yahoo rate limit?

国内建议用 A 股/基金（AKShare）；商品可试 AKShare 或上传 CSV。  
Use AKShare markets in China, or upload CSV for commodities.

### 图表中文乱码 / Chart garbled text?

WSL/Linux 下图表使用**英文标签**，避免字体问题。  
Charts use **English labels** on Linux/WSL.

### 预测线很平 / Flat forecast?

ARIMA 对剧烈行情的外推可能较保守，属模型特性。  
ARIMA may produce flat short-term forecasts in volatile regimes.

### CSV 保存失败 / CSV permission error?

关闭正在打开的 Excel 文件后重试。  
Close Excel if the CSV file is open.

---

## 技术栈 / Tech Stack

- [pandas](https://pandas.pydata.org/) · [numpy](https://numpy.org/)
- [statsmodels](https://www.statsmodels.org/) · [pmdarima](https://alkaline-ml.com/pmdarima/)
- [arch](https://arch.readthedocs.io/) (GARCH)
- [akshare](https://akshare.akfamily.xyz/) · [yfinance](https://github.com/ranaroussi/yfinance)
- [Streamlit](https://streamlit.io/) · [matplotlib](https://matplotlib.org/)

---

## 上传到 GitHub / Publish to GitHub

```bash
git init
git add app.py forecast_core.py silver_price_arima_garch.py requirements.txt README.md .gitignore
git commit -m "feat: ARIMA+GARCH forecast with Streamlit UI"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

建议为仓库添加 Topics：`python` `arima` `garch` `streamlit` `finance` `time-series`

---

## 许可证 / License

MIT License

---

## 作者 / Author

Your Name — [GitHub](https://github.com/YOUR_USERNAME)
