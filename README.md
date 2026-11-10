This folder contains the HW2 submission: a cleaning/validation notebook, a
parameterized stock-risk report, a Papermill batch runner, and shared functions in
`analysis.py`.

## Local setup and run order

Use Python 3.12 or later. Download the course SQLite database, keep the original
elsewhere, and place a working copy at `data/data.db`. It must contain `ohlc_hw`,
`ohlc`, and `market_caps`. Do not commit the database or virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name hw2 --display-name "Python (HW2)"
```

Run `01_data_cleaning.ipynb` from top to bottom with the HW2 kernel. It diagnoses,
cleans, validates, and saves `ohlc_hw_cleaned`, quarantine and quality tables, plus
audit files under `reports/YYYY-MM-DD/`. The source tables are not modified.

Run `02_stock_report_template.ipynb` directly or execute all stocks with:

```bash
python 03_run_all_stocks.py --db-path data/data.db --kernel-name hw2
```

The batch writes successful notebooks, `portfolio_summary.csv`,
`risk_return_scatter.png`, and `failed_stocks.txt`. No API key or network data is
required.

## Data decisions

Ticker labels are stripped of HTML, whitespace, a leading `$`, and case differences;
the original label is retained. Exact duplicates are removed, while conflicting
`(ticker, ts)` rows are quarantined. Missing sessions are checked against each
ticker's dates in the supplied `ohlc` table and are reported rather than fabricated.

Missing raw prices may be forward-filled only from an earlier row of the same ticker,
then all price and OHLC bounds are rechecked. Missing volume becomes zero only with
an audit flag. Adjusted close is rebuilt only from a valid same-ticker, same-date
factor from `ohlc`; factors are never carried across dates. Unresolved rows are
quarantined. Extreme returns are retained as review flags rather than automatically
deleted. Synthetic checks cover missing dates, invalid values, duplicates, conflicts,
leading gaps, missing factors, and both return tails.

The final data must have unique ticker/date keys, positive finite prices, nonnegative
finite volume, valid OHLC bounds, and ISO dates. The notebook also verifies its saved
output and confirms that `ohlc_hw` and `ohlc` remain unchanged.

## Report interpretation

Reports use adjusted close and daily log returns. Annualization uses 252 trading
days; rolling volatility uses 20 returns; the moving average uses 50 prices. Sharpe
uses a 2% annual risk-free rate. Total return uses the first and last observed prices,
and maximum drawdown is the worst decline from an observed running peak.

Missing sessions remain `NaN`, so no zero return or invented price is introduced.
Each report shows requested and actual coverage. Tickers with shorter histories are
not extrapolated and should be compared with that limitation in mind. CSV metrics are
stored as fractions, with percentage formatting used only for display.

Each ticker is an independent Papermill job. Failures are logged and do not stop the
batch. Only successful current-run notebooks enter the portfolio summary; if all
reports fail, the CSV contains headers only and no scatter plot is produced.

## Current run

The supplied database contained 397,711 rows, 150 original labels, and 30 normalized
tickers covering 1962-01-02 through 2024-12-06. Cleaning removed 29,460 exact
duplicates, rebuilt 28,791 adjusted closes, retained 368,246 valid rows, and
quarantined five invalid rows. All retained adjusted closes match their verified
same-date reference factors.

The batch completed on 2026-11-10 with 30/30 reports succeeding for the common
2024-06-01 to 2024-11-30 request window. AAPL ends on 2024-10-18, so its sample uses
97 observations and reports a 21.25% total return, 1.909 Sharpe ratio, and -11.75%
maximum drawdown without extrapolation.

## Sample outputs

`reports/2026-11-10/` includes executed AAPL, IBM, and NVDA reports, the 30-stock
summary, scatter plot, empty failure log, audit outputs, and `run_manifest.json`.
The database and remaining generated reports are excluded.

Original assignment:
<https://github.com/yang-guo/UCB-MFE-python-preprogram-2026-winter/blob/main/Lectures/Lecture%202/hw2.md>
