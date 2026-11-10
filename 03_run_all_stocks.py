"""Run the HW2 stock reports and compare the successful results."""

import argparse
from datetime import datetime
from pathlib import Path
import sqlite3
from urllib.parse import quote
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import nbformat
import pandas as pd
import papermill as pm
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
SUMMARY_COLUMNS = ["ticker", "avg_daily_return", "avg_volatility", "max_drawdown",
                   "sharpe_ratio", "total_return", "report_path", "actual_start",
                   "actual_end", "observations", "calendar_sessions", "missing_sessions", "valid_daily_returns"]


def reporting_window(latest):
    month_start = pd.Timestamp(latest).normalize().replace(day=1)
    return ((month_start - pd.DateOffset(months=6)).strftime("%Y-%m-%d"),
            (month_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))


def read_report_metrics(report):
    notebook = nbformat.read(report, as_version=4)
    matches = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.outputs:
            values = output.get("data", {}).get("application/json", {})
            if "report_metrics" in values:
                matches.append(values["report_metrics"])
    if len(matches) != 1:
        raise ValueError("Expected one named report_metrics JSON output")
    return matches[0]


def plot_portfolio(summary, destination):
    fig, ax = plt.subplots(figsize=(11, 7), layout="constrained")
    valid = summary.loc[np.isfinite(summary["avg_volatility"]) & np.isfinite(summary["total_return"])]
    known = valid.loc[np.isfinite(valid["sharpe_ratio"])]
    unknown = valid.loc[~np.isfinite(valid["sharpe_ratio"])]
    if not known.empty:
        dots = ax.scatter(known["avg_volatility"], known["total_return"],
                          c=known["sharpe_ratio"], cmap="RdYlGn", s=65, edgecolors="white")
        fig.colorbar(dots, ax=ax, label="Annualized Sharpe ratio")
    if not unknown.empty:
        ax.scatter(unknown["avg_volatility"], unknown["total_return"], color="gray", label="Undefined Sharpe")
        ax.legend()
    for _, row in valid.nlargest(5, "total_return").iterrows():
        ax.annotate(row["ticker"], (row["avg_volatility"], row["total_return"]),
                    xytext=(6, 6), textcoords="offset points")
    if valid.empty:
        ax.text(.5, .5, "No finite rolling-volatility estimates", ha="center", transform=ax.transAxes)
    ax.set(title="Risk-Return Profile of Portfolio", xlabel="Mean annualized 20-day rolling volatility",
           ylabel="Total adjusted-close return")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=.2)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def run_batch(db_path, output_root, kernel_name="python3"):
    db_path = Path(db_path).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"Course database missing: {db_path}. Run the cleaning notebook first.")
    with sqlite3.connect(db_path) as conn:
        tickers = pd.read_sql_query(
            "SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker", conn
        )["ticker"].tolist()
        latest = conn.execute("SELECT MAX(ts) FROM ohlc_hw_cleaned").fetchone()[0]
    if not tickers or latest is None:
        raise ValueError("ohlc_hw_cleaned has no observations")
    start, end = reporting_window(latest)
    destination = Path(output_root).resolve() / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")
    destination.mkdir(parents=True, exist_ok=True)
    records, failures = [], []
    print(f"Requested window for every ticker: {start} to {end}; database latest: {latest}")
    for ticker in tqdm(tickers, desc="Generating reports"):
        report = destination / f"{quote(str(ticker), safe='')}_report.ipynb"
        try:
            pm.execute_notebook(
                str(ROOT / "02_stock_report_template.ipynb"), str(report),
                parameters={"ticker": ticker, "start_date": start,
                            "end_date": end, "db_path": str(db_path)},
                kernel_name=kernel_name, cwd=str(ROOT),
                progress_bar=False, log_output=False,
            )
            metrics = read_report_metrics(report)
            record = {"ticker": ticker, "report_path": report.name}
            for column in SUMMARY_COLUMNS:
                if column not in record:
                    record[column] = metrics[column]
            records.append(record)
        except Exception as exc:
            # One failed notebook should not discard the other stocks' reports.
            failures.append(f"{ticker}\t{type(exc).__name__}: {str(exc).replace(chr(10), ' ')}")
            print(f"Failed {ticker}: {type(exc).__name__}: {exc}")
    (destination / "failed_stocks.txt").write_text("\n".join(failures) + ("\n" if failures else ""), encoding="utf-8")
    summary = pd.DataFrame(records, columns=SUMMARY_COLUMNS)
    text_columns = {"ticker", "report_path", "actual_start", "actual_end"}
    for column in summary.columns:
        if column not in text_columns:
            summary[column] = pd.to_numeric(summary[column], errors="raise")
    summary = summary.sort_values("sharpe_ratio", ascending=False, na_position="last")
    summary.to_csv(destination / "portfolio_summary.csv", index=False)
    print(f"Successfully generated {len(records)}/{len(tickers)} reports")
    if summary.empty:
        stale_plot = destination / "risk_return_scatter.png"
        if stale_plot.exists():
            stale_plot.unlink()
        print("All reports failed. The CSV contains headers only; no portfolio comparison was generated.")
        return summary
    ranked = summary.loc[np.isfinite(summary["sharpe_ratio"])]
    print("Top 10 by Sharpe (undefined ratios excluded):\n", ranked.head(10).to_string(index=False))
    print("Bottom 10 by Sharpe (may overlap the top 10):\n", ranked.tail(10).to_string(index=False))
    print("Compare actual_start/actual_end: shorter histories have different realized coverage.")
    plot_portfolio(summary, destination / "risk_return_scatter.png")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=ROOT / "data/data.db")
    parser.add_argument("--output-root", type=Path, default=ROOT / "reports")
    parser.add_argument("--kernel-name", default="python3")
    args = parser.parse_args()
    run_batch(args.db_path, args.output_root, args.kernel_name)
