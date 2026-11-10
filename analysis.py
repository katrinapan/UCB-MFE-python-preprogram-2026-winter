"""Cleaning and risk calculations shared by the HW2 notebooks."""

from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd


PRICES = ["open", "high", "low", "close"]
COLUMNS = ["ts", "ticker", *PRICES, "adj_close", "volume"]
ISSUES = ["ticker", "issue_type", "issue_date", "description", "severity"]


def normalize(data):
    """Normalize labels, dates and numeric types without filling any values."""
    missing = set(COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    data = data.copy()
    data["original_ticker"] = data["ticker"].astype("string")
    data["ticker"] = (data["original_ticker"].str.replace(r"<[^>]*>", "", regex=True)
                      .str.strip().str.lstrip("$").str.upper())
    data["ts"] = pd.to_datetime(data["ts"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    for col in [*PRICES, "adj_close", "volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data.sort_values(["ticker", "ts"], kind="stable").reset_index(drop=True)


def return_outliers(data):
    """Flag both tails of per-ticker simple close returns using sample std."""
    returns = data.groupby("ticker")["close"].pct_change(fill_method=None)
    returns = returns.replace([np.inf, -np.inf], np.nan)
    grouped = returns.groupby(data["ticker"])
    mean, std = grouped.transform("mean"), grouped.transform("std")
    return (std.gt(0) & returns.notna() & (returns - mean).abs().gt(3 * std)).fillna(False)


def reference_factors(reference):
    """Use normalized, unambiguous reference rows with valid same-date prices."""
    trusted = reference.drop_duplicates(COLUMNS)
    trusted = trusted.loc[~trusted.duplicated(["ticker", "ts"], keep=False)].copy()
    valid = (np.isfinite(trusted["close"]) & trusted["close"].gt(0)
             & np.isfinite(trusted["adj_close"]) & trusted["adj_close"].gt(0))
    trusted = trusted.loc[valid, ["ticker", "ts", "close", "adj_close"]]
    trusted["factor"] = trusted["adj_close"] / trusted["close"]
    return trusted[["ticker", "ts", "factor"]]


def compare_adjustments(data, reference):
    """Show observed and independently expected adjusted prices without repairing."""
    checked = data.merge(reference_factors(reference), on=["ticker", "ts"],
                         how="left", validate="many_to_one")
    checked["expected_adj_close"] = checked["close"] * checked["factor"]
    comparable = (np.isfinite(checked["adj_close"]) & checked["adj_close"].gt(0)
                  & np.isfinite(checked["expected_adj_close"]) & checked["expected_adj_close"].gt(0))
    checked["factor_mismatch"] = comparable & ~np.isclose(
        checked["adj_close"], checked["expected_adj_close"], rtol=1e-5)
    return checked


def diagnose(data, reference=None, calendar=None):
    """Diagnose rows and gaps using reference keys or an explicit session calendar."""
    if reference is None and calendar is None:
        raise ValueError("A reference table or explicit trading calendar is required")
    findings = []

    def record(mask, kind, description, severity="high"):
        for _, row in data.loc[mask].iterrows():
            findings.append((row["ticker"], kind, row["ts"], description, severity))

    record(data["ticker"].isna() | data["ticker"].eq(""), "invalid_ticker", "Missing ticker")
    record(data["ts"].isna(), "invalid_date", "Unparseable observation date")
    if reference is not None:
        comparison = compare_adjustments(data, reference)
        mismatch = pd.Series(comparison["factor_mismatch"].to_numpy(), index=data.index)
        record(mismatch, "adjustment_mismatch", "Adjusted close disagrees with the independent same-date factor")
    if calendar is not None:
        calendar = pd.DatetimeIndex(calendar).normalize().dropna().unique()
        record(data["ts"].notna() & ~data["ts"].isin(calendar),
               "off_calendar", "Observation date is absent from the supplied session calendar")
    for col in [*PRICES, "adj_close", "volume"]:
        record(data[col].isna(), "missing_value", f"Missing {col}", "medium" if col == "volume" else "high")
        invalid = data[col].notna() & (~np.isfinite(data[col]) | data[col].lt(0))
        if col != "volume":
            invalid |= data[col].eq(0)
        record(invalid, "invalid_value", f"Nonfinite or invalid {col}")
    record(data["volume"].eq(0), "zero_volume", "Zero volume; review for stale observations", "low")
    record(data["low"].gt(data["high"]), "price_bounds", "low exceeds high")
    for col in ["open", "close"]:
        record(data[col].lt(data["low"]) | data[col].gt(data["high"]),
               "price_bounds", f"{col} is outside [low, high]")
    record(data.duplicated(COLUMNS, keep=False), "duplicate_record", "Duplicate normalized observation", "medium")
    unique_rows = data.drop_duplicates(COLUMNS)
    conflict_keys = unique_rows.loc[unique_rows.duplicated(["ticker", "ts"], keep=False), ["ticker", "ts"]]
    if not conflict_keys.empty:
        keys = pd.MultiIndex.from_frame(conflict_keys)
        record(pd.Series(pd.MultiIndex.from_frame(data[["ticker", "ts"]]).isin(keys), index=data.index),
               "conflicting_key", "Different records share ticker and date")
    record(return_outliers(data), "return_outlier", "Simple close return exceeds the per-ticker 3-sigma threshold", "medium")
    changed = data["original_ticker"].ne(data["ticker"]).fillna(False)
    record(changed, "ticker_normalized", "Original ticker retained in original_ticker", "low")
    valid = data.dropna(subset=["ticker", "ts"])
    for ticker, group in valid.groupby("ticker"):
        if reference is not None:
            expected = set(reference.loc[reference["ticker"].eq(ticker), "ts"].dropna())
        else:
            # First/last quotes bound observed active history; no listing dates are inferred.
            expected = set(calendar[(calendar >= group["ts"].min()) & (calendar <= group["ts"].max())])
        actual = set(group["ts"])
        for day in sorted(expected - actual):
            findings.append((ticker, "missing_trading_day", day,
                             "Missing supplied-calendar session within observed history", "medium"))
    return pd.DataFrame(findings, columns=ISSUES)


def clean_data(raw, reference=None, calendar=None):
    """Fill within ticker, reconstruct verified adjustments, quarantine unsafe rows."""
    data = normalize(raw)
    reference = normalize(reference) if reference is not None else None
    before = diagnose(data, reference, calendar)
    quarantine = []
    counts = {"input": len(data)}

    def remove(mask, reason):
        nonlocal data
        rejected = data.loc[mask].copy()
        if not rejected.empty:
            rejected["quarantine_reason"] = reason
            quarantine.append(rejected)
        data = data.loc[~mask].copy()

    remove(data["ticker"].isna() | data["ticker"].eq("") | data["ts"].isna(), "Invalid ticker or date")
    if calendar is not None:
        remove(~data["ts"].isin(pd.DatetimeIndex(calendar).normalize()), "Off-calendar observation")
    counts["exact_duplicates_removed"] = int(data.duplicated(COLUMNS).sum())
    data = data.drop_duplicates(COLUMNS).copy()
    remove(data.duplicated(["ticker", "ts"], keep=False), "Conflicting ticker/date records; no arbitrary winner")
    all_missing = data[PRICES].isna().all(axis=1)
    counts["all_raw_missing_removed"] = int(all_missing.sum())
    data = data.loc[~all_missing].copy()
    data = data.sort_values(["ticker", "ts"]).reset_index(drop=True)

    if reference is None:
        data["factor"] = np.nan
    else:
        data = data.merge(reference_factors(reference), on=["ticker", "ts"],
                          how="left", validate="many_to_one")
    original_close = data["close"].copy()
    for col in PRICES:
        data[f"filled_{col}"] = data[col].isna()
        data[col] = data.groupby("ticker")[col].ffill()
    data["volume_unknown"] = data["volume"].isna()
    data["volume"] = data["volume"].fillna(0)

    close_changed = original_close.isna()
    invalid_adjusted = data["adj_close"].isna() | ~np.isfinite(data["adj_close"]) | data["adj_close"].le(0)
    expected_adj = data["close"] * data["factor"]
    inconsistent = data["factor"].notna() & ~np.isclose(data["adj_close"], expected_adj, rtol=1e-5, equal_nan=False)
    needs_adjustment = close_changed | invalid_adjusted | inconsistent
    data["adjustment_rebuilt"] = needs_adjustment & data["factor"].notna()
    data.loc[data["adjustment_rebuilt"], "adj_close"] = expected_adj
    remove(needs_adjustment & data["factor"].isna(), "No verified date-specific adjustment factor")

    bad_prices = (~np.isfinite(data[[*PRICES, "adj_close"]]) | data[[*PRICES, "adj_close"]].le(0)).any(axis=1)
    bounds = (data["low"].gt(data["high"]) | data["open"].lt(data["low"])
              | data["open"].gt(data["high"]) | data["close"].lt(data["low"])
              | data["close"].gt(data["high"]))
    bad_volume = ~np.isfinite(data["volume"]) | data["volume"].lt(0)
    remove(bad_prices | bounds | bad_volume, "Unresolved prices, OHLC bounds, or invalid volume after filling")
    data["return_outlier"] = return_outliers(data)
    data["outlier_review"] = np.where(data["return_outlier"],
                                      "Unverified extreme return retained for review; no event attribution", "")
    data = data.drop(columns="factor").reset_index(drop=True)
    rejected = pd.concat(quarantine, ignore_index=True) if quarantine else pd.DataFrame(columns=[*data.columns, "quarantine_reason"])
    after = diagnose(data, reference, calendar)
    counts.update(retained=len(data), quarantined=len(rejected))
    assert counts["input"] == sum(counts[k] for k in ("exact_duplicates_removed", "all_raw_missing_removed", "retained", "quarantined"))
    validate_cleaned(data)
    return data, rejected, before, after, pd.Series(counts, name="rows")


def validate_cleaned(data):
    """Raise if the retained dataset violates the assignment's invariants."""
    prices = data[[*PRICES, "adj_close"]]
    assert np.isfinite(prices).all().all() and prices.gt(0).all().all()
    assert not data.duplicated(["ticker", "ts"]).any()
    assert data["ticker"].notna().all() and data["ticker"].ne("").all() and data["ts"].notna().all()
    assert np.isfinite(data["volume"]).all() and data["volume"].ge(0).all()
    assert data["low"].le(data["high"]).all()
    for col in ["open", "close"]:
        assert data[col].between(data["low"], data["high"]).all()


def clean_database(db_path):
    """Read raw tables, write cleaned/audit tables, and leave raw tables unchanged."""
    path = Path(db_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Place the course database at {path}; it is not supplied in this project.")
    with sqlite3.connect(path) as conn:
        raw = pd.read_sql_query("SELECT * FROM ohlc_hw", conn)
        reference = pd.read_sql_query("SELECT * FROM ohlc", conn)
    cleaned, rejected, before, after, counts = clean_data(raw, reference)
    with sqlite3.connect(path) as conn:
        for name, frame in [("ohlc_hw_cleaned", cleaned), ("ohlc_hw_quarantine", rejected),
                            ("hw2_quality_before", before), ("hw2_quality_after", after),
                            ("reference_calendar", normalize(reference)[["ticker", "ts"]].drop_duplicates())]:
            saved = frame.copy()
            for col in ("ts", "issue_date"):
                if col in saved:
                    saved[col] = pd.to_datetime(saved[col], errors="coerce").dt.strftime("%Y-%m-%d")
            saved.to_sql(name, conn, if_exists="replace", index=False)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_cleaned_ticker_date ON ohlc_hw_cleaned(ticker, ts)")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_reference_calendar ON reference_calendar(ticker, ts)")
    return raw, reference, cleaned, rejected, before, after, counts


def calculate_returns(prices):
    """Calculate daily log returns from adjusted closing prices."""
    return np.log(prices / prices.shift(1))


def calculate_volatility(returns, window=20):
    """Annualize sample rolling standard deviation, retaining warm-up gaps."""
    return returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(252)


def calculate_max_drawdown(prices):
    """Return current drawdown from each date's running peak, not running worst."""
    return prices / prices.cummax() - 1


def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Use the course's annualized log-return Sharpe approximation."""
    returns = returns.dropna()
    std = returns.std(ddof=1)
    if len(returns) < 2 or not np.isfinite(std) or std <= 1e-12:
        return float("nan")
    return (returns.mean() * 252 - risk_free_rate) / (std * np.sqrt(252))


def load_stock(db_path, ticker, start_date, end_date):
    """Load and validate a single ticker within an inclusive requested window."""
    if not ticker or pd.isna(pd.Timestamp(start_date)) or pd.isna(pd.Timestamp(end_date)):
        raise ValueError("Ticker and valid start/end dates are required")
    if pd.Timestamp(start_date) > pd.Timestamp(end_date):
        raise ValueError("start_date must not follow end_date")
    path = Path(db_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with sqlite3.connect(path) as conn:
        data = pd.read_sql_query(
            "SELECT * FROM ohlc_hw_cleaned WHERE ticker = ? AND ts >= ? AND ts <= ? ORDER BY ts",
            conn, params=(ticker, str(pd.Timestamp(start_date).date()), str(pd.Timestamp(end_date).date())))
        sessions = (pd.read_sql_query("SELECT ts FROM reference_calendar WHERE ticker = ? AND ts >= ? AND ts <= ? ORDER BY ts",
                    conn, params=(ticker, data["ts"].min(), data["ts"].max()))["ts"]
                    if not data.empty else None)
    data = normalize(data)
    validate_cleaned(data)
    if len(data) < 2:
        raise ValueError(f"{ticker}: at least two prices are required; found {len(data)}")
    if sessions is not None:
        data.attrs["session_calendar"] = pd.DatetimeIndex(pd.to_datetime(sessions))
    return data


def report_series(data):
    """Keep missing calendar sessions as NaN, so gap-spanning moves are not daily returns."""
    prices = pd.Series(data["adj_close"].to_numpy(), index=pd.DatetimeIndex(data["ts"]), name="adj_close")
    if "session_calendar" in data.attrs:
        prices = prices.reindex(data.attrs["session_calendar"])
    returns = calculate_returns(prices)
    return prices, returns, calculate_volatility(returns), calculate_max_drawdown(prices)


def stock_metrics(data):
    """Return numerical metrics as fractions, with missing estimates as NaN."""
    prices, daily_returns, vol, drawdown = report_series(data)
    returns = daily_returns.dropna()
    up, down = returns[returns.gt(0)], returns[returns.lt(0)]
    metrics = {
        "total_return": prices.iloc[-1] / prices.iloc[0] - 1,
        "annualized_return": np.expm1(returns.sum() * 252 / len(returns)) if len(returns) else np.nan,
        "annualized_volatility": returns.std(ddof=1) * np.sqrt(252),
        "sharpe_ratio": calculate_sharpe_ratio(returns),
        "max_drawdown": drawdown.min(),
        "best_day": returns.max(), "worst_day": returns.min(),
        "positive_days": returns.gt(0).mean(),
        "average_up_day": up.mean(), "average_down_day": down.mean(),
        "avg_daily_return": returns.mean(), "avg_volatility": vol.mean(),
        "actual_start": data["ts"].min().strftime("%Y-%m-%d"),
        "actual_end": data["ts"].max().strftime("%Y-%m-%d"), "observations": len(data),
        "calendar_sessions": len(prices), "missing_sessions": int(prices.isna().sum()),
        "valid_daily_returns": len(returns),
    }
    return {key: float(value) if isinstance(value, (np.floating, np.integer)) else value
            for key, value in metrics.items()}


def plot_stock(data, ticker, start_date, end_date):
    """Draw the required price, return, rolling-volatility and drawdown panels."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.ticker import PercentFormatter

    prices, returns, vol, drawdown = report_series(data)
    dates = pd.Series(prices.index, index=prices.index)
    mean_vol = vol.mean()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), layout="constrained")
    x = mdates.date2num(dates)
    points = np.column_stack([x, prices])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    colors = ["#969ca4" if pd.isna(v) else "#c74b4b" if v > mean_vol else "#23856b" for v in vol]
    ax = axes[0, 0]
    ax.add_collection(LineCollection(segments, colors=colors[1:], linewidths=1.7))
    ax.scatter(dates, prices, c=colors, s=7)
    ax.plot(dates, prices.rolling(50, min_periods=50).mean(), color="#244e83", label="50-day MA")
    ax.autoscale_view()
    ax.set_title(f"{ticker} Price and 50-day MA")
    ax.set_ylabel("Adjusted close")
    ax.legend(loc="best")
    if not prices.rolling(50, min_periods=50).mean().notna().any():
        ax.text(.03, .93, "No complete 50-session window: MA unavailable", transform=ax.transAxes, fontsize=9)
    ax = axes[0, 1]
    ax.bar(dates, returns, color=np.where(returns.ge(0), "#23856b", "#c74b4b"), width=1.2)
    ax.axhline(0, color="#555", linewidth=.8)
    ax.set_title(f"{ticker} Daily Returns")
    ax.set_ylabel("Daily log return")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax = axes[1, 0]
    ax.plot(dates, vol, color="#244e83", linewidth=1.5)
    if pd.notna(mean_vol):
        ax.axhline(mean_vol, color="#555", linestyle="--", label="Mean rolling volatility")
        ax.fill_between(dates, vol, mean_vol, where=vol.gt(mean_vol), color="#c74b4b", alpha=.25)
        ax.fill_between(dates, vol, mean_vol, where=vol.lt(mean_vol), color="#23856b", alpha=.20)
        ax.legend(loc="best")
    else:
        ax.text(.03, .93, "Need 20 consecutive valid daily returns", transform=ax.transAxes)
    ax.set_title(f"{ticker} Rolling Volatility (20-day)")
    ax.set_ylabel("Annualized volatility")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax = axes[1, 1]
    for fraction in (.25, .5, .75, 1):
        ax.fill_between(dates, drawdown * fraction, 0, color="#c74b4b", alpha=.15)
    ax.plot(dates, drawdown, color="#ad344e", linewidth=1.3)
    worst = drawdown.idxmin()
    ax.scatter([dates.loc[worst]], [drawdown.loc[worst]], color="#78253b", zorder=3)
    align_right = dates.loc[worst] > dates.iloc[0] + (dates.iloc[-1] - dates.iloc[0]) * .75
    ax.annotate(f"{drawdown.min():.2%}", (dates.loc[worst], drawdown.loc[worst]),
                xytext=(-8 if align_right else 8, 12), textcoords="offset points",
                ha="right" if align_right else "left")
    ax.set_title(f"{ticker} Maximum Drawdown")
    ax.set_ylabel("Drawdown from prior peak")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes.flat:
        ax.set_xlim(dates.iloc[0], dates.iloc[-1])
        ax.grid(alpha=.18)
        ax.set_xlabel("Observation date")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.suptitle(f"{ticker} Risk Analysis Report - {start_date} to {end_date}", fontsize=18)
    return fig
