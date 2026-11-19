"""Build the point-in-time feature and target rows from the HW4 course database."""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

FEATURES = ("r21", "r63", "r126")
TEST_START = "2016-04-01"
TEST_END = "2026-09-30"


def load_samples(database: Path) -> pd.DataFrame:
    with duckdb.connect(str(database), read_only=True, config={"threads": 4}) as connection:
        calendar = connection.sql("SELECT date FROM calendar ORDER BY date").df()["date"]
        months = calendar.dt.to_period("M")
        entries = np.flatnonzero(months.ne(months.shift()).to_numpy())
        rows = []
        for i in entries:
            if i < 757 or calendar[i] < pd.Timestamp("2010-01-01"):
                continue
            exit_index = int(calendar.searchsorted(calendar[i] + pd.Timedelta(days=90)))
            exit_date = calendar[exit_index] if exit_index < len(calendar) else pd.NaT
            rows.append(
                (
                    calendar[i],
                    calendar[i - 1],
                    calendar[i - 22],
                    calendar[i - 64],
                    calendar[i - 127],
                    calendar[i - 757],
                    exit_date,
                )
            )
        dates = pd.DataFrame(
            rows, columns=["entry", "signal", "d21", "d63", "d126", "d756", "exit"]
        )
        dates["quarter_start"] = dates["entry"].dt.to_period("Q").dt.start_time
        membership = connection.sql(
            """
            SELECT quarter_start, symbol
            FROM universe_membership
            WHERE universe_rank <= 100 AND selection_shares <= 50000000000
            """
        ).df()
        samples = dates.merge(membership, on="quarter_start")
        for field in ("signal", "d21", "d63", "d126", "d756", "entry", "exit"):
            wanted = pd.DataFrame({"date": dates[field].dropna().unique()})
            connection.register("wanted_dates", wanted)
            prices = connection.sql(
                """
                SELECT p.date, p.symbol, p.adj_close
                FROM prices p
                JOIN wanted_dates USING (date)
                WHERE isfinite(p.adj_close) AND p.adj_close > 0
                """
            ).df()
            prices = prices.rename(columns={"date": field, "adj_close": f"price_{field}"})
            samples = samples.merge(
                prices, on=[field, "symbol"], how="left", validate="many_to_one"
            )

    for window in (21, 63, 126):
        samples[f"r{window}"] = samples["price_signal"] / samples[f"price_d{window}"] - 1
    columns = list(FEATURES)
    samples[columns] = samples[columns].replace([np.inf, -np.inf], np.nan)
    samples = samples.dropna(subset=[*FEATURES, "price_d756"]).copy()
    samples["target"] = samples["price_exit"] / samples["price_entry"] - 1
    return samples
