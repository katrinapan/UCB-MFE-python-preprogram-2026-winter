# HW4: 90-day return prediction and trading

This work was developed with AI assistance using the [assignment](https://github.com/yang-guo/UCB-MFE-python-preprogram-2026-winter/blob/main/Lectures/Lecture%208/hw4.md) and [course data guide](https://github.com/yang-guo/UCB-MFE-python-preprogram-2026-winter/blob/main/Lectures/modeling/README.md). The backtester was written for this project; no lecture backtester code was copied. `Student` is a placeholder directory name.

## Reproduce

Use Python 3.9+ and the full homework database, which should remain outside Git. From this folder:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -p 'test*.py' -v
.venv/bin/python run.py /path/to/equities_top2000.duckdb
```

The run writes daily values, fills and result summaries. It exits with a nonzero status if an unresolved quote gap prevents completion. The class database ends in 2015 and cannot supply the final test. Temporary DuckDB files use `/tmp/hw4_duckdb_temp`.

## Model, timing, and trading rules

At each monthly signal close, calculate adjusted-price returns over 21, 63 and 126 market sessions. Exact calendar-session prices must be positive and finite. A stock must also have an observed adjusted price 756 sessions before the signal, though intervening observations are not checked. Restrict candidates to entry-quarter universe members ranked in the top 100, with selection-date reported shares no greater than 50 billion. These are historical screens; an existing lot remains held after leaving the universe. The share threshold is a rough quality filter, not a verified correction for corporate actions or data errors.

A ridge regression predicts the gross return from the next market close to the first market close at or after 90 calendar days. Feature bounds, target clipping bounds (both 1st and 99th percentiles), and standardization are estimated only on training observations. A missing entry or exit price makes a training label unavailable; it does not determine which stocks can be selected in the backtest. Using adjusted closes avoids separately adding dividends.

The development validation fits on entries in 2010–2013 with outcomes completed before January 2, 2014, and evaluates entries in 2014–2015. The test model fits on 2010–2015 entries whose exits precede April 1, 2016; it is not retrained during the test. RMSE is computed only where the outcome is observed, so censoring may bias it.

On the first market session of each month, buy the 20 highest predicted returns, breaking ties by ticker. Budgets are fixed at the previous close: one quarter of portfolio value, capped by the cash then available, split equally. An unquoted entry is skipped. Fractional adjusted-price units represent total-return exposure. Entry and exit fees are 0.1% of traded value; the buy budget includes the fee. Positions sell at the first market close on or after 90 calendar days, and proceeds cannot fund orders already planned for that close. No borrowing or shorting; cash earns zero. At the end, cash and open positions count toward portfolio value.

For a missing daily holding price, carry the last quote at most five market sessions and flag the mark. A missing required maturity quote or a longer gap stops the run. This does not establish a tradable exit or a recovery value. A carried quote may materially misstate a position, so partial results immediately before a stop should be treated cautiously. The benchmark buys `^SP500TR` at the first evaluation close. CAGR uses actual days/365.25, max drawdown includes initial capital, and daily Sharpe uses 252 sessions and a 2% annual effective risk-free rate.

## Development results

The final specification had 4,323 completed training labels and 2,300 observed validation labels; forecast RMSE was 12.09 percentage points. The validation portfolio completed without price incidents:

| 2014-01-02 to 2015-12-31 | Strategy | S&P 500 total return |
|---|---:|---:|
| Final value | $994,654 | $1,162,750 |
| CAGR | −0.27% | 7.86% |
| Maximum drawdown | −13.67% | −12.04% |
| Daily Sharpe | −0.148 | 0.478 |

The model underperformed by $168,097 over this validation period. This does not establish that the model predicts profitable returns. The portfolio holds overlapping monthly cohorts and can have substantial equity exposure, fees, and concentrated positions.

## Exploratory test and data limitations

The requested full test cannot be reliably completed with these inputs. The current run stopped on December 20, 2024, after TBC lacked a quote for more than five sessions; its last recorded close was December 12. The last completed portfolio observation was December 19, and the TBC position was then marked at that stale December 12 quote. [The partial chart](partial_returns.png) and the following figures are exploratory and provisional; they do not extend to September 30, 2026 and do not establish the final result.

| 2016-04-01 to 2024-12-19, partial | Strategy | S&P 500 total return |
|---|---:|---:|
| Portfolio value | $1,603,484 | $3,303,287 |
| CAGR | 5.57% | 14.69% |
| Maximum drawdown | −25.55% | −33.79% |
| Daily Sharpe | 0.304 | 0.740 |

The partial strategy value was $1,699,803 below the benchmark. The complete-period benchmark alone ended at $4,402,747 on September 30, 2026, with 15.17% CAGR, −33.79% maximum drawdown, and 0.773 Sharpe. A benchmark figure alone cannot establish whether the strategy beat it over the full period.

**The final period is not an untouched test.** The first, unrestricted specification stopped on GNK in validation and RCA during the test. Later test attempts encountered VGAC, RCA, and WWACU gaps. The top-100, share-count and 756-session screens were introduced after those test-period incidents. This is hindsight-informed iteration, even though the screens themselves use data available at the time. The partial test performance is therefore not independent confirmation. Obtain verified corporate-action and delisting proceeds or a better historical security master, choose the specification in development, and evaluate it on a fresh out-of-sample period. Do not screen names using knowledge of future gaps.

The data guide warns of incomplete delisting and ticker histories, static security classifications, restated financials, and incorrect capitalization inputs; for example, it flags APDN and TSYS historical inputs. VGAC and RCA also had implausibly large recorded share counts in the iterations above. The 50-billion cutoff is arbitrary and can exclude genuine securities while missing other errors. The ranking is not a certified point-in-time or survivorship-free universe. Missing labels may introduce bias, finite adjusted prices may still be wrong, cash earns no interest, transaction costs exclude market impact, and a concentrated strategy can suffer large losses. Prediction emitted NumPy/BLAS floating-point warnings on this machine despite finite inspected predictions; independent numerical verification remains advisable.

## Accounting checks and next steps

The automated hand-calculated tests check fees, purchase and sale cash flows, a 90-day maturity, an open lot, nonnegative cash, and refusal to invent a missing maturity fill. The validation run produced no stale-mark incidents. Consider more reliable historical price and corporate-action data, compare forecasts with a simple baseline, and examine different features and allocation rules strictly within development data before a fresh test.
