"""HW4: monthly, next-close signals and an independent lot-level backtest."""
import argparse
import sys
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

START = 1_000_000.0
COST = 0.001
FEATURES = ['r21', 'r63', 'r126']
OUT = Path(__file__).resolve().parent


def load(db):
    con = duckdb.connect(str(db), read_only=True, config={'temp_directory': '/tmp/hw4_duckdb_temp'})
    cal = con.sql("select date from calendar order by date").df().date
    # First session of each month; all signals are from the previous session.
    idx = np.flatnonzero(cal.dt.to_period('M').ne(cal.dt.to_period('M').shift()).to_numpy())
    rows = []
    for i in idx:
        if i < 757 or cal[i] < pd.Timestamp('2010-01-01'):
            continue
        maturity = cal[i] + pd.Timedelta(days=90)
        j = int(cal.searchsorted(maturity))
        rows.append((cal[i], cal[i-1], cal[i-22], cal[i-64], cal[i-127], cal[i-757], cal[j] if j < len(cal) else pd.NaT))
    dates = pd.DataFrame(rows, columns=['entry','signal','d21','d63','d126','d756','exit'])
    con.register('dates', dates)
    con.execute('SET threads=4')
    membership = con.sql("select quarter_start,symbol from universe_membership where universe_rank <= 100 and selection_shares <= 50000000000").df()
    dates['quarter_start'] = dates.entry.dt.to_period('Q').dt.start_time
    samples = dates.merge(membership, on='quarter_start', how='inner')
    # Restrict each scan to one set of monthly dates before joining by symbol.
    for field in ['signal', 'd21', 'd63', 'd126', 'd756', 'entry', 'exit']:
        wanted = pd.DataFrame({'date': dates[field].dropna().unique()})
        con.register('wanted_dates', wanted)
        q = con.sql("""select p.date,p.symbol,p.adj_close from prices p
            join wanted_dates d using(date) where isfinite(p.adj_close) and p.adj_close>0""").df()
        q = q.rename(columns={'date':field, 'adj_close':'price_'+field})
        samples = samples.merge(q, on=[field, 'symbol'], how='left', validate='many_to_one')
    for horizon in [21, 63, 126]:
        samples['r'+str(horizon)] = samples.price_signal / samples['price_d'+str(horizon)] - 1
    samples[FEATURES] = samples[FEATURES].replace([np.inf, -np.inf], np.nan)
    samples = samples.dropna(subset=FEATURES + ['price_d756']).copy()
    samples['entry_price'] = samples.price_entry
    samples['exit_price'] = samples.price_exit
    samples['target'] = samples.exit_price / samples.entry_price - 1
    valid = (np.isfinite(samples.entry_price) & (samples.entry_price > 0) &
             np.isfinite(samples.exit_price) & (samples.exit_price > 0))
    samples.loc[~valid, 'target'] = np.nan
    benchmark = con.sql("select date, adj_close from benchmark where symbol='^SP500TR' order by date").df().set_index('date').adj_close
    return con, cal, samples, benchmark


def fit(samples, cutoff):
    train = samples[(samples.entry >= '2010-01-01') & (samples.entry <= '2015-12-31') &
                    (samples.exit < cutoff) & samples.target.notna()].copy()
    # Cap extreme targets to limit the influence of bad prices, learned from training only.
    lo, hi = train.target.quantile([.01, .99])
    model = make_pipeline(StandardScaler(), Ridge(alpha=1000))
    bounds = train[FEATURES].quantile([.01, .99])
    model.fit(train[FEATURES].clip(bounds.loc[.01], bounds.loc[.99], axis=1), train.target.clip(lo, hi))
    return model, bounds, len(train)


class GapError(RuntimeError):
    def __init__(self, message, history, fills, lots, incidents):
        super().__init__(message)
        self.history, self.fills, self.lots, self.incidents = history, fills, lots, incidents


def backtest(con, cal, samples, benchmark, start, end):
    sessions = pd.DatetimeIndex(cal[(cal >= start) & (cal <= end)])
    candidates = samples[(samples.entry >= start) & (samples.entry <= end)].copy()
    symbols = candidates.symbol.unique().tolist()
    con.register('selected_symbols', pd.DataFrame({'symbol':symbols}))
    quotes = con.execute('''select p.date,p.symbol,p.adj_close from prices p
        join selected_symbols s using(symbol) where p.date between ? and ?
        and isfinite(p.adj_close) and p.adj_close>0''', [start,end]).df()
    prices = quotes.pivot(index='date', columns='symbol', values='adj_close').reindex(sessions)
    orders = {date: frame.sort_values(['score','symbol'], ascending=[False,True]).head(20)
              for date, frame in candidates.groupby('entry')}
    cash, lots, history, fills, incidents = START, [], [], [], []
    for day in sessions:
        row = prices.loc[day]
        # Sell first, only at an observed market close. Missing maturity quotes are unresolved.
        for lot in lots[:]:
            p = row.get(lot['symbol'], np.nan)
            if day >= lot['maturity']:
                if not np.isfinite(p):
                    incidents.append(f"Missing maturity quote: {lot['symbol']} {day.date()}")
                    raise GapError(incidents[-1], history, fills, lots, incidents)
                proceeds = lot['units'] * p * (1-COST)
                cash += proceeds
                fills.append((day, 'SELL', lot['symbol'], p, proceeds))
                lots.remove(lot)
        # Plan at prior close: fixed 1/4 of preceding close equity, split equally.
        if day in orders:
            budget = min(history[-1][2] if history else START, (history[-1][1] if history else START) / 4)
            chosen = orders[day]
            per_stock = budget / len(chosen) if len(chosen) else 0
            for symbol in chosen.symbol:
                p = row.get(symbol, np.nan)
                if not np.isfinite(p):
                    incidents.append(f"Skipped missing entry: {symbol} {day.date()}")
                    continue
                amount = min(per_stock, cash)
                if amount <= 0: continue
                units = amount / (p * (1+COST))
                cash -= amount
                lots.append(dict(symbol=symbol, units=units, maturity=day+pd.Timedelta(days=90), last=p, stale=0))
                fills.append((day, 'BUY', symbol, p, amount))
        value = cash
        for lot in lots:
            p = row.get(lot['symbol'], np.nan)
            if np.isfinite(p): lot['last'], lot['stale'] = p, 0
            else:
                lot['stale'] += 1
                if lot['stale'] > 5:
                    raise GapError(f"Unresolved stale holding: {lot['symbol']} {day.date()}", history, fills, lots, incidents)
                incidents.append(f"Stale mark: {lot['symbol']} {day.date()}")
            value += lot['units'] * lot['last']
        if cash < -1e-6: raise AssertionError('Negative cash')
        history.append((day,value,cash,len(lots)))
    hist = pd.DataFrame(history, columns=['date','strategy','cash','lots']).set_index('date')
    b = benchmark.reindex(sessions)
    if b.isna().any(): raise RuntimeError('Missing benchmark sessions')
    hist['benchmark'] = START * b / b.iloc[0]
    return hist, pd.DataFrame(fills, columns=['date','side','symbol','price','cash_flow']), lots, incidents


def metrics(series):
    # Include initial capital immediately before first close so entry fees count.
    returns = series.pct_change(); returns.iloc[0] = series.iloc[0] / START - 1
    wealth = pd.concat([pd.Series([START]), series.reset_index(drop=True)], ignore_index=True)
    years = (series.index[-1]-series.index[0]).days / 365.25
    return {'final':series.iloc[-1], 'cagr':(series.iloc[-1]/START)**(1/years)-1,
            'max_drawdown':(wealth/wealth.cummax()-1).min(),
            'sharpe':(returns.mean() - (1.02**(1/252)-1))/returns.std(ddof=1)*np.sqrt(252)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('database', type=Path); args=parser.parse_args()
    con, cal, samples, benchmark = load(args.database)
    print('samples',len(samples), flush=True)
    failed = False
    # The large-cap screen was added after an earlier exploratory test encountered a data gap.
    for label, start, end, cutoff in [('validation','2014-01-02','2015-12-31','2014-01-02'),
                                       ('test','2016-04-01','2026-09-30','2016-04-01')]:
        model, bounds, count = fit(samples, pd.Timestamp(cutoff))
        subset = samples[(samples.entry >= start) & (samples.entry <= end)].copy()
        subset['score'] = model.predict(subset[FEATURES].clip(bounds.loc[.01], bounds.loc[.99], axis=1))
        # OOS prediction error is descriptive only; test labels never enter the fit.
        if not np.isfinite(subset.score).all():
            raise RuntimeError('Non-finite predictions')
        valid = subset.target.notna()
        mse = np.mean((subset.loc[valid,'score']-subset.loc[valid,'target'])**2)
        try:
            hist, fills, lots, incidents = backtest(con, cal, subset, benchmark, start, end)
        except GapError as exc:
            failed = True
            (OUT / f'{label}_failure.txt').write_text(str(exc) + '\n')
            print(label, 'STOPPED:', exc, flush=True)
            if exc.history:
                partial = pd.DataFrame(exc.history, columns=['date','strategy','cash','lots']).set_index('date')
                partial['benchmark'] = START * benchmark.reindex(partial.index) / benchmark.loc[partial.index[0]]
                partial.to_csv(OUT / f'{label}_partial_daily.csv')
                print(label, 'PARTIAL THROUGH', partial.index[-1].date(),
                      {'strategy':metrics(partial.strategy), 'benchmark':metrics(partial.benchmark)}, flush=True)
                if label == 'test':
                    ax=(partial[['strategy','benchmark']]/START-1).plot(figsize=(10,5), title='Exploratory partial backtest: stopped before unresolved quote')
                    ax.set_ylabel('Cumulative return'); ax.grid(alpha=.25)
                    plt.tight_layout(); plt.savefig(OUT/'partial_returns.png',dpi=160); plt.close()
            if label == 'test':
                b = benchmark.loc[start:end]
                b = START * b / b.iloc[0]
                (OUT / 'benchmark_results.txt').write_text(repr(metrics(b)) + '\n')
            continue
        hist.to_csv(OUT / f'{label}_daily.csv'); fills.to_csv(OUT / f'{label}_fills.csv',index=False)
        result = {'training_rows':count,'labeled_evaluation_rows':int(valid.sum()), 'prediction_rmse':float(np.sqrt(mse)),
                  'strategy':metrics(hist.strategy), 'benchmark':metrics(hist.benchmark),
                  'open_lots':len(lots),'incidents':incidents}
        (OUT / f'{label}_results.txt').write_text(repr(result)+'\n')
        print(label,result,flush=True)
        if label == 'test':
            ax=(hist[['strategy','benchmark']]/START-1).plot(figsize=(10,5), title='Cumulative return: strategy vs S&P 500 total return')
            ax.set_ylabel('Cumulative return'); ax.grid(alpha=.25); plt.tight_layout(); plt.savefig(OUT/'cumulative_returns.png',dpi=160)
    if failed:
        sys.exit(1)

if __name__ == '__main__': main()
