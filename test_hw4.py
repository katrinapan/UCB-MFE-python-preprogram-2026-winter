import unittest
import duckdb
import numpy as np
import pandas as pd
from run import START, COST, backtest


class BacktestChecks(unittest.TestCase):
    def test_purchase_sale_and_final_open_lot(self):
        dates = pd.to_datetime(['2020-01-02', '2020-01-03', '2020-04-01'])
        con = duckdb.connect()
        con.execute('create table prices(date date, symbol varchar, adj_close double)')
        con.executemany('insert into prices values (?, ?, ?)',
                        [(d.to_pydatetime(), 'A', p) for d, p in zip(dates, [100, 105, 110])] +
                        [(d.to_pydatetime(), 'B', p) for d, p in zip(dates, [50, 50, 60])])
        samples = pd.DataFrame({'entry': [dates[0], dates[1]], 'symbol': ['A', 'B'], 'score': [1., 1.]})
        benchmark = pd.Series([100., 101., 102.], index=dates)
        hist, fills, lots, incidents = backtest(con, pd.Series(dates), samples, benchmark, dates[0], dates[-1])
        a_units = 250_000 / (100 * (1+COST))
        # The second budget is based on the previous close's cash and portfolio value.
        b_budget = hist.iloc[0].strategy / 4
        b_units = b_budget / (50 * (1+COST))
        expected = START - 250_000 - b_budget + a_units*110*(1-COST) + b_units*60
        self.assertAlmostEqual(hist.iloc[-1].strategy, expected, places=5)
        self.assertEqual(list(fills.side), ['BUY', 'BUY', 'SELL'])
        self.assertEqual(fills.iloc[2].date, dates[2])
        self.assertEqual(len(lots), 1)
        self.assertTrue((hist.cash >= 0).all())
        self.assertEqual(incidents, [])

    def test_missing_maturity_does_not_fabricate_sale(self):
        dates = pd.to_datetime(['2020-01-02', '2020-04-01'])
        con = duckdb.connect()
        con.execute('create table prices(date date, symbol varchar, adj_close double)')
        con.execute("insert into prices values ('2020-01-02','A',100)")
        samples = pd.DataFrame({'entry':[dates[0]], 'symbol':['A'], 'score':[1.]})
        with self.assertRaisesRegex(RuntimeError, 'Missing maturity quote'):
            backtest(con, pd.Series(dates), samples, pd.Series([100., 100.], index=dates), dates[0], dates[-1])

if __name__ == '__main__': unittest.main()
