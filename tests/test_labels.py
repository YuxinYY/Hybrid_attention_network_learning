import unittest

import numpy as np
import pandas as pd

from data_processing.scripts.labels import (
    compute_future_realized_vol, daily_top_fraction_labels, relabel_samples,
)


class LabelTests(unittest.TestCase):
    def test_forward_window_is_within_stock_and_excludes_anchor(self):
        dates = pd.date_range('2023-01-01', periods=7)
        stocks = pd.DataFrame({'ticker': ['A']*7 + ['B']*7,
                               'date': list(dates)*2,
                               'RET': [99, 1, 2, 3, 4, 5, 6] + [100]*7})
        out = compute_future_realized_vol(stocks.sample(frac=1, random_state=4))
        a, b = [out[out.ticker == t].reset_index(drop=True) for t in ['A', 'B']]
        self.assertAlmostEqual(a.loc[0, 'RV_5'], np.sqrt(55))
        self.assertAlmostEqual(a.loc[1, 'RV_5'], np.sqrt(90))
        self.assertAlmostEqual(b.loc[0, 'RV_5'], np.sqrt(50000))
        self.assertTrue(a.loc[2:, 'RV_5'].isna().all())
        self.assertEqual(a.loc[0, 'label_end_date'], dates[5])

    def test_missing_return_invalidates_the_forward_window(self):
        stocks = pd.DataFrame({'ticker': ['A']*6,
                               'date': pd.date_range('2023-01-01', periods=6),
                               'RET': [0, 1, 2, np.nan, 4, 5]})
        self.assertTrue(compute_future_realized_vol(stocks).RV_5.isna().all())

    def test_daily_rank_changes_with_date_and_ties_are_deterministic(self):
        vol = pd.DataFrame({'ticker': list('ABCDE')*2,
                            'date': [pd.Timestamp('2023-01-01')]*5 + [pd.Timestamp('2023-01-02')]*5,
                            'RV_5': [5, 4, 3, 2, 1, 1, 2, 3, 5, 5],
                            'label_end_date': pd.Timestamp('2023-01-09')})
        labels = daily_top_fraction_labels(vol.sample(frac=1, random_state=4))
        self.assertEqual(labels.groupby('date').target.sum().tolist(), [1, 1])
        self.assertEqual(labels.loc[labels.target == 1, 'ticker'].tolist(), ['A', 'D'])
        with self.assertRaises(ValueError):
            daily_top_fraction_labels(pd.concat([vol, vol.iloc[:1]]))

    def test_exact_four_of_twenty_and_rounding(self):
        vol = pd.DataFrame({'ticker': [f'T{i:02}' for i in range(20)],
                            'date': pd.Timestamp('2023-01-01'), 'RV_5': np.arange(20.),
                            'label_end_date': pd.Timestamp('2023-01-09')})
        labels = daily_top_fraction_labels(vol)
        self.assertEqual(labels.target.sum(), 4)
        self.assertEqual(set(labels.loc[labels.target == 1, 'ticker']), {'T16', 'T17', 'T18', 'T19'})
        self.assertEqual(daily_top_fraction_labels(vol.iloc[:6]).target.sum(), 2)

    def test_purge_uses_entire_daily_ranking_horizon(self):
        date = pd.Timestamp('2023-07-25')
        labels = daily_top_fraction_labels(pd.DataFrame({
            'ticker': ['A', 'B'], 'date': [date]*2, 'RV_5': [1., 2.],
            'label_end_date': pd.to_datetime(['2023-07-31', '2023-08-01'])}))
        samples = [('A', date, [], 1), ('B', date, [], 0), ('missing', date, [], 0)]
        self.assertEqual(relabel_samples(samples, labels, '2023-08-01'), [])
        self.assertEqual([s[3] for s in relabel_samples(samples, labels)], [0, 1])


if __name__ == '__main__':
    unittest.main()
