"""Run with: python -m unittest discover -s scripts -p test_features.py"""
import unittest
import numpy as np
import pandas as pd
from extract_features import extract_recording, METADATA
from gpvs_data import EXPECTED


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({c: np.full(3000, 2.0) for c in EXPECTED})
        self.frame['Time'] = 0.2 + np.arange(3000) / 10000
        self.settings = dict(window_samples=1000, stride_samples=200, startup_exclusion_s=0)

    def test_known_values_and_labels(self):
        result = extract_recording(self.frame, 'F0L', self.settings)
        self.assertTrue(result.label.eq(0).all())
        self.assertTrue(result.pv_power_mean.eq(4).all())
        self.assertTrue(result.ia_rms.eq(2).all())
        self.assertTrue(result.current_rms_spread.eq(0).all())
        self.assertTrue(result.Ipv_std.eq(0).all())
        self.assertEqual(len([c for c in result if c not in METADATA]), 15)

    def test_future_changes_do_not_affect_past(self):
        before = extract_recording(self.frame, 'F0L', self.settings)
        self.frame.loc[1600:, 'Ipv'] = 900
        after = extract_recording(self.frame, 'F0L', self.settings)
        pd.testing.assert_frame_equal(before[before.end_sample < 1600],
                                      after[after.end_sample < 1600])

    def test_proposals_remain_missing_and_startup_is_excluded(self):
        self.settings['startup_exclusion_s'] = 0.1
        result = extract_recording(self.frame, 'F1L', self.settings)
        self.assertTrue(result.label.isna().all())
        self.assertTrue((result.start_s >= self.frame.Time.iloc[0] + 0.1).all())

    def test_invalid_measurements_rejected(self):
        self.frame.loc[2, 'ia'] = np.nan
        with self.assertRaises(ValueError):
            extract_recording(self.frame, 'F0L', self.settings)


if __name__ == '__main__':
    unittest.main()
