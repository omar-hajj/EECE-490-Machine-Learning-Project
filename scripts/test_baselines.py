import unittest
import numpy as np
import pandas as pd
from baselines import CURRENT, VOLTAGE, SPREAD, fit_rules, predict_rules, persistent_alarm


class BaselineTests(unittest.TestCase):
    def table(self):
        table = pd.DataFrame({c: [2., 2., 2., 2.] for c in CURRENT + VOLTAGE + SPREAD})
        return table.assign(recording='F0L', label=0, start_s=0.,
                            end_s=np.arange(4) * .02, end_sample=np.arange(4) * 200)

    def test_partition_guard(self):
        with self.assertRaises(ValueError):
            fit_rules(self.table(), ['F0M'])

    def test_only_accepted_normals_calibrate(self):
        table = self.table()
        table.loc[3, 'label'] = np.nan
        table.loc[3, 'ia_rms'] = 100
        self.assertEqual(fit_rules(table, ['F0L'])['B0_upper']['ia_rms'], 2.)

    def test_rule_and_persistence(self):
        table = self.table()
        model = fit_rules(table, ['F0L'])
        table['ia_rms'] = 3.
        result = predict_rules(table, model, 200)
        self.assertEqual(result.B0_alarm.tolist(), [False, False, True, True])
        self.assertFalse(result.B1_alarm.any())

    def test_gap_resets(self):
        self.assertEqual(persistent_alarm([True]*4, [0, 200, 600, 800], 200, 3).tolist(),
                         [False]*4)

    def test_recording_resets(self):
        table = self.table()
        model = fit_rules(table, ['F0L'])
        table['ia_rms'] = 3.
        table.loc[2:, 'recording'] = 'F0M'
        self.assertFalse(predict_rules(table, model, 200).B0_alarm.any())


if __name__ == '__main__':
    unittest.main()
