"""Small regression checks for timing and peak metrics, using synthetic data."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import peak_forecasting as model


class PeakMetricsTests(unittest.TestCase):
    def test_peak_metrics_include_zero_for_peak_overprediction(self):
        frame = pd.DataFrame({
            model.TARGET: [180.0, 190.0, 170.0, 200.0],
            "pred": [170.0, 185.0, 180.0, 200.0],
        })
        result = model.metrics(frame)
        self.assertAlmostEqual(result["MAE"], 6.25)
        self.assertAlmostEqual(result["피크MAE"], 5.0)
        self.assertAlmostEqual(result["과소예측피크"], 5.0)
        self.assertAlmostEqual(result["과소예측P95"], 9.5)
        self.assertAlmostEqual(result["PeakRecall"], 2 / 3)
        self.assertAlmostEqual(result["PeakPrecision"], 2 / 3)
        self.assertEqual(result["MissedPeak"], 1)
        self.assertAlmostEqual(result["RMSE"], 7.5)

    def test_no_peak_denominators_are_nan(self):
        frame = pd.DataFrame({model.TARGET: [100.0], "pred": [101.0]})
        result = model.metrics(frame)
        self.assertTrue(np.isnan(result["피크MAE"]))
        self.assertTrue(np.isnan(result["PeakRecall"]))
        self.assertTrue(np.isnan(result["PeakPrecision"]))


class TrainingCutoffTests(unittest.TestCase):
    def setUp(self):
        timestamps = pd.to_datetime([
            "2021-08-29 21:00", "2021-08-29 22:00",
            "2021-08-29 23:00", "2021-08-30 00:00",
            "2021-08-30 01:00",
        ])
        self.frame = pd.DataFrame({
            "ts": timestamps,
            "date": timestamps.normalize(),
            "시간": timestamps.hour,
            "x": np.arange(len(timestamps)),
            model.TARGET: 150.0,
        })

    def capture_training(self, horizon, copy_days=None):
        seen = []
        def predict(_name, train, test, _features, _seed):
            seen.append(train["ts"].tolist())
            return np.zeros(len(test))
        with patch.object(model, "fit_predict", side_effect=predict):
            model.rolling(self.frame, set() if copy_days is None else copy_days, ["x"], horizon, period="final")
        return seen[0]

    def test_A_excludes_sunday_23h_incomplete_at_issue(self):
        seen = self.capture_training("A")
        self.assertEqual(max(seen), pd.Timestamp("2021-08-29 22:00"))

    def test_B_can_use_completed_sunday_23h(self):
        seen = self.capture_training("B")
        self.assertEqual(max(seen), pd.Timestamp("2021-08-29 23:00"))

    def test_copied_day_is_not_in_training(self):
        seen = self.capture_training("B", {pd.Timestamp("2021-08-29")})
        self.assertEqual(seen, [])

    def test_A_rejects_B_only_power_lags(self):
        with self.assertRaises(ValueError):
            model.rolling(self.frame, set(), ["전력_1h전"], "A", period="final")


class FeatureTimingTests(unittest.TestCase):
    @staticmethod
    def raw_frame():
        timestamps = pd.date_range("2021-07-01", periods=24 * 45, freq="h")
        i = np.arange(len(timestamps), dtype=float)
        power = 120 + (i % 29) + 8 * np.sin(i / 7)
        frame = pd.DataFrame({
            "date": timestamps.normalize(),
            "시간": timestamps.hour,
            "ts": timestamps,
            "생산량": np.where((timestamps.hour >= 8) & (timestamps.hour < 18), 10.0, 0.0),
            "기온": 25 + np.sin(i / 24),
            "습도": 60.0,
            "풍속": 2.0,
            "시간당강수량": 0.0,
            "전기요금(계절)": 1.0,
            "시간단위_학습사용": 1,
        })
        for column in model.POWER_COLS:
            frame[column] = power
        return frame

    @staticmethod
    def features(frame):
        return model.add_candidates(model.add_base_features(model.add_day_info(frame)))

    def check_invariance(self, horizon):
        raw = self.raw_frame()
        clean = self.features(raw.copy())
        target_day = pd.Timestamp("2021-08-10")
        if horizon == "A":
            cutoff = target_day - pd.Timedelta(hours=1)
            rows = clean["date"] == target_day
            columns = model.FINAL_A
        else:
            cutoff = target_day + pd.Timedelta(hours=12)
            rows = clean["ts"] == cutoff
            columns = model.FINAL_B
        dirty = raw.copy()
        mask = dirty["ts"] >= cutoff
        dirty.loc[mask, model.POWER_COLS] = dirty.loc[mask, model.POWER_COLS] * 3 + 777
        changed = self.features(dirty)
        np.testing.assert_allclose(
            clean.loc[rows, columns].to_numpy(float),
            changed.loc[rows, columns].to_numpy(float),
            equal_nan=True,
        )

    def test_A_final_features_ignore_unavailable_future_power(self):
        self.check_invariance("A")

    def test_B_final_features_ignore_unavailable_future_power(self):
        self.check_invariance("B")


if __name__ == "__main__":
    unittest.main()
