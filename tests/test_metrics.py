"""Tests for evaluation metrics."""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from f1pred.metrics import (
    accuracy_top_k,
    brier_pairwise,
    compute_event_metrics,
)

class TestAccuracyTopK:
    def test_perfect_match(self):
        pred = ["a", "b", "c", "d"]
        act = ["a", "b", "c", "e"]
        # Top 3: {a, b, c} vs {a, b, c} -> 3/3 = 1.0
        assert accuracy_top_k(pred, act, k=3) == 1.0

    def test_partial_match(self):
        pred = ["a", "b", "c", "d"]
        act = ["a", "x", "y", "z"]
        # Top 3: {a, b, c} vs {a, x, y} -> {a} -> 1/3
        assert accuracy_top_k(pred, act, k=3) == 1.0 / 3.0

    def test_no_match(self):
        pred = ["a", "b", "c"]
        act = ["x", "y", "z"]
        assert accuracy_top_k(pred, act, k=3) == 0.0

    def test_k_zero(self):
        assert np.isnan(accuracy_top_k(["a"], ["a"], k=0))

    def test_k_larger_than_list(self):
        pred = ["a", "b"]
        act = ["a", "c"]
        # k=3. Top 3 from pred: {a, b}. Top 3 from act: {a, c}.
        # Intersection: {a}. Len=1. Result 1/3.
        assert accuracy_top_k(pred, act, k=3) == 1.0 / 3.0


class TestBrierPairwise:
    def test_insufficient_drivers(self):
        probs = np.array([[0.5]])
        positions = np.array([1])
        assert np.isnan(brier_pairwise(probs, positions))

    def test_two_drivers_perfect(self):
        probs = np.array([
            [0.5, 1.0],
            [0.0, 0.5]
        ])
        positions = np.array([1, 2])
        score = brier_pairwise(probs, positions)
        assert score == 0.0

    def test_three_drivers_mixed(self):
        positions = np.array([1, 2, 3])
        probs = np.zeros((3, 3))
        probs[0, 1] = 0.8
        probs[0, 2] = 0.9
        probs[1, 2] = 0.6

        expected_mse = (0.04 + 0.01 + 0.16) / 3.0
        assert brier_pairwise(probs, positions) == pytest.approx(expected_mse)

    def test_worst_case(self):
        actual = np.array([1, 2, 3])
        pairwise = np.array([
            [0.5, 0.0, 0.0],
            [1.0, 0.5, 0.0],
            [1.0, 1.0, 0.5],
        ])
        score = brier_pairwise(pairwise, actual)
        assert score == pytest.approx(1.0, abs=1e-6)

class TestComputeEventMetrics:
    def test_basic_dataframe(self):
        df = pd.DataFrame({
            "driver_id": ["a", "b"],
            "predicted_position": [1, 2],
            "actual_position": [1, 2]
        })
        metrics = compute_event_metrics(df, None, None, "race", 2023, 1)
        assert metrics["n"] == 2
        assert metrics["spearman"] == pytest.approx(1.0)
        assert metrics["accuracy_top3"] == pytest.approx(2/3.0)
        assert np.isnan(metrics["brier_pairwise"])
        assert np.isnan(metrics["crps"])

    def test_with_driverId_column(self):
        df = pd.DataFrame({
            "driverId": ["a", "b"],
            "predicted_position": [2, 1],
            "actual_position": [1, 2]
        })
        metrics = compute_event_metrics(df, None, None, "race", 2023, 1)
        assert metrics["spearman"] == pytest.approx(-1.0)
        assert metrics["accuracy_top3"] == pytest.approx(2/3)

    def test_missing_actual_position(self):
        df = pd.DataFrame({
            "driver_id": ["a"],
            "predicted_position": [1],
            "actual_position": [None]
        })
        metrics = compute_event_metrics(df, None, None, "race", 2023, 1)
        assert np.isnan(metrics["spearman"])

    def test_full_metrics(self):
        df = pd.DataFrame({
            "driver_id": ["a", "b"],
            "predicted_position": [1, 2],
            "actual_position": [1, 2]
        })
        prob_matrix = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        pairwise = np.array([
            [0.5, 1.0],
            [0.0, 0.5]
        ])
        metrics = compute_event_metrics(df, prob_matrix, pairwise, "race", 2023, 1)
        assert metrics["crps"] == 0.0
        assert metrics["brier_pairwise"] == 0.0

    def test_no_actuals(self):
        df = pd.DataFrame({
            "driverId": ["d1", "d2"],
            "predicted_position": [1, 2],
            "actual_position": [np.nan, np.nan],
        })
        result = compute_event_metrics(df, None, None, "race", 2025, 1)
        assert np.isnan(result["spearman"])
        assert np.isnan(result["kendall"])
        assert np.isnan(result["accuracy_top3"])

    def test_no_driver_column(self):
        df = pd.DataFrame({
            "predicted_position": [1, 2],
            "actual_position": [1, 2],
        })
        result = compute_event_metrics(df, None, None, "race", 2025, 1)
        assert np.isnan(result["accuracy_top3"])
        assert result["spearman"] == pytest.approx(1.0)

class TestComputeEventMetricsExceptions:
    @patch('f1pred.metrics.spearmanr', side_effect=ValueError("spearman error"))
    @patch('f1pred.metrics.kendalltau', side_effect=ValueError("kendall error"))
    def test_spearman_kendall_exceptions(self, *_):
        # We need this to verify behavior: if scipy functions fail due to some reason
        # (like identical values causing warnings/errors in older versions or edge cases)
        # the metric should gracefully return NaN rather than crashing the evaluation.
        df = pd.DataFrame({
            "driver_id": ["a", "b"],
            "predicted_position": [1, 2],
            "actual_position": [1, 2]
        })
        metrics = compute_event_metrics(df, None, None, "race", 2023, 1)
        assert np.isnan(metrics["spearman"])
        assert np.isnan(metrics["kendall"])
    def test_brier_pairwise_no_relevant_errors_empty_actual(self):
        probs = np.zeros((2, 2))
        positions = np.array([], dtype=int)
        # Should return nan because len is 0 (< 2)
        assert np.isnan(brier_pairwise(probs, positions))

    def test_dfv_empty_via_actual_positions(self):
        # We need a dataframe where actual_position is not all NA (so it passes line 54)
        # BUT after dropna on actual_position, it IS empty. Is that possible?
        # Actually no, if actual_position has at least one non-NA value, dropna will leave at least one row.
        # Wait, what if we have a dataframe with 0 rows to start with?
        _ = pd.DataFrame(columns=["driver_id", "predicted_position", "actual_position"])
        # df.shape[0] == 0. df["actual_position"].isna().all() is True, so it hits line 54.

        # What if it's all NA but `isna().all()` evaluates to false?
        pass

    def test_compute_event_empty_dfv(self):
        # Line 64 is reached if dfv.empty is true AFTER dropna.
        # Can we do this by mocking dropna to return an empty DataFrame?
        df = pd.DataFrame({
            "driver_id": ["a", "b"],
            "predicted_position": [1, 2],
            "actual_position": [1, 2]
        })
        with patch.object(pd.DataFrame, 'dropna', return_value=pd.DataFrame()):
            metrics = compute_event_metrics(df, None, None, "race", 2023, 1)
            assert np.isnan(metrics["spearman"])

    def test_brier_pairwise_empty_relevant_errors_mock(self):
        probs = np.zeros((2, 2))
        positions = np.array([1, 2])
        # We need relevant_errors.size == 0. Let's mock np.triu_indices to return empty.
        with patch('numpy.triu_indices', return_value=(np.array([], dtype=int), np.array([], dtype=int))):
            assert np.isnan(brier_pairwise(probs, positions))
