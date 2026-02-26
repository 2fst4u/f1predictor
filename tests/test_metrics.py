import numpy as np
import pandas as pd
import pytest
from f1pred.metrics import (
    accuracy_top_k,
    brier_pairwise,
    crps_position,
    compute_event_metrics
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
        # Driver 0 finishes 1st (pos 1), Driver 1 finishes 2nd (pos 2).
        # Actual matrix Y:
        # 0 vs 0: False
        # 0 vs 1: pos[0](1) < pos[1](2) -> True (1.0)
        # 1 vs 0: pos[1](2) < pos[0](1) -> False (0.0)
        # 1 vs 1: False
        # Y = [[0, 1], [0, 0]]
        # Pairwise prob should match Y for 0 error.
        probs = np.array([
            [0.5, 1.0],
            [0.0, 0.5]
        ])
        positions = np.array([1, 2])
        score = brier_pairwise(probs, positions)
        assert score == 0.0

    def test_three_drivers_mixed(self):
        # Drivers: A(1), B(2), C(3)
        positions = np.array([1, 2, 3])
        # Y matrix (i < j ?):
        #   A B C
        # A 0 1 1
        # B 0 0 1
        # C 0 0 0
        # Upper triangle (k=1): (0,1)=>1, (0,2)=>1, (1,2)=>1.

        # Probs:
        # A>B=0.8 (err=(0.8-1)^2 = 0.04)
        # A>C=0.9 (err=(0.9-1)^2 = 0.01)
        # B>C=0.6 (err=(0.6-1)^2 = 0.16)
        probs = np.zeros((3, 3))
        probs[0, 1] = 0.8
        probs[0, 2] = 0.9
        probs[1, 2] = 0.6

        expected_mse = (0.04 + 0.01 + 0.16) / 3.0
        assert brier_pairwise(probs, positions) == pytest.approx(expected_mse)

    def test_nan_result(self):
        # Should result in NaN if no relevant errors (though logic implies if n>=2 we have pairs)
        # If relevant_errors.size == 0 is checked.
        pass

class TestCRPSPosition:
    def test_winner_perfect(self):
        # Predicted 100% on pos 1.
        prob = np.array([1.0, 0.0, 0.0])
        actual = 1
        # F = [1, 1, 1]
        # H (actual=1) -> H[:0]=0 -> H=[1, 1, 1]
        # (F-H)^2 = 0
        assert crps_position(prob, actual) == 0.0

    def test_second_place(self):
        # Predicted 50% pos 1, 50% pos 2.
        prob = np.array([0.5, 0.5, 0.0])
        actual = 2
        # F = [0.5, 1.0, 1.0]
        # H (actual=2) -> H[:1]=0 -> H=[0, 1, 1]
        # Diff = [0.5, 0, 0]
        # Sq = [0.25, 0, 0]
        # Mean = 0.25 / 3 = 0.08333...
        assert crps_position(prob, actual) == pytest.approx(0.25 / 3.0)

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
        # top 3 of 2 items: {a, b}. intersection {a, b}. len 2. 2/3 = 0.666...
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
        # Pred order: b, a. Actual order: a, b.
        # Top 3 (k=3) intersection of {a,b} and {a,b} is {a,b} size 2.
        # Accuracy = 2/3 = 0.666...
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
        # 2 drivers
        df = pd.DataFrame({
            "driver_id": ["a", "b"],
            "predicted_position": [1, 2],
            "actual_position": [1, 2]
        })

        # Prob matrix 2x2
        # Driver A: [1.0, 0.0] (sure 1st)
        # Driver B: [0.0, 1.0] (sure 2nd)
        prob_matrix = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])

        # Pairwise
        # A(0) vs B(1). A is 1st, B is 2nd. A < B.
        # Y[0,1] = 1.
        # Prob[0,1] = 1.0 (A beats B)
        pairwise = np.array([
            [0.5, 1.0],
            [0.0, 0.5]
        ])

        metrics = compute_event_metrics(df, prob_matrix, pairwise, "race", 2023, 1)

        assert metrics["crps"] == 0.0
        assert metrics["brier_pairwise"] == 0.0

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        metrics = compute_event_metrics(df, None, None, "race", 2023, 1)
        assert metrics["n"] == 0
        assert np.isnan(metrics["spearman"])
