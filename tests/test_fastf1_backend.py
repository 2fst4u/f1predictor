"""Tests for f1pred.data.fastf1_backend.

This module was previously ~20% covered with no dedicated test file. The
position-derivation logic (patch_missing_positions), the datetime
normalisation helper, the cache-init guard, and the weather/classification
readers are all exercisable without a real FastF1 install by feeding
synthetic DataFrames / fake session objects, which is what these tests do.
"""
import sys
import types
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import pytest

from f1pred.data import fastf1_backend as fb


# --------------------------------------------------------------------------
# patch_missing_positions — pure ranking derivation
# --------------------------------------------------------------------------

class TestPatchMissingPositions:
    def test_qualifying_ranks_by_qtimes_q3_first(self):
        """Lower Q3 wins; missing Q3 falls back to later columns / last place."""
        df = pd.DataFrame({
            "Abbreviation": ["A", "B", "C"],
            "Position": [np.nan, np.nan, np.nan],
            "Q1": ["00:01:31", "00:01:30", "00:01:32"],
            "Q2": ["00:01:30", "00:01:29", "00:01:40"],
            "Q3": ["00:01:30", "00:01:29", ""],
        })
        out = fb.patch_missing_positions("Qualifying", df)
        pos = dict(zip(out["Abbreviation"], out["Position"]))
        assert pos == {"B": 1, "A": 2, "C": 3}

    def test_race_ranks_by_total_time(self):
        df = pd.DataFrame({
            "Abbreviation": ["A", "B", "C"],
            "Position": [np.nan, np.nan, np.nan],
            "Time": ["00:00:10", "00:00:09", "00:00:11"],
        })
        out = fb.patch_missing_positions("Race", df)
        pos = dict(zip(out["Abbreviation"], out["Position"]))
        assert pos == {"B": 1, "A": 2, "C": 3}

    def test_already_populated_returns_unchanged(self):
        df = pd.DataFrame({
            "Abbreviation": ["A", "B"],
            "Position": [1, 2],
            "Time": ["00:00:11", "00:00:09"],  # would re-rank if it tried
        })
        out = fb.patch_missing_positions("Race", df)
        # Untouched: not even re-sorted, despite Time disagreeing with Position.
        assert list(out["Position"]) == [1, 2]
        assert out is df

    def test_no_time_columns_leaves_positions_nan(self):
        df = pd.DataFrame({
            "Abbreviation": ["A", "B"],
            "Position": [np.nan, np.nan],
        })
        out = fb.patch_missing_positions("Race", df)
        assert out["Position"].isna().all()

    def test_all_qtimes_missing_leaves_positions_nan(self):
        df = pd.DataFrame({
            "Abbreviation": ["A", "B"],
            "Position": [np.nan, np.nan],
            "Q1": ["", "NaT"],
            "Q2": ["nan", ""],
            "Q3": ["", ""],
        })
        out = fb.patch_missing_positions("Qualifying", df)
        assert out["Position"].isna().all()

    def test_accepts_native_timedelta_columns(self):
        df = pd.DataFrame({
            "Abbreviation": ["A", "B"],
            "Position": [np.nan, np.nan],
            "Time": pd.to_timedelta(["00:00:10", "00:00:09"]),
        })
        out = fb.patch_missing_positions("Race", df)
        pos = dict(zip(out["Abbreviation"], out["Position"]))
        assert pos == {"B": 1, "A": 2}


# --------------------------------------------------------------------------
# _to_py_datetime — timezone normalisation
# --------------------------------------------------------------------------

class TestToPyDatetime:
    def test_naive_datetime_assumed_utc(self):
        out = fb._to_py_datetime(datetime(2024, 5, 1, 12, 0, 0))
        assert out.tzinfo == timezone.utc
        assert out.hour == 12

    def test_aware_datetime_converted_to_utc(self):
        plus2 = timezone(timedelta(hours=2))
        out = fb._to_py_datetime(datetime(2024, 5, 1, 12, 0, 0, tzinfo=plus2))
        assert out.tzinfo == timezone.utc
        assert out.hour == 10  # 12:00 +02:00 -> 10:00 UTC

    def test_pandas_timestamp_via_to_pydatetime(self):
        out = fb._to_py_datetime(pd.Timestamp("2024-05-01 12:00:00"))
        assert isinstance(out, datetime)
        assert out.tzinfo == timezone.utc

    def test_non_datetime_returns_none(self):
        assert fb._to_py_datetime("not-a-date") is None

    def test_none_returns_none(self):
        assert fb._to_py_datetime(None) is None


# --------------------------------------------------------------------------
# init_fastf1 — cache guard, never raises
# --------------------------------------------------------------------------

@pytest.fixture
def reset_cache_flag():
    original = fb._CACHE_INITED
    fb._CACHE_INITED = False
    yield
    fb._CACHE_INITED = original


def _fake_fastf1_module(enable_side_effect=None):
    mod = types.ModuleType("fastf1")

    class _Cache:
        @staticmethod
        def enable_cache(path):
            if enable_side_effect:
                enable_side_effect(path)

    mod.Cache = _Cache
    return mod


class TestInitFastf1:
    def test_returns_false_when_fastf1_missing(self, monkeypatch, reset_cache_flag):
        # Setting the module to None makes `import fastf1` raise ImportError.
        monkeypatch.setitem(sys.modules, "fastf1", None)
        assert fb.init_fastf1("/tmp/cache") is False

    def test_enables_cache_and_sets_flag(self, monkeypatch, reset_cache_flag):
        calls = []
        monkeypatch.setitem(
            sys.modules, "fastf1",
            _fake_fastf1_module(enable_side_effect=lambda p: calls.append(p)),
        )
        assert fb.init_fastf1("/tmp/cache") is True
        assert calls == ["/tmp/cache"]
        assert fb._CACHE_INITED is True

    def test_idempotent_when_already_inited(self, monkeypatch, reset_cache_flag):
        calls = []
        monkeypatch.setitem(
            sys.modules, "fastf1",
            _fake_fastf1_module(enable_side_effect=lambda p: calls.append(p)),
        )
        fb._CACHE_INITED = True
        assert fb.init_fastf1("/tmp/cache") is True
        assert calls == []  # enable_cache not called the second time

    def test_returns_false_on_cache_failure(self, monkeypatch, reset_cache_flag):
        def boom(path):
            raise RuntimeError("disk full")

        monkeypatch.setitem(
            sys.modules, "fastf1", _fake_fastf1_module(enable_side_effect=boom),
        )
        assert fb.init_fastf1("/tmp/cache") is False
        assert fb._CACHE_INITED is False


# --------------------------------------------------------------------------
# get_event — swallows failures
# --------------------------------------------------------------------------

def test_get_event_returns_none_when_fastf1_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastf1", None)
    assert fb.get_event(2024, 1) is None


# --------------------------------------------------------------------------
# get_session_weather_status
# --------------------------------------------------------------------------

class _FakeSession:
    def __init__(self, weather_data=None, laps=None, results=None, session_info=None, name="Race"):
        self.weather_data = weather_data
        self.laps = laps
        self.results = results
        self.session_info = session_info or {}
        self.name = name


class TestWeatherStatus:
    def test_none_when_event_missing(self, monkeypatch):
        monkeypatch.setattr(fb, "get_event", lambda s, r: None)
        assert fb.get_session_weather_status(2024, 1, "Race") is None

    def test_dry_when_no_rain_or_wet_compound(self, monkeypatch):
        sess = _FakeSession(
            weather_data=pd.DataFrame({"Rainfall": [False, False]}),
            laps=pd.DataFrame({"Compound": ["SOFT", "MEDIUM"]}),
        )
        monkeypatch.setattr(fb, "get_event", lambda s, r: object())
        monkeypatch.setattr(fb, "_load_session_with_timeout", lambda *a, **k: sess)
        out = fb.get_session_weather_status(2024, 1, "Race")
        assert out == {"is_wet": False, "rainfall": False, "track_status": "DRY"}

    def test_wet_from_rainfall(self, monkeypatch):
        sess = _FakeSession(
            weather_data=pd.DataFrame({"Rainfall": [False, True]}),
            laps=pd.DataFrame({"Compound": ["SOFT"]}),
        )
        monkeypatch.setattr(fb, "get_event", lambda s, r: object())
        monkeypatch.setattr(fb, "_load_session_with_timeout", lambda *a, **k: sess)
        out = fb.get_session_weather_status(2024, 1, "Race")
        assert out["is_wet"] is True
        assert out["rainfall"] is True
        assert out["track_status"] == "WET"

    def test_wet_from_intermediate_compound(self, monkeypatch):
        sess = _FakeSession(
            weather_data=pd.DataFrame({"Rainfall": [False]}),
            laps=pd.DataFrame({"Compound": ["SOFT", "INTERMEDIATE"]}),
        )
        monkeypatch.setattr(fb, "get_event", lambda s, r: object())
        monkeypatch.setattr(fb, "_load_session_with_timeout", lambda *a, **k: sess)
        out = fb.get_session_weather_status(2024, 1, "Race")
        assert out["is_wet"] is True
        assert out["track_status"] == "WET"

    def test_returns_none_on_load_failure(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("timeout")

        monkeypatch.setattr(fb, "get_event", lambda s, r: object())
        monkeypatch.setattr(fb, "_load_session_with_timeout", boom)
        assert fb.get_session_weather_status(2024, 1, "Race") is None


# --------------------------------------------------------------------------
# get_session_classification
# --------------------------------------------------------------------------

class TestGetSessionClassification:
    def test_none_when_event_missing(self, monkeypatch):
        monkeypatch.setattr(fb, "get_event", lambda s, r: None)
        assert fb.get_session_classification(2024, 1, "Race") is None

    def test_returns_populated_results(self, monkeypatch):
        results = pd.DataFrame({"Abbreviation": ["A", "B"], "Position": [1, 2]})
        sess = _FakeSession(results=results, name="Race")
        monkeypatch.setattr(fb, "get_event", lambda s, r: object())
        monkeypatch.setattr(fb, "_load_session_with_timeout", lambda *a, **k: sess)
        out = fb.get_session_classification(2024, 1, "Race")
        assert list(out["Position"]) == [1, 2]

    def test_none_on_load_failure(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(fb, "get_event", lambda s, r: object())
        monkeypatch.setattr(fb, "_load_session_with_timeout", boom)
        assert fb.get_session_classification(2024, 1, "Race") is None


# --------------------------------------------------------------------------
# get_session_times
# --------------------------------------------------------------------------

class TestGetSessionTimes:
    def test_none_when_event_missing(self):
        assert fb.get_session_times(None, "Race") is None

    def test_reads_start_and_end_from_session_info(self, monkeypatch):
        start = datetime(2024, 5, 1, 13, 0, tzinfo=timezone.utc)
        end = datetime(2024, 5, 1, 15, 0, tzinfo=timezone.utc)
        sess = _FakeSession(session_info={"StartDate": start, "EndDate": end})
        monkeypatch.setattr(fb, "_load_session_with_timeout", lambda *a, **k: sess)
        out = fb.get_session_times(object(), "Race")
        assert out == (start, end)

    def test_none_on_load_failure(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(fb, "_load_session_with_timeout", boom)
        assert fb.get_session_times(object(), "Race") is None
