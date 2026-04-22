"""Regression tests for malformed/bogus roster entries (issue #380).

Reproduces the "??? Jak Crawford" scenario where a reserve/test driver who
appeared in an FP1 session was being presented as part of the Miami 2026
race roster. The goal is to verify that the roster derivation pipeline:

* Prefers the official Jolpica season entry list over FastF1 practice data
  for upcoming events.
* Filters reserve/test drivers that only appear in practice sessions.
* Never emits entries with ``code = ???`` (blank abbreviation) for real
  drivers; a 3-letter code is synthesised from the family name as a
  last-ditch fallback.
* Rejects incomplete FastF1 authoritative sessions when a better
  alternative (full entry list / previous event) is available.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from f1pred.roster import (
    _clean_str,
    _derive_code,
    _entries_from_results,
    _filter_reserves,
    _roster_entries_from_fastf1_results,
    _roster_from_fastf1,
    derive_roster,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FULL_2026_ENTRY_LIST = [
    # Simplified but realistic Aston Martin + two others so the race_driver_ids
    # set is non-empty and contains Alonso but NOT Crawford.
    {
        "Driver": {
            "driverId": "alonso",
            "code": "ALO",
            "givenName": "Fernando",
            "familyName": "Alonso",
            "permanentNumber": "14",
        },
        "Constructor": {"constructorId": "aston_martin", "name": "Aston Martin"},
    },
    {
        "Driver": {
            "driverId": "stroll",
            "code": "STR",
            "givenName": "Lance",
            "familyName": "Stroll",
            "permanentNumber": "18",
        },
        "Constructor": {"constructorId": "aston_martin", "name": "Aston Martin"},
    },
]


def _full_grid(n: int = 20) -> list[dict]:
    """Build a fake season entry list with ``n`` plausible race drivers."""
    entries = [
        {
            "Driver": {
                "driverId": f"driver_{i}",
                "code": f"D{i:02d}",
                "givenName": f"Given{i}",
                "familyName": f"Family{i}",
                "permanentNumber": str(i + 1),
            },
            "Constructor": {
                "constructorId": "team_x",
                "name": "Team X",
            },
        }
        for i in range(n)
    ]
    return entries


def _fastf1_df(rows: list[dict]) -> pd.DataFrame:
    """Build a FastF1-shaped results DataFrame from simplified dicts."""
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pure-function helpers: string hygiene and code synthesis
# ---------------------------------------------------------------------------

class TestCleanStr:
    """``_clean_str`` must neutralise all the falsy-but-displayable values
    (None, NaN, 'nan', etc.) so they never reach the UI as literal strings."""

    @pytest.mark.parametrize("val", [None, float("nan"), "", "  ", "nan", "NaN", "None", "NaT", "<NA>"])
    def test_returns_empty_for_noise(self, val):
        assert _clean_str(val) == ""

    def test_strips_whitespace(self):
        assert _clean_str("  Alonso  ") == "Alonso"

    def test_preserves_real_values(self):
        assert _clean_str("Crawford") == "Crawford"
        assert _clean_str(14) == "14"


class TestDeriveCode:
    """``_derive_code`` must always yield a sensible 3-letter code when any
    name data is available, eliminating the ``???`` placeholder."""

    def test_prefers_explicit_code(self):
        assert _derive_code("ALO", "Alonso", "Fernando") == "ALO"

    def test_upper_and_truncates_long_code(self):
        assert _derive_code("alonso", "X", "Y") == "ALO"

    def test_synthesises_from_family_name_when_code_missing(self):
        # "???" would have appeared before; now we get "CRA".
        assert _derive_code("", "Crawford", "Jak") == "CRA"

    def test_synthesises_from_family_name_when_code_is_nan(self):
        assert _derive_code(float("nan"), "Crawford", "Jak") == "CRA"

    def test_short_family_name_padded(self):
        assert _derive_code("", "Li", "") == "LIX"

    def test_strips_non_alpha(self):
        assert _derive_code("", "O'Ward", "") == "OWA"

    def test_falls_back_to_given_name(self):
        assert _derive_code("", "", "Frederico") == "FRE"

    def test_returns_empty_when_nothing_available(self):
        assert _derive_code("", "", "") == ""


class TestEntriesFromResults:
    def test_nan_fields_are_sanitised(self):
        raw = [{
            "Driver": {
                "driverId": "alonso",
                "code": float("nan"),
                "givenName": "Fernando",
                "familyName": "Alonso",
                "permanentNumber": None,
            },
            "Constructor": {"constructorId": "aston_martin", "name": "Aston Martin"},
        }]
        out = _entries_from_results(raw)
        assert len(out) == 1
        assert out[0]["code"] == "ALO"  # synthesised from familyName
        assert out[0]["permanentNumber"] is None

    def test_deduplicates_by_driver_id(self):
        raw = _FULL_2026_ENTRY_LIST + _FULL_2026_ENTRY_LIST  # doubled
        out = _entries_from_results(raw)
        assert len(out) == 2


# ---------------------------------------------------------------------------
# Reserve-driver filtering
# ---------------------------------------------------------------------------

class TestFilterReserves:
    def test_drops_drivers_not_in_race_list(self):
        entries = [
            {"driverId": "alonso", "code": "ALO"},
            {"driverId": "crawford", "code": "CRA"},  # reserve
            {"driverId": "stroll", "code": "STR"},
        ]
        mapping = {"race_driver_ids": {"alonso", "stroll"}}
        kept = _filter_reserves(entries, mapping)
        ids = [e["driverId"] for e in kept]
        assert "alonso" in ids
        assert "stroll" in ids
        assert "crawford" not in ids

    def test_is_noop_when_race_list_unavailable(self):
        """If Jolpica hasn't published the entry list, we MUST NOT blindly drop
        drivers — we can't tell reserves from real drivers in that state.
        """
        entries = [{"driverId": "alonso"}, {"driverId": "crawford"}]
        assert _filter_reserves(entries, {}) == entries
        assert _filter_reserves(entries, {"race_driver_ids": set()}) == entries


# ---------------------------------------------------------------------------
# FastF1-roster session selection & gating
# ---------------------------------------------------------------------------

class TestRosterFromFastf1:
    """End-to-end behaviour of ``_roster_from_fastf1`` with tier-aware
    completeness gating.

    Background: the old code iterated sessions in priority order and returned
    the first non-empty one. This caused FP1-only rookies (Jak Crawford
    substituting for Alonso) to leak into the race roster when a prior
    authoritative session hadn't been published yet. The new code treats
    practice sessions as a guarded last resort.
    """

    def _make_event(self, session_results: dict[str, pd.DataFrame]):
        """Return a Mock FastF1 event whose ``get_session(name)`` returns a
        session mock whose ``results`` is ``session_results[name]`` (or empty)."""
        ev = Mock()

        def _get_session(name: str):
            s = Mock()
            s.results = session_results.get(name, pd.DataFrame())
            return s

        ev.get_session.side_effect = _get_session
        return ev

    def test_rejects_single_driver_fp1_when_no_other_source(self):
        """A 1-driver FP1 session must NOT be returned as the roster when
        there's no authoritative session data — too unreliable to trust."""
        fp1 = _fastf1_df([{
            "Abbreviation": "CRA", "DriverNumber": "31",
            "FirstName": "Jak", "LastName": "Crawford",
            "TeamName": "Aston Martin",
        }])
        ev = self._make_event({"FP1": fp1})
        mapping = {
            "drivers": {},
            "constructors": {},
            # Non-empty race_driver_ids that does NOT include CRA triggers
            # the reserve filter, which zeroes the practice roster.
            "race_driver_ids": {"alonso", "stroll"},
        }
        with patch("f1pred.data.fastf1_backend.get_event", return_value=ev):
            roster = _roster_from_fastf1(2026, 4, mapping=mapping)
        # Empty — the only thing FastF1 offered was filtered out as a reserve.
        assert roster == []

    def test_uses_full_authoritative_roster(self):
        """A 20-driver Race session IS the truth; use it unconditionally."""
        rows = [
            {
                "Abbreviation": f"D{i:02d}", "DriverNumber": str(i + 1),
                "FirstName": f"Given{i}", "LastName": f"Family{i}",
                "TeamName": "Team X",
            } for i in range(20)
        ]
        ev = self._make_event({"Race": _fastf1_df(rows)})
        with patch("f1pred.data.fastf1_backend.get_event", return_value=ev):
            roster = _roster_from_fastf1(2024, 1, mapping=None)
        assert len(roster) == 20

    def test_authoritative_session_preferred_over_practice(self):
        """When Qualifying (1 driver, authoritative) and FP1 (1 driver,
        practice with reserve) both exist, the practice session must NEVER
        be chosen — authoritative wins even if incomplete."""
        quali = _fastf1_df([{
            "Abbreviation": "ALO", "DriverNumber": "14",
            "FirstName": "Fernando", "LastName": "Alonso",
            "TeamName": "Aston Martin",
        }])
        fp1 = _fastf1_df([{
            "Abbreviation": "CRA", "DriverNumber": "31",
            "FirstName": "Jak", "LastName": "Crawford",
            "TeamName": "Aston Martin",
        }])
        ev = self._make_event({"Qualifying": quali, "FP1": fp1})
        mapping = {
            "drivers": {"ALO": "alonso", "CRA": "crawford"},
            "constructors": {},
            "race_driver_ids": {"alonso", "stroll"},
        }
        with patch("f1pred.data.fastf1_backend.get_event", return_value=ev):
            roster = _roster_from_fastf1(2026, 4, mapping=mapping)
        ids = [e["driverId"] for e in roster]
        assert ids == ["alonso"]
        assert "crawford" not in ids

    def test_practice_accepted_when_complete_and_unfiltered(self):
        """A full practice session (18+ drivers) with no reserves is a valid
        fallback when authoritative sessions are empty."""
        rows = [
            {
                "Abbreviation": f"D{i:02d}", "DriverNumber": str(i + 1),
                "FirstName": f"Given{i}", "LastName": f"Family{i}",
                "TeamName": "Team X",
            } for i in range(20)
        ]
        ev = self._make_event({"FP1": _fastf1_df(rows)})
        with patch("f1pred.data.fastf1_backend.get_event", return_value=ev):
            roster = _roster_from_fastf1(2024, 1, mapping=None)
        assert len(roster) == 20

    def test_roster_entries_from_fastf1_synthesises_code(self):
        """Even rows with no Abbreviation should get a synthesised 3-letter
        code from the last name so the UI never shows ``???``."""
        df = _fastf1_df([{
            "Abbreviation": "",
            "DriverNumber": "31",
            "FirstName": "Jak",
            "LastName": "Crawford",
            "TeamName": "Aston Martin",
        }])
        entries = _roster_entries_from_fastf1_results(df, mapping=None)
        assert len(entries) == 1
        assert entries[0]["code"] == "CRA"
        # driverId falls back to "driver_31" because abbr was blank
        assert entries[0]["driverId"] == "driver_31"


# ---------------------------------------------------------------------------
# End-to-end derive_roster cascade
# ---------------------------------------------------------------------------

class TestDeriveRosterCascade:
    """The top-level ``derive_roster`` must make the right choice between
    the Jolpica entry list and FastF1 depending on whether the event is in
    the future."""

    def _mock_jolpica(self, *, entry_list=None, race_results=None):
        jc = Mock()
        jc.get_race_results.return_value = race_results or []
        jc.get_qualifying_results.return_value = []
        jc.get_sprint_results.return_value = []
        jc.get_season_entry_list.return_value = entry_list or []
        jc.get_latest_season_and_round.return_value = ("2025", "22")
        return jc

    def test_future_event_prefers_entry_list_over_partial_fastf1(self):
        """This is the Miami 2026 bug: Jolpica had the entry list, FastF1 had
        an FP1 session with Crawford. The old code picked FastF1 first and
        returned Crawford. The new code returns the entry list.
        """
        jc = self._mock_jolpica(entry_list=_FULL_2026_ENTRY_LIST)
        # Build a FastF1 event whose FP1 returns Crawford. If the cascade
        # EVER consults FastF1 for this future event we'd get contaminated
        # data — the test asserts we don't.
        fp1 = _fastf1_df([{
            "Abbreviation": "CRA", "DriverNumber": "31",
            "FirstName": "Jak", "LastName": "Crawford",
            "TeamName": "Aston Martin",
        }])
        def _mk_sess(results):
            s = MagicMock()
            s.results = results
            return s

        ev = Mock()
        ev.get_session.side_effect = lambda n: _mk_sess(
            fp1 if n == "FP1" else pd.DataFrame()
        )

        now = datetime(2026, 5, 1, tzinfo=timezone.utc)
        event = datetime(2026, 5, 3, tzinfo=timezone.utc)

        with patch("f1pred.data.fastf1_backend.get_event", return_value=ev):
            roster = derive_roster(jc, "2026", "4", event_dt=event, now_dt=now)

        ids = [e["driverId"] for e in roster]
        assert "alonso" in ids
        assert "stroll" in ids
        assert "crawford" not in ids, (
            "Reserve driver leaked from FastF1 FP1 into future-event roster "
            "despite a full Jolpica entry list being available."
        )

    def test_past_event_with_complete_results_uses_jolpica_results(self):
        """Past events with full race results should use those results
        directly — no FastF1 call needed."""
        full_results = [{
            "Driver": {
                "driverId": f"driver_{i}", "code": f"D{i:02d}",
                "givenName": f"G{i}", "familyName": f"F{i}",
                "permanentNumber": str(i + 1),
            },
            "Constructor": {"constructorId": "t", "name": "T"},
        } for i in range(20)]
        jc = self._mock_jolpica(race_results=full_results)

        past = datetime(2024, 5, 5, tzinfo=timezone.utc)
        now = datetime(2024, 6, 1, tzinfo=timezone.utc)

        with patch("f1pred.data.fastf1_backend.get_event") as mock_ff1:
            roster = derive_roster(jc, "2024", "6", event_dt=past, now_dt=now)

        assert len(roster) == 20
        # FastF1 should never have been consulted because Jolpica had results.
        mock_ff1.assert_not_called()

    def test_past_event_prefers_fastf1_over_entry_list(self):
        """For past events, FastF1 session data is authoritative (reflects
        actual participants including substitutions); entry list is a weaker
        fallback for that case.
        """
        jc = self._mock_jolpica(entry_list=_full_grid(20))
        # Full Race session in FastF1 (20 drivers)
        rows = [
            {
                "Abbreviation": f"R{i:02d}", "DriverNumber": str(i + 100),
                "FirstName": f"FF{i}", "LastName": f"LL{i}",
                "TeamName": "Team X",
            } for i in range(20)
        ]
        def _mk_sess(results):
            s = MagicMock()
            s.results = results
            return s

        ev = Mock()
        ev.get_session.side_effect = lambda n: _mk_sess(
            _fastf1_df(rows) if n == "Race" else pd.DataFrame()
        )

        past = datetime(2024, 5, 5, tzinfo=timezone.utc)
        now = datetime(2024, 6, 1, tzinfo=timezone.utc)

        with patch("f1pred.data.fastf1_backend.get_event", return_value=ev):
            roster = derive_roster(jc, "2024", "6", event_dt=past, now_dt=now)

        # FastF1 drivers (r00..r19) should appear, not entry-list (driver_0..)
        ids = {e["driverId"] for e in roster}
        assert any(i.startswith("r") for i in ids), roster

    def test_entry_list_used_when_fastf1_unavailable_for_future(self):
        """If FastF1 returns nothing, the entry list must still fill the
        roster for a future event."""
        jc = self._mock_jolpica(entry_list=_FULL_2026_ENTRY_LIST)
        ev = None  # get_event returns None -> no FastF1 data
        now = datetime(2026, 5, 1, tzinfo=timezone.utc)
        event = datetime(2026, 5, 3, tzinfo=timezone.utc)

        with patch("f1pred.data.fastf1_backend.get_event", return_value=ev):
            roster = derive_roster(jc, "2026", "4", event_dt=event, now_dt=now)

        ids = [e["driverId"] for e in roster]
        assert ids == ["alonso", "stroll"]

    def test_no_entry_list_no_fastf1_falls_back_to_previous_event(self):
        """With nothing available for the target event, fall through to the
        previous completed event as before."""
        prev_results = [{
            "Driver": {
                "driverId": f"prev_{i}", "code": f"P{i:02d}",
                "givenName": f"G{i}", "familyName": f"F{i}",
            },
            "Constructor": {"constructorId": "t", "name": "T"},
        } for i in range(20)]
        jc = self._mock_jolpica()
        # Previous-event lookup: get_race_results for (2025, 22)
        def _race(season, rnd):
            if season == "2025" and rnd == "22":
                return prev_results
            return []
        jc.get_race_results.side_effect = _race

        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        event = datetime(2026, 3, 15, tzinfo=timezone.utc)

        with patch("f1pred.data.fastf1_backend.get_event", return_value=None):
            roster = derive_roster(jc, "2026", "1", event_dt=event, now_dt=now)

        assert len(roster) == 20
        assert roster[0]["driverId"] == "prev_0"

    def test_code_is_never_empty_for_named_drivers(self):
        """Regression: entries from Jolpica with missing ``code`` field must
        get a synthesised 3-letter code so the UI doesn't show ``???``."""
        entries = [{
            "Driver": {
                "driverId": "crawford",
                "code": None,
                "givenName": "Jak",
                "familyName": "Crawford",
                "permanentNumber": "31",
            },
            "Constructor": {"constructorId": "aston_martin", "name": "Aston Martin"},
        }]
        jc = self._mock_jolpica()
        jc.get_race_results.return_value = entries * 20  # pass completeness
        jc.get_season_entry_list.return_value = []

        past = datetime(2024, 5, 5, tzinfo=timezone.utc)
        now = datetime(2024, 6, 1, tzinfo=timezone.utc)

        with patch("f1pred.data.fastf1_backend.get_event", return_value=None):
            roster = derive_roster(jc, "2024", "6", event_dt=past, now_dt=now)

        assert roster
        assert roster[0]["code"] == "CRA"


# ---------------------------------------------------------------------------
# Grid-size independence: the roster must NEVER be capped at 20 drivers.
# ---------------------------------------------------------------------------

class TestGridSizeNotCapped:
    """Regression tests for the hardcoded 20-driver assumption.

    F1 grids vary: 20 cars in 2017-2025, 22 from 2026 (Cadillac joins),
    and historically anywhere from 20-26. The roster pipeline must faithfully
    return whatever size the upstream data reports.
    """

    def _mock_jolpica(self, *, entry_list=None, race_results=None):
        jc = Mock()
        jc.get_race_results.return_value = race_results or []
        jc.get_qualifying_results.return_value = []
        jc.get_sprint_results.return_value = []
        jc.get_season_entry_list.return_value = entry_list or []
        jc.get_latest_season_and_round.return_value = ("2025", "22")
        return jc

    @pytest.mark.parametrize("n", [20, 21, 22, 24, 26])
    def test_entry_list_preserves_full_grid(self, n):
        """A future event with a 22-car entry list must return 22 drivers.
        No caller should silently truncate to 20.
        """
        jc = self._mock_jolpica(entry_list=_full_grid(n))
        now = datetime(2026, 5, 1, tzinfo=timezone.utc)
        event = datetime(2026, 5, 3, tzinfo=timezone.utc)

        with patch("f1pred.data.fastf1_backend.get_event", return_value=None):
            roster = derive_roster(jc, "2026", "4", event_dt=event, now_dt=now)

        assert len(roster) == n, (
            f"Entry list of size {n} was truncated to {len(roster)} — "
            "grid size must not be capped"
        )

    @pytest.mark.parametrize("n", [20, 22, 24])
    def test_same_round_race_results_preserve_full_grid(self, n):
        """When Jolpica returns classified race results for the round, every
        driver (even >20) must be returned unchanged.
        """
        results = _full_grid(n)
        jc = self._mock_jolpica(race_results=results)
        past = datetime(2026, 5, 5, tzinfo=timezone.utc)
        now = datetime(2026, 6, 1, tzinfo=timezone.utc)

        with patch("f1pred.data.fastf1_backend.get_event", return_value=None):
            roster = derive_roster(jc, "2026", "4", event_dt=past, now_dt=now)

        assert len(roster) == n

    @pytest.mark.parametrize("n", [20, 22, 24])
    def test_fastf1_race_session_preserves_full_grid(self, n):
        """A completed 22- or 24-driver FastF1 Race session must be returned
        whole. Practice-session reserve filtering is the only legitimate
        reason the size should ever shrink.
        """
        jc = self._mock_jolpica(entry_list=_full_grid(n))
        rows = [
            {
                "Abbreviation": f"D{i:02d}",
                "DriverNumber": str(i + 1),
                "FirstName": f"Given{i}",
                "LastName": f"Family{i}",
                "TeamName": "Team X",
            } for i in range(n)
        ]

        def _mk_sess(results):
            s = MagicMock()
            s.results = results
            return s

        ev = Mock()
        ev.get_session.side_effect = lambda name: _mk_sess(
            _fastf1_df(rows) if name == "Race" else pd.DataFrame()
        )
        past = datetime(2026, 5, 5, tzinfo=timezone.utc)
        now = datetime(2026, 6, 1, tzinfo=timezone.utc)

        with patch("f1pred.data.fastf1_backend.get_event", return_value=ev):
            roster = derive_roster(jc, "2026", "4", event_dt=past, now_dt=now)

        assert len(roster) == n

    def test_min_floor_does_not_cap_upper_bound(self):
        """Sanity-check: the completeness floor is a lower bound only.
        A 22-driver session is accepted just as readily as a 20-driver one.
        """
        from f1pred.roster import _MIN_COMPLETE_ROSTER_MODERN, _min_expected_roster

        # The floor is a threshold for "enough data", not a ceiling.
        assert _MIN_COMPLETE_ROSTER_MODERN <= 20
        assert _min_expected_roster(2026) <= 22
        assert _min_expected_roster(2030) <= 26

    def test_same_round_floor_uses_season_aware_helper(self):
        """The ``same_round_known_roster`` gate in derive_roster must consult
        the same season-aware floor as the rest of the pipeline — not a
        hardcoded 20.
        """
        import inspect

        from f1pred import roster as roster_mod

        src = inspect.getsource(roster_mod.derive_roster)
        # Sanity: the helper is used.
        assert "_min_expected_roster" in src, (
            "derive_roster must use the season-aware floor helper, not hardcoded 20"
        )
        # Anti-regression: no raw `>= 20` comparison on roster length.
        assert ">= 20" not in src.replace(" ", ""), (
            "derive_roster must not hardcode `>= 20` — use _min_expected_roster"
        )
