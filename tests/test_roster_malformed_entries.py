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

    def test_rejects_fp1_with_any_reserve_present(self):
        """An FP1 session containing a reserve driver must be rejected in
        full when the entry list is available — even if the reserve would be
        filterable out, the remaining drivers are not a faithful picture of
        the race roster (they're just the subset of regulars who happened to
        run in FP1). The reject is on presence of a reserve, not count.
        """
        fp1 = _fastf1_df([{
            "Abbreviation": "CRA", "DriverNumber": "31",
            "FirstName": "Jak", "LastName": "Crawford",
            "TeamName": "Aston Martin",
        }])
        ev = self._make_event({"FP1": fp1})
        mapping = {
            "drivers": {"CRA": "crawford", "ALO": "alonso"},
            "constructors": {},
            # Non-empty race_driver_ids — enables reserve filter.
            "race_driver_ids": {"alonso", "stroll"},
        }
        with patch("f1pred.data.fastf1_backend.get_event", return_value=ev):
            roster = _roster_from_fastf1(2026, 4, mapping=mapping)
        # Empty — Crawford is a reserve, so the whole session is rejected.
        assert roster == []

    def test_authoritative_race_accepted_regardless_of_size(self):
        """A 14-driver classified Race session (e.g. after mass DSQ/DNF at
        Spa) must be returned as-is. Disqualified or not-classified drivers
        are legitimately absent from the classification; demanding a minimum
        count would wrongly reject real race data.
        """
        rows = [
            {
                "Abbreviation": f"D{i:02d}", "DriverNumber": str(i + 1),
                "FirstName": f"Given{i}", "LastName": f"Family{i}",
                "TeamName": "Team X",
            } for i in range(14)
        ]
        ev = self._make_event({"Race": _fastf1_df(rows)})
        with patch("f1pred.data.fastf1_backend.get_event", return_value=ev):
            roster = _roster_from_fastf1(2024, 1, mapping=None)
        assert len(roster) == 14, (
            "14-driver classified race must be accepted verbatim; no "
            "minimum-count gate is permitted in the authoritative path."
        )

    def test_authoritative_single_driver_session_still_wins(self):
        """If the only data FastF1 has is a 1-driver authoritative session
        (e.g. Qualifying has only produced one lap-time so far), it still
        beats any practice session fallback. There is no minimum count —
        incomplete authoritative data is still authoritative.
        """
        quali = _fastf1_df([{
            "Abbreviation": "ALO", "DriverNumber": "14",
            "FirstName": "Fernando", "LastName": "Alonso",
            "TeamName": "Aston Martin",
        }])
        ev = self._make_event({"Qualifying": quali})
        with patch("f1pred.data.fastf1_backend.get_event", return_value=ev):
            roster = _roster_from_fastf1(2026, 4, mapping=None)
        assert len(roster) == 1
        assert roster[0]["driverId"] == "alo"

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
        """When Qualifying (authoritative) and FP1 (practice with reserve)
        both exist, the practice session must NEVER be chosen — authoritative
        wins irrespective of driver counts."""
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

    def test_practice_accepted_when_no_reserves_and_no_entry_list(self):
        """When no entry list is available to check reserves against, the
        practice session is accepted verbatim at whatever size — the pipeline
        has no way to do better.
        """
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

    def test_practice_accepted_when_all_drivers_are_registered(self):
        """A practice session where every listed driver is in the season's
        race-driver set is accepted. The gate is "zero reserves present",
        not "≥ N drivers present".
        """
        # 3-driver FP3, but all three ARE registered race drivers. Unusual
        # (most FP3 sessions have the full grid), but the pipeline must
        # accept it when it's all we have.
        rows = [
            {"Abbreviation": "ALO", "DriverNumber": "14",
             "FirstName": "Fernando", "LastName": "Alonso",
             "TeamName": "Aston Martin"},
            {"Abbreviation": "STR", "DriverNumber": "18",
             "FirstName": "Lance", "LastName": "Stroll",
             "TeamName": "Aston Martin"},
            {"Abbreviation": "VER", "DriverNumber": "1",
             "FirstName": "Max", "LastName": "Verstappen",
             "TeamName": "Red Bull"},
        ]
        mapping = {
            "drivers": {"ALO": "alonso", "STR": "stroll", "VER": "verstappen"},
            "constructors": {},
            "race_driver_ids": {"alonso", "stroll", "verstappen"},
        }
        ev = self._make_event({"FP3": _fastf1_df(rows)})
        with patch("f1pred.data.fastf1_backend.get_event", return_value=ev):
            roster = _roster_from_fastf1(2026, 4, mapping=mapping)
        ids = {e["driverId"] for e in roster}
        assert ids == {"alonso", "stroll", "verstappen"}

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

    def test_future_event_uses_previous_round_not_drivers_json(self):
        """Regression for the Miami 2026 bug.

        ``Jolpica.get_season_entry_list`` is implemented on top of
        ``/{season}/drivers.json``, which is a cumulative UNION of every
        driver seen in *any* session — including practice reservists like
        Jak Crawford. Treating that as the authoritative entry list yields
        23-driver grids with reservists on them.

        The correct answer for an upcoming round is the roster of the most
        recently classified round of the same season. This test asserts the
        cascade ignores ``drivers.json`` and uses the previous round instead,
        so a reserve who *only* appears in drivers.json never makes it into
        the grid.
        """
        # 22-driver previous-round classification (the actual 2026 roster).
        prev_results = [{
            "Driver": {
                "driverId": f"driver_{i}", "code": f"D{i:02d}",
                "givenName": f"G{i}", "familyName": f"F{i}",
                "permanentNumber": str(i + 1),
            },
            "Constructor": {"constructorId": "t", "name": "T"},
        } for i in range(22)]

        # Jolpica's drivers.json additionally contains Crawford (a reserve
        # who did FP1 earlier in the season but has never raced / qualified).
        drivers_json_superset = prev_results + [{
            "Driver": {
                "driverId": "crawford", "code": "CRA",
                "givenName": "Jak", "familyName": "Crawford",
                "permanentNumber": "31",
            },
            "Constructor": {"constructorId": "aston_martin", "name": "Aston Martin"},
        }]

        jc = self._mock_jolpica(entry_list=drivers_json_superset)
        # Target round has no results yet.
        def _race(season, rnd):
            if season == "2026" and rnd == "3":
                return prev_results
            return []
        jc.get_race_results.side_effect = _race
        # Season schedule for _previous_completed_event_global lookup
        jc.get_season_schedule.return_value = [
            {"round": str(i)} for i in range(1, 5)
        ]
        jc.get_latest_season_and_round.return_value = ("2026", "3")

        # Build a FastF1 event whose FP1 also returns Crawford. If the
        # cascade ever consults FastF1 or drivers.json for this future
        # event we'd get contaminated data — this asserts we don't.
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
        assert len(ids) == 22, f"Expected 22-driver grid, got {len(ids)}: {ids}"
        assert "crawford" not in ids, (
            "Reserve from drivers.json / FP1 leaked into upcoming-event roster. "
            "derive_roster must use previous completed round, not drivers.json."
        )

    def test_future_event_ignores_drivers_json_even_when_it_is_the_only_source(self):
        """If no previous round has been classified (e.g. round 1) and there's
        no FastF1 data either, we return an empty roster rather than trusting
        ``drivers.json``. An empty roster signals "unknown", which is the
        truthful answer — the call site decides how to surface that.
        """
        # drivers.json contains someone, but no race/quali has been classified.
        jc = self._mock_jolpica(entry_list=_FULL_2026_ENTRY_LIST)
        jc.get_race_results.return_value = []
        jc.get_qualifying_results.return_value = []
        jc.get_season_schedule.return_value = [{"round": "1"}]
        jc.get_latest_season_and_round.return_value = ("2026", "1")

        now = datetime(2026, 3, 1, tzinfo=timezone.utc)
        event = datetime(2026, 3, 15, tzinfo=timezone.utc)

        with patch("f1pred.data.fastf1_backend.get_event", return_value=None), \
             patch("f1pred.roster._previous_completed_event_global", return_value=None):
            roster = derive_roster(jc, "2026", "1", event_dt=event, now_dt=now)

        assert roster == [], (
            "When no classified round exists yet, roster must not fall back to "
            "drivers.json — that endpoint contains reservists."
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

    def test_previous_round_used_when_fastf1_unavailable_for_future(self):
        """If FastF1 returns nothing, the roster for a future event must be
        taken from the most recent completed round — NOT from Jolpica's
        drivers.json superset.
        """
        prev_results = [{
            "Driver": {
                "driverId": "alonso", "code": "ALO",
                "givenName": "Fernando", "familyName": "Alonso",
                "permanentNumber": "14",
            },
            "Constructor": {"constructorId": "aston_martin", "name": "Aston Martin"},
        }, {
            "Driver": {
                "driverId": "stroll", "code": "STR",
                "givenName": "Lance", "familyName": "Stroll",
                "permanentNumber": "18",
            },
            "Constructor": {"constructorId": "aston_martin", "name": "Aston Martin"},
        }]

        jc = self._mock_jolpica()
        def _race(season, rnd):
            if season == "2026" and rnd == "3":
                return prev_results
            return []
        jc.get_race_results.side_effect = _race
        jc.get_season_schedule.return_value = [
            {"round": str(i)} for i in range(1, 5)
        ]
        jc.get_latest_season_and_round.return_value = ("2026", "3")

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
# Grid-size independence: the roster must accept any size — no cap, no floor.
# ---------------------------------------------------------------------------

class TestRosterSizeIsUnconstrained:
    """Regression tests that assert the pipeline applies NEITHER a minimum
    nor maximum driver count.

    F1 grids vary: 20 cars in 2017-2025, 22 from 2026 (Cadillac joins),
    historically anywhere from 20-26. And within a single event, cars can
    be disqualified, fail to start, or not be classified — so a legitimately
    published classification can have 14 drivers (Spa 2021) or fewer. The
    roster pipeline must faithfully return whatever size the upstream data
    reports, with no caller silently truncating, padding, or rejecting.
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
    def test_previous_round_preserves_full_grid(self, n):
        """A future event whose previous completed round had n drivers must
        return exactly n drivers, for any grid size.

        Historically this test asserted the same invariant for Jolpica's
        "entry list" — but that endpoint is actually a cumulative union of
        ``drivers.json`` that includes FP1 reservists, so we no longer use
        it. The current invariant is that the previous completed round is
        the ground truth for an upcoming round.
        """
        prev_results = _full_grid(n)  # shape matches race Results rows
        jc = self._mock_jolpica()
        def _race(season, rnd):
            if season == "2026" and rnd == "3":
                return prev_results
            return []
        jc.get_race_results.side_effect = _race
        jc.get_season_schedule.return_value = [
            {"round": str(i)} for i in range(1, 5)
        ]
        jc.get_latest_season_and_round.return_value = ("2026", "3")

        now = datetime(2026, 5, 1, tzinfo=timezone.utc)
        event = datetime(2026, 5, 3, tzinfo=timezone.utc)

        with patch("f1pred.data.fastf1_backend.get_event", return_value=None):
            roster = derive_roster(jc, "2026", "4", event_dt=event, now_dt=now)

        assert len(roster) == n, (
            f"Previous-round roster of size {n} was truncated to {len(roster)} — "
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

    # -- No minimum: DSQ / not-classified scenarios must be accepted --------

    @pytest.mark.parametrize("n", [1, 5, 10, 14, 17])
    def test_same_round_accepts_below_20_drivers(self, n):
        """Jolpica returning a small classification (because of mass DSQ,
        DNS, or not-classified) MUST be accepted at face value. The old
        ``len(same) >= 20`` and ``len(same) >= _min_expected_roster`` gates
        would have wrongly fallen through to the entry list / previous-event
        cascade and replaced legitimate results with stale data.
        """
        results = _full_grid(n)
        jc = self._mock_jolpica(race_results=results)
        past = datetime(2024, 5, 5, tzinfo=timezone.utc)
        now = datetime(2024, 6, 1, tzinfo=timezone.utc)

        with patch("f1pred.data.fastf1_backend.get_event", return_value=None):
            roster = derive_roster(jc, "2024", "6", event_dt=past, now_dt=now)

        assert len(roster) == n, (
            f"Small classification of size {n} was replaced by fallback data "
            "— roster must have no minimum-count gate."
        )

    @pytest.mark.parametrize("n", [1, 5, 10, 14, 17])
    def test_fastf1_authoritative_accepts_below_20_drivers(self, n):
        """A classified Race session with fewer than 20 drivers (after
        widespread DSQ or mechanical carnage) must be returned as-is. There
        is no minimum-driver gate in the authoritative path.
        """
        rows = [
            {
                "Abbreviation": f"D{i:02d}", "DriverNumber": str(i + 1),
                "FirstName": f"Given{i}", "LastName": f"Family{i}",
                "TeamName": "Team X",
            } for i in range(n)
        ]
        jc = self._mock_jolpica()  # empty race_results -> cascade to FastF1

        def _mk_sess(results):
            s = MagicMock()
            s.results = results
            return s

        ev = Mock()
        ev.get_session.side_effect = lambda name: _mk_sess(
            _fastf1_df(rows) if name == "Race" else pd.DataFrame()
        )
        past = datetime(2024, 5, 5, tzinfo=timezone.utc)
        now = datetime(2024, 6, 1, tzinfo=timezone.utc)

        with patch("f1pred.data.fastf1_backend.get_event", return_value=ev):
            roster = derive_roster(jc, "2024", "6", event_dt=past, now_dt=now)

        assert len(roster) == n

    def test_no_minimum_roster_constants_remain(self):
        """Anti-regression: neither the module-level floor constants nor
        the ``_min_expected_roster`` helper should reappear. Any minimum
        encoding is by definition a violation of the "cars may be disqualified
        or not classified" invariant.
        """
        from f1pred import roster as roster_mod

        for banned in (
            "_MIN_COMPLETE_ROSTER_MODERN",
            "_MIN_COMPLETE_ROSTER_LEGACY",
            "_min_expected_roster",
        ):
            assert not hasattr(roster_mod, banned), (
                f"{banned} must not exist — minimum-count gates reject "
                "legitimately small rosters (DSQ / not classified)."
            )

    def test_no_minimum_gate_in_derive_roster_source(self):
        """Source-level anti-regression: nobody reintroduces a numeric
        minimum driver-count check in the cascade.
        """
        import inspect
        import re

        from f1pred import roster as roster_mod

        src = inspect.getsource(roster_mod.derive_roster)
        normalised = src.replace(" ", "")
        # No `len(...) >= <number>` or `<number> <=` patterns on rosters.
        # We just look for the giveaway shapes.
        assert not re.search(r"len\([^)]+\)>=\d", normalised), (
            "derive_roster must not gate on a minimum driver count."
        )
        assert "_min_expected_roster" not in src, (
            "derive_roster must not use _min_expected_roster (removed)."
        )

    def test_no_minimum_gate_in_fastf1_roster_source(self):
        """Same anti-regression but for the FastF1 path — no min-driver gate
        on authoritative or practice results.
        """
        import inspect
        import re

        from f1pred import roster as roster_mod

        src = inspect.getsource(roster_mod._roster_from_fastf1)
        normalised = src.replace(" ", "")
        assert not re.search(r"len\([^)]+\)>=\d", normalised), (
            "_roster_from_fastf1 must not gate on a minimum driver count."
        )
        assert not re.search(r"min_expected", src), (
            "min_expected variables must not return to _roster_from_fastf1."
        )
