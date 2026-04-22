from __future__ import annotations
from typing import List, Dict, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
import time

from .util import get_logger
from .data.jolpica import JolpicaClient

logger = get_logger(__name__)

# Maximum seconds to spend inside _roster_from_fastf1 before giving up.
# FastF1 sess.load() has no internal timeout, so this prevents hangs when
# F1 live-timing servers are slow or sessions don't exist yet.
_FASTF1_ROSTER_TIMEOUT = 15

# F1 grid size is NEVER assumed or enforced. Grids vary widely — 20 cars in
# 2017-2025, 22 from 2026 with Cadillac joining, historically 20-26 — and
# legitimate end-of-session rosters can be smaller still because cars are
# routinely disqualified, failed to start, or not classified (didn't complete
# 90% of race distance). A 14-driver classified race at Spa-Francorchamps
# after a chaotic wet race is just as real and authoritative as a 22-driver
# dry race. The pipeline MUST NOT apply a minimum driver count anywhere:
# doing so would reject legitimate results and trigger spurious fallbacks.
#
# The real question a roster pipeline must answer is "is this data source
# authoritative?", not "does it have enough rows?". Authority is decided by
# *which session* the data came from, and (for practice sessions) whether
# every driver listed is actually registered as a race driver for the season.
# Reserve/test drivers are rejected by name (via the entry list), not by
# driver count.

# Authoritative sessions — these determine the true race-weekend roster.
# Drivers participating in these are the ones who will start the race.
_AUTHORITATIVE_SESSIONS = (
    "Race",
    "Sprint",
    "Qualifying",
    "Sprint Qualifying",
    "Sprint Shootout",
)

# Practice sessions — these frequently feature reserve/test/rookie drivers
# (e.g. Jak Crawford subbing for Alonso in FP1) who are NOT the race roster.
# We only trust them as a fallback when:
#   1. No authoritative session has any data at all, AND
#   2. Every driver in the session is a registered race driver (i.e. the
#      reserve filter against the Jolpica entry list removes nobody, OR the
#      entry list is unavailable so we can't distinguish).
# Neither of those conditions is a driver-count threshold.
_PRACTICE_SESSIONS = ("FP3", "FP2", "FP1")


def _clean_str(val) -> str:
    """Return a safe, stripped string for a field value.

    Handles None, NaN (float), pandas NA, and empty strings uniformly so that
    downstream rendering never shows placeholder junk like ``??? nan``.
    """
    if val is None:
        return ""
    try:
        # pandas/numpy NaN is a float that doesn't equal itself
        if isinstance(val, float) and val != val:  # NaN check
            return ""
    except Exception:
        pass
    s = str(val).strip()
    if s.lower() in ("nan", "none", "nat", "<na>"):
        return ""
    return s


def _derive_code(code: str, family_name: str, given_name: str) -> str:
    """Return a sensible 3-letter code for a driver.

    Guarantees a non-empty, uppercase, 3-char code when any name data is
    available, preventing the ``???`` placeholder from showing for real
    drivers in the UI (predict.py:1784 and prediction_manager.py:782).
    """
    c = _clean_str(code).upper()
    if c:
        return c[:3]
    fn = _clean_str(family_name)
    if fn:
        # Strip non-alphabetic (hyphens, spaces, accents) before taking prefix
        alpha = "".join(ch for ch in fn if ch.isalpha())
        if len(alpha) >= 3:
            return alpha[:3].upper()
        if alpha:
            return alpha.upper().ljust(3, "X")
    gn = _clean_str(given_name)
    if gn:
        alpha = "".join(ch for ch in gn if ch.isalpha())
        if len(alpha) >= 3:
            return alpha[:3].upper()
    return ""


def _entries_from_results(results: List[Dict]) -> List[Dict]:
    """Shape results rows into a deduplicated entry list.

    Also normalises optional string fields so downstream code never sees
    ``NaN``/``None`` bleeding into the displayed roster.
    """
    entries: List[Dict] = []
    for r in results or []:
        drv = r.get("Driver", {}) if isinstance(r.get("Driver"), dict) else {}
        cons = r.get("Constructor", {}) if isinstance(r.get("Constructor"), dict) else {}
        given = _clean_str(drv.get("givenName"))
        family = _clean_str(drv.get("familyName"))
        entries.append({
            "driverId": _clean_str(drv.get("driverId")) or None,
            "code": _derive_code(drv.get("code"), family, given),
            "givenName": given,
            "familyName": family,
            "permanentNumber": _clean_str(drv.get("permanentNumber")) or None,
            "constructorId": _clean_str(cons.get("constructorId")) or None,
            "constructorName": _clean_str(cons.get("name")),
        })
    # Deduplicate by driverId, preserving first
    seen = set()
    deduped = []
    for e in entries:
        did = e.get("driverId")
        if did and did not in seen:
            seen.add(did)
            deduped.append(e)
    return deduped


def _same_round_known_roster(jc: JolpicaClient, season: str, rnd: str) -> List[Dict]:
    """
    Try to get the event roster from the same round, in order:
      race -> sprint -> qualifying.
    """
    try:
        r = jc.get_race_results(season, rnd)
        if r:
            return _entries_from_results(r)
    except Exception:
        pass
    try:
        s = jc.get_sprint_results(season, rnd)
        if s:
            return _entries_from_results(s)
    except Exception:
        pass
    try:
        q = jc.get_qualifying_results(season, rnd)
        if q:
            return _entries_from_results(q)
    except Exception:
        pass
    return []


def _latest_completed_round_in_season(jc: JolpicaClient, season: str) -> Optional[str]:
    """Return latest round number in a season that has results (race preferred, else qualifying)."""
    try:
        races = jc.get_season_schedule(season)
        for r in reversed(races):
            if not r.get("round"):
                continue
            rnd = str(r.get("round"))
            if jc.get_race_results(season, rnd):
                return rnd
            if jc.get_qualifying_results(season, rnd):
                return rnd
    except Exception:
        pass
    return None


def _previous_completed_round_in_season(jc: JolpicaClient, season: str, upto_round: str) -> Optional[str]:
    """Return the most recent completed round strictly before upto_round (race preferred, else qualifying)."""
    try:
        r_cur = int(upto_round)
    except Exception:
        return None
    for r in range(r_cur - 1, 0, -1):
        r_str = str(r)
        try:
            if jc.get_race_results(season, r_str):
                return r_str
            if jc.get_qualifying_results(season, r_str):
                return r_str
        except Exception:
            continue
    return None


def _previous_completed_event_global(jc: JolpicaClient, season: int, upto_round: Optional[int]) -> Optional[Tuple[str, str]]:
    """
    Find the immediately previous completed event going backwards:
      - If upto_round is given: search rounds < upto_round within this season.
      - Otherwise: latest completed round in this season.
      - If none: move to previous seasons, latest completed round.
    Returns (season, round) or None.
    """
    # Optimization: Check global "latest" pointer to skip scanning future seasons/rounds
    try:
        last_s_str, last_r_str = jc.get_latest_season_and_round()
        last_s, last_r = int(last_s_str), int(last_r_str)

        # Case 1: Requested season is in the future relative to latest completed event
        if season > last_s:
            return last_s_str, last_r_str

        # Case 2: Requested season IS the latest season
        if season == last_s:
            # If we want the latest in this season, or if our cutoff is beyond the latest
            if upto_round is None or upto_round > last_r:
                return last_s_str, last_r_str
            # If upto_round <= last_r, we must search backwards as usual (might be gaps?)
    except Exception as e:
        logger.info(f"[roster] Optimization failed: {e}; falling back to full scan")

    # First, within the same season
    if upto_round is not None:
        prev_in_season = _previous_completed_round_in_season(jc, str(season), str(upto_round))
        if prev_in_season:
            return str(season), prev_in_season
    latest_in_season = _latest_completed_round_in_season(jc, str(season))
    if latest_in_season:
        return str(season), latest_in_season

    # Then step into previous seasons
    for yr in range(season - 1, 1949, -1):
        latest_prev = _latest_completed_round_in_season(jc, str(yr))
        if latest_prev:
            return str(yr), latest_prev
    return None


def _roster_from_round(jc: JolpicaClient, season: str, rnd: str) -> List[Dict]:
    """
    Build roster from a specific (season, round), preferring race results, then qualifying.
    """
    try:
        r = jc.get_race_results(season, rnd)
        if r:
            return _entries_from_results(r)
    except Exception:
        pass
    try:
        q = jc.get_qualifying_results(season, rnd)
        if q:
            return _entries_from_results(q)
    except Exception:
        pass
    return []


def _build_canonical_mapping_from_entries(raw_entries: List[Dict]) -> Dict[str, 'Any']:  # noqa: F821
    """Build canonical-id lookup tables directly from a pre-fetched entry list.

    Factored out of :func:`_get_canonical_mapping` so that callers who have
    already fetched the entry list don't pay the network cost twice.

    Returns a dict with keys:
      - ``drivers``:      {key(number/code/name) -> driverId}
      - ``constructors``: {name -> constructorId}
      - ``race_driver_ids``: set of canonical driverIds registered as race
        drivers for the season.
    """
    if not raw_entries:
        return {}

    driver_map: Dict[str, str] = {}
    constructor_map: Dict[str, str] = {}
    race_driver_ids: Set[str] = set()

    for r in raw_entries:
        drv = r.get("Driver", {}) or {}
        cons = r.get("Constructor", {}) or {}

        d_id = drv.get("driverId")
        c_id = cons.get("constructorId")
        c_name = cons.get("name")

        if d_id:
            race_driver_ids.add(d_id)

            # Map by Number
            num = drv.get("permanentNumber")
            if num:
                driver_map[str(num)] = d_id

            # Map by Code (Abbreviation)
            code = drv.get("code")
            if code:
                driver_map[code.upper()] = d_id

            # Map by Name (Normalised)
            gn = drv.get("givenName", "") or ""
            fn = drv.get("familyName", "") or ""
            if gn and fn:
                name_key = f"{gn} {fn}".lower().strip()
                driver_map[name_key] = d_id

        if c_id and c_name:
            constructor_map[c_name.lower().strip()] = c_id

    return {
        "drivers": driver_map,
        "constructors": constructor_map,
        "race_driver_ids": race_driver_ids,
    }


def _get_canonical_mapping(jc: JolpicaClient, season: str) -> Dict[str, 'Any']:  # noqa: F821
    """Fetch season entry list and build canonical-id lookup tables."""
    try:
        raw_entries = jc.get_season_entry_list(season)
    except Exception as e:
        logger.warning(f"[roster] Failed to fetch canonical entry list: {e}")
        return {}
    try:
        return _build_canonical_mapping_from_entries(raw_entries or [])
    except Exception as e:
        logger.warning(f"[roster] Failed to build canonical mapping: {e}")
        return {}


def _roster_entries_from_fastf1_results(
    results, mapping: Optional[Dict]
) -> List[Dict]:
    """Convert a FastF1 results DataFrame into normalised roster entries.

    - Canonicalises driverId using the mapping when possible.
    - Falls back to ``abbr.lower()`` (legacy behaviour) when no mapping hit,
      preserving backwards compatibility for seasons/drivers missing from
      the Jolpica entry list (e.g. brand-new rookies).
    - Drops rows with no identifying information at all.
    """
    d_map = (mapping or {}).get("drivers", {})
    c_map = (mapping or {}).get("constructors", {})

    entries: List[Dict] = []
    for _, r in results.iterrows():
        abbr = _clean_str(r.get("Abbreviation"))
        num = _clean_str(r.get("DriverNumber"))
        gn = _clean_str(r.get("FirstName"))
        fn = _clean_str(r.get("LastName"))
        tname = _clean_str(r.get("TeamName"))

        # Try to canonicalise driverId in priority order: abbr > number > name
        did: Optional[str] = None
        if abbr and abbr.upper() in d_map:
            did = d_map[abbr.upper()]
        elif num and num in d_map:
            did = d_map[num]
        elif gn and fn:
            name_key = f"{gn} {fn}".lower().strip()
            if name_key in d_map:
                did = d_map[name_key]

        # Legacy fallback: synthesise a stable id from abbr. This keeps older
        # seasons / brand-new rookies working even when Jolpica has no mapping.
        # We do NOT drop the entry here — that would hide rookies. The tier
        # gating in _roster_from_fastf1 is responsible for rejecting whole
        # sessions that are clearly wrong.
        if not did:
            if abbr:
                did = abbr.lower()
            elif num:
                did = f"driver_{num}"
            else:
                # No abbr, no number, no mapped name: genuinely malformed row.
                logger.debug(
                    "[roster] Skipping FastF1 row with no identifying info: %r", r
                )
                continue

        # Constructor canonicalisation with graceful fallback
        cid = None
        if tname and tname.lower().strip() in c_map:
            cid = c_map[tname.lower().strip()]
        if not cid and tname:
            cid = tname.lower().replace(" ", "_")

        code = _derive_code(abbr, fn, gn)

        # Require a human-readable label before accepting the entry. This is
        # the guard that stops "??? " rows from ever reaching the grid.
        if not code and not (gn or fn):
            logger.debug("[roster] Skipping unnamed FastF1 row (did=%s)", did)
            continue

        entries.append({
            "driverId": did,
            "code": code,
            "givenName": gn,
            "familyName": fn,
            "permanentNumber": num or None,
            "constructorId": cid,
            "constructorName": tname,
        })

    # Deduplicate by driverId, preserving first occurrence
    seen: Set[str] = set()
    deduped: List[Dict] = []
    for e in entries:
        did = e.get("driverId")
        if did and did not in seen:
            seen.add(did)
            deduped.append(e)
    return deduped


def _filter_reserves(
    entries: List[Dict], mapping: Optional[Dict]
) -> List[Dict]:
    """Remove entries whose canonical driverId is NOT in the season race roster.

    A driver that appears in a FastF1 session (often a practice session) but
    is *not* registered in Jolpica's season entry list is almost certainly a
    reserve / test / rookie FP1 stand-in (e.g. Jak Crawford subbing for
    Alonso). Such drivers should not pollute the predicted race grid.

    Only applied when we actually have a trustworthy ``race_driver_ids`` set —
    if Jolpica hasn't published the entry list yet, we can't distinguish
    reserves from the real roster and must keep everyone.
    """
    race_ids: Set[str] = (mapping or {}).get("race_driver_ids") or set()
    if not race_ids:
        return entries

    kept, dropped = [], []
    for e in entries:
        if e.get("driverId") in race_ids:
            kept.append(e)
        else:
            dropped.append(e)

    if dropped:
        logger.info(
            "[roster] Filtered %d non-race-roster driver(s) from FastF1 data: %s",
            len(dropped),
            ", ".join(sorted({d.get("code") or d.get("driverId") or "?" for d in dropped})),
        )
    return kept


def _try_load_fastf1_session(ev, sname: str, remaining: float):
    """Attempt to load a single FastF1 session with a wall-clock timeout.

    Returns the session's ``results`` DataFrame on success, or ``None`` on
    timeout / any failure. Raises ``FuturesTimeoutError`` only — all other
    exceptions are swallowed and converted to ``None``.
    """
    def _load_session():
        sess = ev.get_session(sname)
        sess.load(telemetry=False, laps=False, weather=False, messages=False)
        return sess

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_load_session)
        sess = future.result(timeout=remaining)

    if sess.results is None or sess.results.empty:
        return None
    return sess.results


def _roster_from_fastf1(season: int, rnd: int, mapping: Optional[Dict] = None) -> List[Dict]:
    """Attempt to fetch roster from FastF1 (live timing).

    Strategy:
      1. Try *authoritative* sessions (Race → Sprint → Qualifying → SQ → SS)
         in order. The first one with any drivers is the truth, regardless
         of count — cars can legitimately be disqualified or not classified,
         so a 14-driver classified race is just as authoritative as a
         22-driver one.
      2. Only if no authoritative session returned anything, fall back to
         *practice* sessions (FP3 → FP2 → FP1). Practice rosters are
         filtered against the Jolpica race-driver list because they
         routinely contain FP1 reservists (rookies, test drivers) who are
         NOT the actual race roster. A practice session is only trusted
         when, after reserve filtering, it still contains drivers and every
         one of them is a registered race driver (i.e. nothing was dropped,
         or the entry list was unavailable so filtering was skipped).

    Applies a wall-clock timeout (_FASTF1_ROSTER_TIMEOUT) so that unresponsive
    F1 live-timing servers cannot block the entire prediction pipeline.
    """
    t0 = time.monotonic()

    def _budget_remaining() -> float:
        return _FASTF1_ROSTER_TIMEOUT - (time.monotonic() - t0)

    try:
        # Import here to avoid circular dependencies if not already imported
        from .data import fastf1_backend as ff1

        ev = ff1.get_event(season, rnd)
        if ev is None:
            return []

        # --- Pass 1: Authoritative sessions ---------------------------------
        # Any authoritative session that produced any entries at all is truth:
        # DSQ/DNS/DNF/not-classified drivers are legitimately absent from
        # the classification, and we must not second-guess that by demanding
        # a minimum count.
        for sname in _AUTHORITATIVE_SESSIONS:
            remaining = _budget_remaining()
            if remaining <= 0:
                logger.info(
                    "[roster] FastF1 timeout exhausted in authoritative pass "
                    "(after %ss)", _FASTF1_ROSTER_TIMEOUT,
                )
                break
            try:
                results = _try_load_fastf1_session(ev, sname, remaining)
            except FuturesTimeoutError:
                logger.info(
                    "[roster] FastF1 session %s timed out after %.1fs",
                    sname, remaining,
                )
                # Authoritative sessions are attempted in order; if one hangs
                # the others likely will too, but keep trying the remaining
                # budget rather than abandoning the cascade.
                continue
            except Exception as exc:
                logger.debug(
                    "[roster] FastF1 session %s unavailable: %s", sname, exc
                )
                continue

            if results is None:
                continue

            entries = _roster_entries_from_fastf1_results(results, mapping)
            if not entries:
                continue

            logger.info(
                "[roster] Using FastF1 %s as authoritative roster (%d drivers)",
                sname, len(entries),
            )
            return entries

        # --- Pass 2: Practice sessions (reserve-filtered) -------------------
        # Practice rosters can contain FP1 reservists. We trust a practice
        # session only if EVERY driver in it is a registered race driver —
        # i.e. the reserve filter drops nobody (or, when the entry list is
        # unavailable, we have no way to tell reserves from real drivers and
        # must accept the data as-is). Any session where reserves are present
        # and the entry list is available is rejected outright, because we
        # can't tell whether the remaining filtered roster faithfully
        # represents the race grid or is just the subset of reservists who
        # happened to have a Jolpica entry from a prior year.
        race_ids: Set[str] = (mapping or {}).get("race_driver_ids") or set()
        have_entry_list = bool(race_ids)

        for sname in _PRACTICE_SESSIONS:
            remaining = _budget_remaining()
            if remaining <= 0:
                break
            try:
                results = _try_load_fastf1_session(ev, sname, remaining)
            except FuturesTimeoutError:
                logger.info(
                    "[roster] FastF1 practice session %s timed out", sname
                )
                continue
            except Exception:
                continue

            if results is None:
                continue

            raw_entries = _roster_entries_from_fastf1_results(results, mapping)
            if not raw_entries:
                continue

            if have_entry_list:
                filtered = _filter_reserves(raw_entries, mapping)
                if len(filtered) != len(raw_entries):
                    # At least one reserve was present — session is not a
                    # faithful picture of the race roster, skip it.
                    logger.info(
                        "[roster] FastF1 %s contained %d reserve(s); not "
                        "trusting this session for the race roster",
                        sname, len(raw_entries) - len(filtered),
                    )
                    continue
                logger.info(
                    "[roster] Using FastF1 %s — all %d drivers are registered "
                    "race drivers", sname, len(filtered),
                )
                return filtered

            # No entry list available; can't distinguish reserves, but also
            # have nothing better to go on. Accept the session verbatim.
            logger.info(
                "[roster] Using FastF1 %s (no entry list for reserve check, "
                "%d drivers)", sname, len(raw_entries),
            )
            return raw_entries

    except Exception as e:
        logger.warning(f"[roster] FastF1 fetch failed: {e}")
    return []


def _is_future_event(event_dt: Optional[datetime], now_dt: datetime) -> bool:
    """Return True when the event is in the future (or we simply don't know).

    We take a conservative stance: if ``event_dt`` is not supplied, callers
    typically predict upcoming events, so we default to ``True``. A known-past
    event returns ``False``.
    """
    if event_dt is None:
        return True
    return event_dt >= now_dt


def derive_roster(
    jc: JolpicaClient,
    season: str,
    rnd: str,
    event_dt: Optional[datetime] = None,
    now_dt: Optional[datetime] = None,
) -> List[Dict]:
    """
    Derive roster using a prioritised cascade.

    For *past* events (event_dt in the past):
      1. Jolpica known results for this round (race/sprint/qualifying).
      2. FastF1 live-timing session results.
      3. Season entry list.
      4. Previous completed event roster.

    For *upcoming* events (event_dt in the future, or unknown):
      1. Jolpica known results for this round (in case results were posted
         but event_dt metadata is stale).
      2. Official Jolpica season entry list — the authoritative registered
         race drivers. Preferred BEFORE FastF1 because FastF1 practice
         sessions often contain reserve/test drivers (e.g. Jak Crawford
         subbing for Alonso in FP1) that are not part of the real race roster.
      3. FastF1 live-timing session results.
      4. Previous completed event roster.
    """
    # Normalise times
    if now_dt is None:
        now_dt = datetime.now(timezone.utc)
    if event_dt is not None and event_dt.tzinfo is None:
        event_dt = event_dt.replace(tzinfo=timezone.utc)

    s_int = int(season)
    r_int = int(rnd)
    future = _is_future_event(event_dt, now_dt)

    # 1. Jolpica known results for this round — always first because if a
    #    session has actually run, its classification is the ground truth.
    #    Any non-empty result is accepted: cars can be disqualified or not
    #    classified, so a 14-driver result set after a chaotic race is as
    #    authoritative as a 22-driver one. We must not apply a minimum-count
    #    gate here, or we'd reject legitimate results and cascade into stale
    #    fallbacks.
    same = _same_round_known_roster(jc, season, rnd)
    if same:
        return same

    # --- Previous completed event (this season first, then earlier seasons) --
    # The roster for an upcoming round is, in practice, the roster of the
    # previous completed round of the same season — the drivers who raced
    # last weekend are the drivers who will race this weekend. Mid-season
    # swaps only enter the picture once the new driver has actually been
    # classified in a session; before that, nobody — including Jolpica —
    # can know about them.
    #
    # NB: we deliberately DO NOT consult Jolpica's ``/{season}/drivers.json``
    # endpoint here. That endpoint is a UNION of every driverId seen in
    # *any* session this season, including practice. It contains FP1
    # stand-ins (e.g. Jak Crawford subbing for Alonso) who are not race
    # drivers, and treating it as an entry list produces 23-driver grids
    # with reservists on them. There is no public Jolpica/Ergast endpoint
    # for the true season entry list, so the correct answer is to ignore
    # that superset entirely and fall back to the previous completed
    # round, whose classification is by construction the real roster.
    try:
        prev = _previous_completed_event_global(jc, s_int, r_int)
    except Exception:
        prev = _previous_completed_event_global(jc, s_int, None)

    prev_roster: List[Dict] = []
    if prev:
        ps, pr = prev
        prev_roster = _roster_from_round(jc, ps, pr)

    # Build the canonical mapping from the previous roster so ``race_driver_ids``
    # (used by _filter_reserves for FastF1 practice-session fallback) reflects
    # the real race roster rather than the Jolpica-drivers.json superset.
    def _to_mapping_rows(entries: List[Dict]) -> List[Dict]:
        """Shape normalised entries back into the ``{"Driver": ..., "Constructor": ...}``
        dicts that _build_canonical_mapping_from_entries expects."""
        out: List[Dict] = []
        for e in entries:
            out.append({
                "Driver": {
                    "driverId": e.get("driverId"),
                    "code": e.get("code"),
                    "givenName": e.get("givenName"),
                    "familyName": e.get("familyName"),
                    "permanentNumber": e.get("permanentNumber"),
                },
                "Constructor": {
                    "constructorId": e.get("constructorId"),
                    "name": e.get("constructorName"),
                },
            })
        return out

    mapping = _build_canonical_mapping_from_entries(_to_mapping_rows(prev_roster))

    def _fastf1() -> List[Dict]:
        return _roster_from_fastf1(s_int, r_int, mapping=mapping)

    # --- Future events --------------------------------------------------
    # Preferred order:
    #   1. Previous completed round in the same season (ground truth for
    #      the current roster, reservists excluded by construction).
    #   2. FastF1 live-timing session data (only useful very close to the
    #      event; authoritative sessions haven't happened yet but FP1
    #      reservists filtered via ``mapping.race_driver_ids``).
    #   3. FastF1 without a mapping (in case we have nothing else — best
    #      effort; the practice-session reserve filter degrades to a
    #      pass-through but authoritative sessions still win).
    if future:
        if prev_roster:
            logger.info(
                f"[roster] Using previous completed round as roster for "
                f"upcoming {season} R{rnd} ({prev[0]} R{prev[1]}, "
                f"{len(prev_roster)} drivers)"
            )
            return prev_roster
        ff1_roster = _fastf1()
        if ff1_roster:
            return ff1_roster
        return []

    # --- Past events ----------------------------------------------------
    # For a past event whose same-round results we couldn't fetch, FastF1
    # is more reliable than falling back to a neighbouring event, because
    # FastF1 reflects who actually participated (including any mid-season
    # substitutions that may not be present in another round).
    ff1_roster = _fastf1()
    if ff1_roster:
        return ff1_roster

    if prev_roster:
        logger.info(
            f"[roster] Falling back to roster from {prev[0]} R{prev[1]} "
            f"({len(prev_roster)} drivers)"
        )
        return prev_roster

    return []
