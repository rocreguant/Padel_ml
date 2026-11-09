"""CSV importers - the reliable way to load historical data, fixtures and odds.

matches.csv columns (header required):
    date, team_a_1, team_a_2, team_b_1, team_b_2, score, [winner], [tournament],
    [category], [round], [gender]
  - `score` is from team A's perspective, e.g. "6-4 3-6 7-5". Leave empty for an
    upcoming fixture.
  - `winner` (A/B) is optional when `score` is present.

players.csv columns:
    name, [gender], [country], [birth_date], [hand], [side], [height_cm],
    [ranking_position], [ranking_points], [ranking_date]

odds.csv columns (one row per match, decimal odds):
    date, team_a_1, team_a_2, team_b_1, team_b_2, odds_a, odds_b, [bookmaker], [captured_at]
"""

from __future__ import annotations

import logging
from datetime import date as _date
from pathlib import Path

import pandas as pd

from ..db import Database

log = logging.getLogger(__name__)


def _s(v) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    v = str(v).strip()
    return v or None


def import_matches(db: Database, path: str | Path, source: str = "csv") -> int:
    df = pd.read_csv(path, dtype=str).fillna("")
    n = 0
    for r in df.to_dict("records"):
        db.upsert_match(
            date=r["date"], team_a=(r["team_a_1"], r["team_a_2"]),
            team_b=(r["team_b_1"], r["team_b_2"]), score=_s(r.get("score")),
            winner=_s(r.get("winner")), tournament=_s(r.get("tournament")),
            category=_s(r.get("category")), round=_s(r.get("round")),
            gender=_s(r.get("gender")), source=source,
        )
        n += 1
    return n


def import_players(db: Database, path: str | Path) -> int:
    df = pd.read_csv(path, dtype=str).fillna("")
    n = 0
    for r in df.to_dict("records"):
        attrs = {k: _s(r.get(k)) for k in
                 ("gender", "country", "birth_date", "hand", "side", "height_cm")}
        pid = db.upsert_player(r["name"], source="csv", **attrs)
        if _s(r.get("ranking_position")) or _s(r.get("ranking_points")):
            pos = _s(r.get("ranking_position"))
            pts = _s(r.get("ranking_points"))
            db.add_ranking(pid, _s(r.get("ranking_date")) or _date.today().isoformat(),
                           int(pos) if pos else None, float(pts) if pts else None, "csv")
        n += 1
    return n


def import_odds(db: Database, path: str | Path, default_bookmaker: str = "bet365",
                create_missing: bool = True) -> tuple[int, int]:
    """Attach odds to fixtures. Returns (imported, skipped)."""
    df = pd.read_csv(path, dtype=str).fillna("")
    ok = skipped = 0
    for r in df.to_dict("records"):
        ta, tb = (r["team_a_1"], r["team_a_2"]), (r["team_b_1"], r["team_b_2"])
        if not _s(r.get("odds_a")) or not _s(r.get("odds_b")):
            skipped += 1  # template row not filled in yet
            continue
        odds_a = float(str(r["odds_a"]).replace(",", "."))
        odds_b = float(str(r["odds_b"]).replace(",", "."))
        if odds_a <= 1 or odds_b <= 1:
            raise ValueError(f"decimal odds must be > 1 (got {odds_a}, {odds_b}) for {ta} vs {tb}")
        d = _s(r.get("date"))
        # with a date we can also match completed games (historical odds for backtests)
        found = db.find_match(ta, tb, date=d, upcoming_only=d is None)
        if found is None and create_missing:
            mid = db.upsert_match(date=r["date"], team_a=ta, team_b=tb,
                                  tournament=_s(r.get("tournament")), source="odds-csv")
            found = (mid, False)
        if found is None:
            log.warning("no fixture for %s vs %s - skipped", ta, tb)
            skipped += 1
            continue
        mid, swapped = found
        if swapped:
            odds_a, odds_b = odds_b, odds_a
        db.add_odds(mid, _s(r.get("bookmaker")) or default_bookmaker, odds_a, odds_b,
                    _s(r.get("captured_at")))
        ok += 1
    return ok, skipped
