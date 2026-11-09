"""Padel API (https://padelapi.org) client - the main source of historical matches.

Padel API is a public REST API for professional padel: Premier Padel and FIP
(complete draws from 2023), World Padel Tour and Padel Pro Tour (finals only for
2006-2022), plus weekly rankings. Its terms let you store responses in your own
systems; reselling or rebuilding an equivalent service is not allowed.

Plans (as published on padelapi.org, Nov 2026):
  Free  - 50k requests/month, 10/min, current rankings, scores of the last 6 months
  Plus  - EUR 19/month, full historical archive, 60/min

Get a token at https://padelapi.org/register and export it:
    export PADEL_API_TOKEN=...

Usage:
    padel-ml sync-matches --since 2023-01-01
    padel-ml sync-rankings
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import date
from typing import Any, Iterator

import requests

from ..config import get_settings
from ..db import Database

log = logging.getLogger(__name__)

BASE_URL = "https://padelapi.org/api"
PREMIER_LEVELS = ("major", "p1", "p2", "finals")
ALL_LEVELS = PREMIER_LEVELS + (
    "fip_platinum", "fip_gold", "fip_silver", "fip_other",
    "wpt_master", "wpt_1000", "wpt_500", "wpt_final",
)
ROUND_NAMES = {0: "Group", 1: "F", 2: "SF", 4: "QF", 8: "R16", 16: "R32", 32: "R64"}
PLAYED = {"finished", "ended", "retired"}
NOT_PLAYED = {"walkover", "bye", "not_played"}
HIDDEN = "hidden_free_plan"
GENDER = {"men": "M", "women": "F"}


class PadelApiError(RuntimeError):
    pass


class PadelApiClient:
    def __init__(self, token: str | None = None, base_url: str = BASE_URL,
                 requests_per_minute: int | None = None,
                 session: requests.Session | None = None):
        token = token or os.environ.get("PADEL_API_TOKEN")
        if not token:
            raise PadelApiError(
                "PADEL_API_TOKEN is not set. Create a free token at "
                "https://padelapi.org/register and `export PADEL_API_TOKEN=...`")
        self.base_url = base_url.rstrip("/")
        rpm = requests_per_minute or int(os.environ.get("PADEL_API_RPM", "10"))
        self.min_interval = 60.0 / max(rpm, 1)
        self.session = session or requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": get_settings().user_agent,
        })
        self._last = 0.0
        self.requests_made = 0

    def get(self, path: str, params: dict[str, Any] | None = None,
            max_retries: int = 5) -> dict:
        url = path if path.startswith("http") else f"{self.base_url}/{path.lstrip('/')}"
        for attempt in range(max_retries):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            r = self.session.get(url, params=params, timeout=30)
            self.requests_made += 1
            if r.status_code == 429:
                retry = float(r.headers.get("Retry-After", 60))
                log.warning("rate limited by Padel API; sleeping %.0fs", retry)
                time.sleep(retry)
                continue
            if r.status_code == 401:
                raise PadelApiError("Padel API rejected the token (401) - check PADEL_API_TOKEN")
            if r.status_code == 402:
                raise PadelApiError(f"Padel API: paid plan required for {url} ({r.text[:200]})")
            if r.status_code >= 500:
                time.sleep(2 ** (attempt + 1))
                continue
            if r.status_code >= 400:
                raise PadelApiError(f"Padel API {r.status_code} for {url}: {r.text[:300]}")
            remaining = r.headers.get("X-RateLimit-Remaining")
            if remaining is not None and remaining.isdigit() and int(remaining) < 3:
                log.info("Padel API quota nearly used up (%s left in window)", remaining)
            return r.json()
        raise PadelApiError(f"Padel API: giving up on {url} after {max_retries} attempts")

    def paginate(self, path: str, params: dict[str, Any] | None = None) -> Iterator[dict]:
        params = {**(params or {}), "per_page": 50, "page": 1}
        while True:
            body = self.get(path, params)
            yield from body.get("data", [])
            meta = body.get("meta") or {}
            if not body.get("links", {}).get("next") or \
                    meta.get("current_page", 1) >= meta.get("last_page", 1):
                return
            params["page"] += 1

    # ------------------------------------------------------------ endpoints
    def tournaments(self, after_date: str | None = None, before_date: str | None = None,
                    levels: tuple[str, ...] = PREMIER_LEVELS) -> Iterator[dict]:
        params = {"level": ",".join(levels), "sort_by": "start_date", "order_by": "asc"}
        if after_date:
            params["after_date"] = after_date
        if before_date:
            params["before_date"] = before_date
        return self.paginate("/tournaments", params)

    def tournament_matches(self, tournament_id: int, updated_after: str | None = None
                           ) -> Iterator[dict]:
        params = {"updated_after": updated_after} if updated_after else {}
        return self.paginate(f"/tournaments/{tournament_id}/matches", params)

    def rankings(self, category: str = "men", ranking_type: str = "official",
                 on: str | None = None) -> Iterator[dict]:
        params = {"category": category, "type": ranking_type}
        if on:
            params["date"] = on
        return self.paginate("/rankings", params)


# ---------------------------------------------------------------- mapping
def _games(v: Any) -> str:
    """'7(5)' -> '7'."""
    m = re.match(r"\s*(\d+)", str(v))
    return m.group(1) if m else "0"


def score_to_text(score: Any) -> str | None:
    """[{team_1:'6',team_2:'4'}, {team_1:'7(5)',team_2:'6'}] -> '6-4 7-6'."""
    if not score or score == HIDDEN or not isinstance(score, list):
        return None
    return " ".join(f"{_games(s.get('team_1'))}-{_games(s.get('team_2'))}" for s in score)


def _pair(players: Any) -> list[dict]:
    if isinstance(players, dict):  # tolerate {"0": {...}, "1": {...}}
        players = list(players.values())
    return [p for p in (players or []) if isinstance(p, dict) and p.get("name")]


def store_match(db: Database, m: dict, tournament: dict) -> str:
    """Store one API match. Returns 'played', 'fixture' or the reason it was skipped."""
    status = m.get("status")
    if status in NOT_PLAYED:
        return "not_played"
    t1, t2 = _pair(m.get("players", {}).get("team_1")), _pair(m.get("players", {}).get("team_2"))
    if len(t1) != 2 or len(t2) != 2:
        return "teams_unknown"
    gender = GENDER.get(m.get("category"))
    for p in (*t1, *t2):
        db.upsert_player(p["name"], gender=gender, country=p.get("nationality"),
                         side=p.get("side"), source="padelapi", source_id=str(p.get("id")))
    score, winner = None, None
    if status in PLAYED:
        if m.get("score") == HIDDEN or m.get("winner") == HIDDEN:
            return "hidden_free_plan"
        score = score_to_text(m.get("score"))
        winner = {"team_1": "A", "team_2": "B"}.get(m.get("winner"))
        if winner is None:
            return "no_winner"
        if status == "retired" and score:
            score += " ret"
    elif status != "scheduled" and status != "live":
        return f"status_{status}"
    played_at = m.get("played_at") or tournament.get("start_date")
    if not played_at:
        return "no_date"
    if winner is None and played_at < date.today().isoformat() and status != "live":
        return "stale_fixture"
    db.upsert_match(
        date=played_at, team_a=(t1[0]["name"], t1[1]["name"]),
        team_b=(t2[0]["name"], t2[1]["name"]), score=score, winner=winner,
        tournament=tournament.get("name"), category=tournament.get("level"),
        round=ROUND_NAMES.get(m.get("round"), str(m.get("round"))), gender=gender,
        source="padelapi", source_key=f"padelapi:{m['id']}",
    )
    return "played" if winner else "fixture"


def sync_matches(db: Database, client: PadelApiClient, since: str | None = None,
                 until: str | None = None, levels: tuple[str, ...] = PREMIER_LEVELS
                 ) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in client.tournaments(since, until, levels):
        if t.get("status") in ("cancelled", "postponed") or t.get("format") == "teams":
            continue
        n_before = sum(counts.values())
        for m in client.tournament_matches(t["id"]):
            outcome = store_match(db, m, t)
            counts[outcome] = counts.get(outcome, 0) + 1
        log.info("%s (%s, %s): %d matches", t.get("name"), t.get("level"),
                 t.get("start_date"), sum(counts.values()) - n_before)
    counts["api_requests"] = client.requests_made
    return counts


def sync_rankings(db: Database, client: PadelApiClient, on: str | None = None,
                  categories: tuple[str, ...] = ("men", "women")) -> int:
    n = 0
    for cat in categories:
        for r in client.rankings(cat, "official", on):
            if not r.get("name"):
                continue
            pid = db.upsert_player(r["name"], gender=GENDER.get(cat),
                                   country=r.get("nationality"), source="padelapi",
                                   source_id=str(r.get("id")))
            db.add_ranking(pid, (r.get("date") or on or date.today().isoformat())[:10],
                           r.get("ranking"), r.get("points"), "padelapi")
            n += 1
    return n
