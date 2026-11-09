"""Bookmaker odds sources.

bet365 does not offer a public API, renders its prices with JavaScript behind
anti-bot protection, and its terms of use forbid automated collection. This project
therefore does NOT scrape bet365.es. Instead:

1. `padel-ml odds-template` writes a CSV with every upcoming fixture.
2. You copy the bet365.es prices into the `odds_a` / `odds_b` columns
   (takes a minute or two per tournament day).
3. `padel-ml import-odds odds.csv` stores them and `padel-ml value-bets` compares.

If you get legitimate access to an odds feed (a licensed data provider or an
odds-comparison API that covers padel), implement `OddsProvider.fetch()` returning
`OddsQuote`s and pass it to `store_quotes` - nothing else needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..db import Database
from ..scrapers.csv_import import import_odds


@dataclass(frozen=True)
class OddsQuote:
    date: str
    team_a: tuple[str, str]
    team_b: tuple[str, str]
    odds_a: float
    odds_b: float
    bookmaker: str = "bet365"
    captured_at: str | None = None


class OddsProvider(ABC):
    @abstractmethod
    def fetch(self) -> list[OddsQuote]:
        ...

    def store(self, db: Database) -> tuple[int, int]:
        ok = skipped = 0
        for q in self.fetch():
            found = db.find_match(q.team_a, q.team_b, date=q.date)
            if found is None:
                mid = db.upsert_match(date=q.date, team_a=q.team_a, team_b=q.team_b,
                                      source=f"odds-{q.bookmaker}")
                found = (mid, False)
            mid, swapped = found
            oa, ob = (q.odds_b, q.odds_a) if swapped else (q.odds_a, q.odds_b)
            db.add_odds(mid, q.bookmaker, oa, ob, q.captured_at)
            ok += 1
        return ok, skipped


class CsvOddsProvider(OddsProvider):
    """Odds typed in by hand from bet365.es (or any bookmaker) into a CSV."""

    def __init__(self, path: str | Path, bookmaker: str = "bet365"):
        self.path = Path(path)
        self.bookmaker = bookmaker

    def fetch(self) -> list[OddsQuote]:
        df = pd.read_csv(self.path, dtype=str).fillna("")
        out = []
        for r in df.to_dict("records"):
            if not r.get("odds_a") or not r.get("odds_b"):
                continue
            out.append(OddsQuote(
                r["date"], (r["team_a_1"], r["team_a_2"]), (r["team_b_1"], r["team_b_2"]),
                float(r["odds_a"].replace(",", ".")), float(r["odds_b"].replace(",", ".")),
                r.get("bookmaker") or self.bookmaker, r.get("captured_at") or None,
            ))
        return out

    def store(self, db: Database) -> tuple[int, int]:  # reuse the CSV importer
        return import_odds(db, self.path, default_bookmaker=self.bookmaker)


def export_odds_template(db: Database, path: str | Path) -> int:
    """Write upcoming fixtures to a CSV ready to be filled with bet365 odds."""
    q = """
    SELECT m.date, m.tournament, m.round,
           pa1.name AS team_a_1, pa2.name AS team_a_2,
           pb1.name AS team_b_1, pb2.name AS team_b_2
    FROM matches m
    JOIN players pa1 ON pa1.id=m.a1 JOIN players pa2 ON pa2.id=m.a2
    JOIN players pb1 ON pb1.id=m.b1 JOIN players pb2 ON pb2.id=m.b2
    WHERE m.winner IS NULL ORDER BY m.date, m.id
    """
    df = pd.read_sql_query(q, db.conn)
    df["odds_a"] = ""
    df["odds_b"] = ""
    df["bookmaker"] = "bet365"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return len(df)
