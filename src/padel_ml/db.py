"""SQLite storage: players, rankings, matches (played + upcoming), odds, predictions."""

from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd

from .utils import ParsedScore, normalize_name, parse_score

SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
    id              INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    normalized_name TEXT NOT NULL UNIQUE,
    gender          TEXT,              -- 'M' / 'F'
    country         TEXT,
    birth_date      TEXT,
    hand            TEXT,              -- 'R' / 'L'
    side            TEXT,              -- 'drive' / 'reves'
    height_cm       REAL,
    source          TEXT,
    source_id       TEXT,
    updated_at      TEXT
);

CREATE TABLE IF NOT EXISTS rankings (
    player_id   INTEGER NOT NULL REFERENCES players(id),
    date        TEXT NOT NULL,
    position    INTEGER,
    points      REAL,
    source      TEXT NOT NULL,
    PRIMARY KEY (player_id, date, source)
);

-- Completed matches have a winner; upcoming fixtures have winner NULL.
CREATE TABLE IF NOT EXISTS matches (
    id          INTEGER PRIMARY KEY,
    date        TEXT NOT NULL,
    tournament  TEXT,
    category    TEXT,                  -- Major, P1, P2, FIP Gold, ...
    round       TEXT,
    gender      TEXT,
    a1          INTEGER NOT NULL REFERENCES players(id),
    a2          INTEGER NOT NULL REFERENCES players(id),
    b1          INTEGER NOT NULL REFERENCES players(id),
    b2          INTEGER NOT NULL REFERENCES players(id),
    winner      TEXT CHECK (winner IN ('A','B') OR winner IS NULL),
    score       TEXT,
    sets_a      INTEGER, sets_b INTEGER,
    games_a     INTEGER, games_b INTEGER,
    retired     INTEGER DEFAULT 0,
    source      TEXT,
    source_key  TEXT UNIQUE
);
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(date);

CREATE TABLE IF NOT EXISTS odds (
    id          INTEGER PRIMARY KEY,
    match_id    INTEGER NOT NULL REFERENCES matches(id),
    bookmaker   TEXT NOT NULL,
    odds_a      REAL NOT NULL,
    odds_b      REAL NOT NULL,
    captured_at TEXT NOT NULL,
    UNIQUE (match_id, bookmaker, captured_at)
);

CREATE TABLE IF NOT EXISTS predictions (
    match_id    INTEGER NOT NULL REFERENCES matches(id),
    model       TEXT NOT NULL,
    prob_a      REAL NOT NULL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (match_id, model)
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self.conn:
            yield self.conn

    # ---------------------------------------------------------------- players
    def upsert_player(self, name: str, **attrs) -> int:
        """Insert or update a player identified by normalized name. Returns id."""
        key = normalize_name(name)
        if not key:
            raise ValueError(f"empty player name: {name!r}")
        attrs = {k: v for k, v in attrs.items() if v not in (None, "")}
        row = self.conn.execute("SELECT id FROM players WHERE normalized_name=?", (key,)).fetchone()
        with self.tx() as c:
            if row:
                if attrs:
                    sets = ", ".join(f"{k}=?" for k in attrs)
                    c.execute(
                        f"UPDATE players SET {sets}, updated_at=? WHERE id=?",
                        (*attrs.values(), now_iso(), row["id"]),
                    )
                return int(row["id"])
            cols = ["name", "normalized_name", "updated_at", *attrs]
            cur = c.execute(
                f"INSERT INTO players ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                (name.strip(), key, now_iso(), *attrs.values()),
            )
            return int(cur.lastrowid)

    def find_player(self, name: str) -> int | None:
        row = self.conn.execute(
            "SELECT id FROM players WHERE normalized_name=?", (normalize_name(name),)
        ).fetchone()
        return int(row["id"]) if row else None

    def add_ranking(self, player_id: int, date: str, position: int | None,
                    points: float | None, source: str) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO rankings VALUES (?,?,?,?,?)",
                (player_id, date, position, points, source),
            )

    # ---------------------------------------------------------------- matches
    def upsert_match(
        self, *, date: str, team_a: tuple[str, str], team_b: tuple[str, str],
        score: str | None = None, winner: str | None = None, tournament: str | None = None,
        category: str | None = None, round: str | None = None, gender: str | None = None,
        source: str = "manual", source_key: str | None = None,
    ) -> int:
        """Insert a played match or an upcoming fixture (no score/winner).

        `source_key` makes re-scraping idempotent; when absent it's derived from
        date + players so the same fixture is not inserted twice.
        """
        ids = [self.upsert_player(n, gender=gender) for n in (*team_a, *team_b)]
        parsed = parse_score(score)
        if winner is None and parsed is not None:
            winner = parsed.winner
        if source_key is None:
            team_key = lambda x, y: "+".join(sorted((str(x), str(y))))
            teams = sorted([team_key(ids[0], ids[1]), team_key(ids[2], ids[3])])
            source_key = f"{date}|{tournament or ''}|{teams[0]}|{teams[1]}"
        with self.tx() as c:
            existing = c.execute("SELECT a1, a2, b1, b2 FROM matches WHERE source_key=?",
                                 (source_key,)).fetchone()
        if existing and {ids[0], ids[1]} == {existing["b1"], existing["b2"]}:
            # same match reported with teams the other way round: keep the stored
            # orientation so odds/predictions attached to it stay valid
            ids = [ids[2], ids[3], ids[0], ids[1]]
            winner = {"A": "B", "B": "A"}.get(winner) if winner else None
            if parsed is not None:
                parsed = ParsedScore(parsed.sets_b, parsed.sets_a, parsed.games_b,
                                     parsed.games_a, parsed.retired)
            if score:
                score = re.sub(r"(\d+)\s*([-/])\s*(\d+)", r"\3\2\1", score)
        vals = dict(
            date=date, tournament=tournament, category=category, round=round, gender=gender,
            a1=ids[0], a2=ids[1], b1=ids[2], b2=ids[3], winner=winner, score=score,
            sets_a=parsed.sets_a if parsed else None, sets_b=parsed.sets_b if parsed else None,
            games_a=parsed.games_a if parsed else None, games_b=parsed.games_b if parsed else None,
            retired=int(parsed.retired) if parsed else 0, source=source, source_key=source_key,
        )
        with self.tx() as c:
            row = c.execute("SELECT id FROM matches WHERE source_key=?", (source_key,)).fetchone()
            if row:
                sets = ", ".join(f"{k}=?" for k in vals)
                c.execute(f"UPDATE matches SET {sets} WHERE id=?", (*vals.values(), row["id"]))
                return int(row["id"])
            cur = c.execute(
                f"INSERT INTO matches ({', '.join(vals)}) VALUES ({', '.join('?' * len(vals))})",
                tuple(vals.values()),
            )
            return int(cur.lastrowid)

    def find_match(self, team_a: tuple[str, str], team_b: tuple[str, str],
                   date: str | None = None, upcoming_only: bool = True) -> tuple[int, bool] | None:
        """Locate a match by its four players. Returns (match_id, swapped) where
        `swapped` is True if the given team_a is stored as team B."""
        ida = {self.find_player(n) for n in team_a}
        idb = {self.find_player(n) for n in team_b}
        if None in ida or None in idb:
            return None
        q = "SELECT id, a1, a2, b1, b2 FROM matches WHERE 1=1"
        params: list = []
        if upcoming_only:
            q += " AND winner IS NULL"
        if date:
            q += " AND date=?"
            params.append(date)
        for r in self.conn.execute(q + " ORDER BY date DESC", params):
            sa, sb = {r["a1"], r["a2"]}, {r["b1"], r["b2"]}
            if sa == ida and sb == idb:
                return int(r["id"]), False
            if sa == idb and sb == ida:
                return int(r["id"]), True
        return None

    # ---------------------------------------------------------------- odds / predictions
    def add_odds(self, match_id: int, bookmaker: str, odds_a: float, odds_b: float,
                 captured_at: str | None = None) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO odds (match_id, bookmaker, odds_a, odds_b, captured_at)"
                " VALUES (?,?,?,?,?)",
                (match_id, bookmaker, odds_a, odds_b, captured_at or now_iso()),
            )

    def save_prediction(self, match_id: int, model: str, prob_a: float) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO predictions VALUES (?,?,?,?)",
                (match_id, model, float(prob_a), now_iso()),
            )

    # ---------------------------------------------------------------- dataframes
    def matches_df(self, completed: bool | None = None) -> pd.DataFrame:
        q = "SELECT * FROM matches"
        if completed is True:
            q += " WHERE winner IS NOT NULL"
        elif completed is False:
            q += " WHERE winner IS NULL"
        df = pd.read_sql_query(q + " ORDER BY date, id", self.conn)
        return df

    def rankings_df(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM rankings ORDER BY date", self.conn)

    def players_df(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM players", self.conn)

    def latest_odds_df(self, bookmaker: str | None = None) -> pd.DataFrame:
        q = """
        SELECT o.* FROM odds o
        JOIN (SELECT match_id, bookmaker, MAX(captured_at) AS ts FROM odds GROUP BY match_id, bookmaker) l
          ON o.match_id=l.match_id AND o.bookmaker=l.bookmaker AND o.captured_at=l.ts
        """
        params: tuple = ()
        if bookmaker:
            q += " WHERE o.bookmaker=?"
            params = (bookmaker,)
        return pd.read_sql_query(q, self.conn, params=params)

    def stats(self) -> dict[str, int]:
        out = {}
        for t in ("players", "rankings", "matches", "odds", "predictions"):
            out[t] = self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        out["upcoming"] = self.conn.execute(
            "SELECT COUNT(*) FROM matches WHERE winner IS NULL").fetchone()[0]
        return out
