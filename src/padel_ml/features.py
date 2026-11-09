"""Leak-free feature engineering for doubles padel matches.

Matches are replayed in chronological order. For every match the *pre-match* state
(ratings, form, head-to-head, ranking) is turned into a feature vector, and only then
is the state updated with the result. The same state, after replaying all history,
produces features for upcoming fixtures.

All features are "team A minus team B" differences, so flipping the teams negates
the vector - which lets us augment training data and symmetrise predictions.
"""

from __future__ import annotations

import bisect
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

FEATURES = [
    "elo_diff",            # mean individual Elo
    "pair_elo_diff",       # Elo of the exact pairing
    "elo_min_diff",        # weakest player of each team
    "form_diff",           # shrunk win-rate over last N matches
    "games_share_diff",    # share of games won over last N matches
    "experience_diff",     # log(1 + matches played)
    "pair_experience_diff",  # log(1 + matches played together)
    "h2h_diff",            # shrunk head-to-head between these two pairs
    "rank_points_diff",    # log(1 + summed ranking points)
    "rank_pos_diff",       # -log(best ranking position)
    "rest_diff",           # log(1 + days since last match), capped
]

BASE_ELO = 1500.0
FORM_N = 10


def _d(s: str) -> date:
    return date.fromisoformat(str(s)[:10])


@dataclass
class _PlayerState:
    elo: float = BASE_ELO
    n: int = 0
    recent: deque = field(default_factory=lambda: deque(maxlen=FORM_N))  # (won, gw, gl)
    last: date | None = None


class FeatureState:
    def __init__(self, rankings: pd.DataFrame | None = None, k_base: float = 48.0):
        self.k_base = k_base
        self.players: dict[int, _PlayerState] = defaultdict(_PlayerState)
        self.pair_elo: dict[frozenset, float] = {}
        self.pair_n: dict[frozenset, int] = defaultdict(int)
        self.h2h: dict[tuple[frozenset, frozenset], int] = defaultdict(int)  # wins of first
        self._rank: dict[int, tuple[list[str], list[tuple[float, float]]]] = {}
        if rankings is not None and len(rankings):
            for pid, g in rankings.sort_values("date").groupby("player_id"):
                self._rank[int(pid)] = (
                    [str(x)[:10] for x in g["date"]],
                    list(zip(g["position"].astype(float).fillna(np.nan),
                             g["points"].astype(float).fillna(np.nan))),
                )

    # ------------------------------------------------------------- helpers
    def _ranking(self, pid: int, on: str) -> tuple[float, float]:
        """Most recent (position, points) published on or before `on`.

        Strictly as-of: using today's ranking for a 2024 match would leak the
        future into training. Re-run `scrape-rankings` regularly to build history."""
        if pid not in self._rank:
            return (np.nan, np.nan)
        dates, vals = self._rank[pid]
        i = bisect.bisect_right(dates, str(on)[:10]) - 1
        return vals[i] if i >= 0 else (np.nan, np.nan)

    def _team(self, p1: int, p2: int, on: str) -> dict[str, float]:
        a, b = self.players[p1], self.players[p2]
        pair = frozenset((p1, p2))

        def form(ps: _PlayerState) -> float:
            w = sum(r[0] for r in ps.recent)
            return (w + 2.0) / (len(ps.recent) + 4.0)  # Beta(2,2) shrinkage

        def gshare(ps: _PlayerState) -> float:
            gw = sum(r[1] for r in ps.recent)
            gl = sum(r[2] for r in ps.recent)
            return (gw + 6.0) / (gw + gl + 12.0)

        def rest(ps: _PlayerState) -> float:
            if ps.last is None:
                return math.log1p(60)
            return math.log1p(min(max((_d(on) - ps.last).days, 0), 60))

        r1, r2 = self._ranking(p1, on), self._ranking(p2, on)
        pts = np.nansum([r1[1], r2[1]]) if not (np.isnan(r1[1]) and np.isnan(r2[1])) else 0.0
        positions = [p for p in (r1[0], r2[0]) if not np.isnan(p) and p > 0]
        best_pos = min(positions) if positions else 1000.0
        return {
            "elo": (a.elo + b.elo) / 2,
            "pair_elo": self.pair_elo.get(pair, (a.elo + b.elo) / 2),
            "elo_min": min(a.elo, b.elo),
            "form": (form(a) + form(b)) / 2,
            "games_share": (gshare(a) + gshare(b)) / 2,
            "experience": math.log1p((a.n + b.n) / 2),
            "pair_experience": math.log1p(self.pair_n[pair]),
            "rank_points": math.log1p(max(pts, 0.0)),
            "rank_pos": -math.log(best_pos),
            "rest": (rest(a) + rest(b)) / 2,
        }

    # ------------------------------------------------------------- public
    def features(self, a1: int, a2: int, b1: int, b2: int, on: str) -> dict[str, float]:
        ta, tb = self._team(a1, a2, on), self._team(b1, b2, on)
        pa, pb = frozenset((a1, a2)), frozenset((b1, b2))
        wa, wb = self.h2h[(pa, pb)], self.h2h[(pb, pa)]
        f = {f"{k}_diff": ta[k] - tb[k] for k in ta}
        f["h2h_diff"] = (wa - wb) / (wa + wb + 2.0)
        return {k: f[k] for k in FEATURES}

    def elo_prob(self, a1: int, a2: int, b1: int, b2: int) -> float:
        ra = (self.players[a1].elo + self.players[a2].elo) / 2
        rb = (self.players[b1].elo + self.players[b2].elo) / 2
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400))

    def update(self, a1: int, a2: int, b1: int, b2: int, a_won: bool, on: str,
               games_a: float | None = None, games_b: float | None = None,
               sets_a: float | None = None, sets_b: float | None = None) -> None:
        exp_a = self.elo_prob(a1, a2, b1, b2)
        s = 1.0 if a_won else 0.0
        # margin of victory: straight-sets wins move ratings a bit more
        mov = 1.0
        if sets_a is not None and sets_b is not None and not (
                pd.isna(sets_a) or pd.isna(sets_b)):
            mov = 1.25 if abs(sets_a - sets_b) >= 2 else 1.0
        for team, sign in (((a1, a2), 1), ((b1, b2), -1)):
            for p in team:
                ps = self.players[p]
                k = max(16.0, self.k_base / (1 + ps.n / 30))
                ps.elo += sign * k * mov * (s - exp_a)
        pa, pb = frozenset((a1, a2)), frozenset((b1, b2))
        ea = self.pair_elo.get(pa, BASE_ELO)
        eb = self.pair_elo.get(pb, BASE_ELO)
        pexp = 1.0 / (1.0 + 10 ** ((eb - ea) / 400))
        self.pair_elo[pa] = ea + 32 * mov * (s - pexp)
        self.pair_elo[pb] = eb - 32 * mov * (s - pexp)
        self.pair_n[pa] += 1
        self.pair_n[pb] += 1
        self.h2h[(pa, pb) if a_won else (pb, pa)] += 1
        ga = 0.0 if games_a is None or pd.isna(games_a) else float(games_a)
        gb = 0.0 if games_b is None or pd.isna(games_b) else float(games_b)
        d = _d(on)
        for team, won, gw, gl in (((a1, a2), a_won, ga, gb), ((b1, b2), not a_won, gb, ga)):
            for p in team:
                ps = self.players[p]
                ps.n += 1
                ps.recent.append((int(won), gw, gl))
                ps.last = d


def build_training_set(matches: pd.DataFrame, rankings: pd.DataFrame | None = None
                       ) -> tuple[pd.DataFrame, FeatureState]:
    """Replay completed matches; return per-match features + label and the final state."""
    state = FeatureState(rankings)
    rows = []
    done = matches[matches["winner"].notna()].sort_values(["date", "id"])
    for m in done.itertuples(index=False):
        f = state.features(m.a1, m.a2, m.b1, m.b2, m.date)
        f.update(match_id=m.id, date=m.date, y=int(m.winner == "A"),
                 elo_prob=state.elo_prob(m.a1, m.a2, m.b1, m.b2))
        rows.append(f)
        state.update(m.a1, m.a2, m.b1, m.b2, m.winner == "A", m.date,
                     m.games_a, m.games_b, m.sets_a, m.sets_b)
    cols = ["match_id", "date", *FEATURES, "elo_prob", "y"]
    return pd.DataFrame(rows, columns=cols), state


def fixture_features(state: FeatureState, fixtures: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in fixtures.itertuples(index=False):
        f = state.features(m.a1, m.a2, m.b1, m.b2, m.date)
        f.update(match_id=m.id, date=m.date, elo_prob=state.elo_prob(m.a1, m.a2, m.b1, m.b2))
        rows.append(f)
    return pd.DataFrame(rows, columns=["match_id", "date", *FEATURES, "elo_prob"])
