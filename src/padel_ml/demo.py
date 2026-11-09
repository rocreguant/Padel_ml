"""Synthetic data so the whole pipeline can be exercised without network access.

Players get a hidden skill; pairs form and split; matches are simulated with a
logistic link on skill difference; a fake bookmaker prices matches from a noisy view
of skill plus a ~6% margin. NONE of this is real data - it only proves the plumbing.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from .db import Database


def _sim_score(rng: np.random.Generator, p: float) -> str:
    sets, sa, sb = [], 0, 0
    while sa < 2 and sb < 2:
        a_wins = rng.random() < p
        loser_games = int(rng.integers(0, 6))
        g = (6, loser_games) if loser_games < 5 else (7, loser_games)
        sets.append(g if a_wins else g[::-1])
        sa += a_wins
        sb += not a_wins
    return " ".join(f"{a}-{b}" for a, b in sets)


def populate(db: Database, n_players: int = 64, n_days: int = 540, seed: int = 7,
             n_upcoming: int = 12, historical_odds: bool = True) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    names = [f"Demo Player {i:02d}" for i in range(n_players)]
    skill = rng.normal(0, 1, n_players)
    order = np.argsort(-skill)
    for rank, i in enumerate(order, 1):
        pid = db.upsert_player(names[i], gender="M", country="ESP", source="demo")
        db.add_ranking(pid, "2025-01-01", rank, round(max(0, 10000 * np.exp(-rank / 15)), 1),
                       "demo")

    def make_pairs() -> list[tuple[int, int]]:
        idx = list(order)  # pair adjacent-ish skill levels, like real tours
        pairs = []
        while len(idx) >= 2:
            a = idx.pop(0)
            j = min(int(abs(rng.normal(0, 2))), len(idx) - 1)
            pairs.append((a, idx.pop(j)))
        return pairs

    start = date(2025, 1, 6)
    pairs = make_pairs()
    n_matches = n_odds = 0
    for week in range(n_days // 7):
        if week % 12 == 11:
            pairs = make_pairs()  # partnerships change during the season
        day = start + timedelta(weeks=week)
        sel = rng.permutation(len(pairs))[: 16]
        for k in range(0, len(sel) - 1, 2):
            pa, pb = pairs[sel[k]], pairs[sel[k + 1]]
            diff = skill[list(pa)].mean() - skill[list(pb)].mean()
            p = 1 / (1 + np.exp(-1.6 * diff))
            score = _sim_score(rng, p)
            mid = db.upsert_match(
                date=day.isoformat(), team_a=(names[pa[0]], names[pa[1]]),
                team_b=(names[pb[0]], names[pb[1]]), score=score,
                tournament=f"Demo Open W{week}", category="P2", round="R16", gender="M",
                source="demo")
            n_matches += 1
            if historical_odds:
                n_odds += _add_fake_odds(db, rng, mid, diff, day.isoformat())
    # upcoming fixtures
    day = start + timedelta(weeks=n_days // 7 + 1)
    sel = rng.permutation(len(pairs))[: n_upcoming * 2]
    for k in range(0, len(sel) - 1, 2):
        pa, pb = pairs[sel[k]], pairs[sel[k + 1]]
        mid = db.upsert_match(date=day.isoformat(), team_a=(names[pa[0]], names[pa[1]]),
                              team_b=(names[pb[0]], names[pb[1]]),
                              tournament="Demo Masters", category="P1", round="R16",
                              gender="M", source="demo")
        diff = skill[list(pa)].mean() - skill[list(pb)].mean()
        n_odds += _add_fake_odds(db, rng, mid, diff, day.isoformat())
    return {"players": n_players, "matches": n_matches, "odds": n_odds}


def _add_fake_odds(db: Database, rng: np.random.Generator, mid: int, diff: float,
                   on: str) -> int:
    noisy = diff + rng.normal(0, 0.45)  # bookmaker's imperfect view
    p = 1 / (1 + np.exp(-1.6 * noisy))
    p = float(np.clip(p, 0.04, 0.96))
    margin = 1.06
    db.add_odds(mid, "bet365", round(1 / (p * margin), 2), round(1 / ((1 - p) * margin), 2),
                f"{on}T09:00:00+00:00")
    return 1
