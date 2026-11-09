"""Compare model probabilities with bookmaker odds and pick value bets.

Terminology (decimal odds):
  implied probability      1 / odds
  overround (margin)       1/odds_a + 1/odds_b - 1
  fair probability         implied prob normalised so both sides sum to 1
  expected value (EV)      p_model * odds - 1    (profit per 1 unit staked)
  Kelly fraction           (p*odds - 1) / (odds - 1)
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BetRules:
    bankroll: float = 100.0
    kelly_fraction: float = 0.25   # fractional Kelly - full Kelly is far too aggressive
    min_edge: float = 0.05         # minimum EV per unit staked
    min_prob: float = 0.20         # ignore long shots the model is unsure about
    max_odds: float = 6.0
    max_stake_pct: float = 0.05    # never stake more than this share of bankroll
    shrink: float = 0.0            # blend model prob towards the market (0 = pure model)


def remove_margin(odds_a: float, odds_b: float) -> tuple[float, float, float]:
    ia, ib = 1 / odds_a, 1 / odds_b
    over = ia + ib
    return ia / over, ib / over, over - 1


def kelly(p: float, odds: float) -> float:
    if odds <= 1:
        return 0.0
    return max(0.0, (p * odds - 1) / (odds - 1))


def evaluate(df: pd.DataFrame, rules: BetRules) -> pd.DataFrame:
    """`df` needs columns prob_a, odds_a, odds_b (+ any descriptive columns).

    Returns one row per match side with EV, stake and a BET / no-bet flag."""
    rows = []
    for r in df.to_dict("records"):
        fa, fb, margin = remove_margin(r["odds_a"], r["odds_b"])
        pa_model = r["prob_a"]
        pa = (1 - rules.shrink) * pa_model + rules.shrink * fa
        for side, p, odds, fair in (("A", pa, r["odds_a"], fa), ("B", 1 - pa, r["odds_b"], fb)):
            ev = p * odds - 1
            stake_frac = min(rules.kelly_fraction * kelly(p, odds), rules.max_stake_pct)
            bet = ev >= rules.min_edge and p >= rules.min_prob and odds <= rules.max_odds
            rows.append({
                **{k: v for k, v in r.items() if k not in ("odds_a", "odds_b")},
                "pick": side,
                "pick_team": r.get("team_a" if side == "A" else "team_b"),
                "odds": odds,
                "model_prob": round(p, 4),
                "bookie_prob": round(fair, 4),
                "margin": round(margin, 4),
                "edge_prob": round(p - fair, 4),
                "ev": round(ev, 4),
                "stake": round(stake_frac * rules.bankroll, 2) if bet else 0.0,
                "bet": bet,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["bet", "ev"], ascending=[False, False]).reset_index(drop=True)


def backtest(df: pd.DataFrame, rules: BetRules) -> dict[str, float]:
    """Flat-rules backtest on matches with known outcome (`y` = 1 if A won).

    Stakes are sized from a fixed starting bankroll (no compounding) to keep it simple."""
    ev = evaluate(df, rules)
    bets = ev[ev["bet"]].copy()
    if bets.empty:
        return {"bets": 0, "staked": 0.0, "profit": 0.0, "roi": 0.0, "hit_rate": 0.0}
    won = ((bets["pick"] == "A") & (bets["y"] == 1)) | ((bets["pick"] == "B") & (bets["y"] == 0))
    profit = (bets["stake"] * (bets["odds"] - 1)).where(won, -bets["stake"]).sum()
    staked = bets["stake"].sum()
    return {
        "bets": int(len(bets)),
        "staked": round(float(staked), 2),
        "profit": round(float(profit), 2),
        "roi": round(float(profit / staked), 4) if staked else 0.0,
        "hit_rate": round(float(won.mean()), 4),
    }
