import numpy as np
import pytest

from padel_ml import betting, demo
from padel_ml.features import FEATURES, FeatureState, build_training_set
from padel_ml.model import holdout_predictions, predict_upcoming, train


def test_features_antisymmetric_and_no_leak(db):
    demo.populate(db, n_players=16, n_days=70, n_upcoming=2)
    ds, state = build_training_set(db.matches_df(), db.rankings_df())
    # very first match: no history, so only ranking-based features can be non-zero
    first = ds.iloc[0]
    for f in FEATURES:
        if not f.startswith("rank") and f != "rest_diff":
            assert first[f] == 0, f
    fa = state.features(1, 2, 3, 4, "2026-01-01")
    fb = state.features(3, 4, 1, 2, "2026-01-01")
    for f in FEATURES:
        assert fa[f] == pytest.approx(-fb[f])


def test_strict_asof_ranking():
    import pandas as pd
    st = FeatureState(pd.DataFrame({"player_id": [1], "date": ["2026-06-01"],
                                    "position": [1], "points": [1000.0]}))
    assert np.isnan(st._ranking(1, "2026-01-01")[0])
    assert st._ranking(1, "2026-07-01") == (1.0, 1000.0)


def test_betting_math():
    pa, pb, margin = betting.remove_margin(1.80, 2.00)
    assert pa + pb == pytest.approx(1)
    assert margin == pytest.approx(1 / 1.8 + 0.5 - 1)
    assert betting.kelly(0.6, 2.0) == pytest.approx(0.2)
    assert betting.kelly(0.4, 2.0) == 0
    import pandas as pd
    res = betting.evaluate(pd.DataFrame([{"prob_a": 0.6, "odds_a": 2.0, "odds_b": 1.8}]),
                           betting.BetRules(bankroll=100, kelly_fraction=0.5, max_stake_pct=1))
    top = res.iloc[0]
    assert top.pick == "A" and top.bet and top.ev == pytest.approx(0.2)
    assert top.stake == pytest.approx(10.0)
    assert not res.iloc[1].bet


def test_end_to_end_demo(db, tmp_path):
    demo.populate(db, n_players=40, n_days=300, n_upcoming=6)
    rep = train(db, tmp_path)
    assert rep.validation[rep.chosen]["log_loss"] < rep.validation["elo_baseline"]["log_loss"] + 0.05
    pred = predict_upcoming(db, tmp_path)
    assert len(pred) == 6 and pred.prob_a.between(0, 1).all()
    hp = holdout_predictions(db)
    assert len(hp) > 20 and hp.prob_a.between(0, 1).all()
