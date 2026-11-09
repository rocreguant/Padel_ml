from padel_ml.utils import normalize_name, parse_score


def test_normalize_name():
    assert normalize_name("  Agustín  TAPIA ") == "agustin tapia"
    assert normalize_name("Tapia, Agustín") == "agustin tapia"


def test_parse_score():
    s = parse_score("6-4 3-6 7-6(5)")
    assert (s.sets_a, s.sets_b, s.games_a, s.games_b) == (2, 1, 16, 16)
    assert s.winner == "A"
    assert parse_score("") is None
    assert parse_score("4-6 2-6").winner == "B"


def test_upsert_player_is_idempotent(db):
    a = db.upsert_player("Agustín Tapia", country="ARG")
    b = db.upsert_player("agustin tapia")
    assert a == b
    assert db.stats()["players"] == 1


def test_match_upsert_and_find_swapped(db):
    mid = db.upsert_match(date="2026-01-01", team_a=("A1", "A2"), team_b=("B1", "B2"))
    again = db.upsert_match(date="2026-01-01", team_a=("B2", "B1"), team_b=("A2", "A1"))
    assert mid == again  # same fixture regardless of team order
    assert db.find_match(("A1", "A2"), ("B1", "B2")) == (mid, False)
    assert db.find_match(("B1", "B2"), ("A2", "A1")) == (mid, True)
    # adding the score turns the fixture into a completed match
    # result reported from the other side: stored orientation must be kept
    db.upsert_match(date="2026-01-01", team_a=("B1", "B2"), team_b=("A1", "A2"), score="3-6 2-6")
    m = db.matches_df(completed=True)
    assert len(m) == 1
    r = m.iloc[0]
    assert (r["winner"], r["games_a"], r["games_b"], r["score"]) == ("A", 12, 5, "6-3 6-2")
