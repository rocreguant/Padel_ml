import pytest

from padel_ml.scrapers import padelapi
from padel_ml.scrapers.padelapi import PadelApiClient, score_to_text, sync_matches, sync_rankings


class FakeResponse:
    def __init__(self, body, status=200, headers=None):
        self._body, self.status_code, self.headers, self.text = body, status, headers or {}, ""

    def json(self):
        return self._body


def page(data, more=False, current=1, last=1):
    return {"data": data, "links": {"next": "x" if more else None},
            "meta": {"current_page": current, "last_page": last}}


def player(i, name, side="drive"):
    return {"id": i, "name": name, "nationality": "ES", "side": side}


def match(mid, status="finished", winner="team_1", score=None, played_at="2026-05-10", rnd=1):
    return {"id": mid, "status": status, "category": "men", "round": rnd,
            "played_at": played_at, "winner": winner,
            "score": score if score is not None else [{"team_1": "6", "team_2": "4"},
                                                      {"team_1": "7(5)", "team_2": "6"}],
            "players": {"team_1": [player(1, "Agustín Tapia"), player(2, "Arturo Coello")],
                        "team_2": [player(3, "Alejandro Galán"), player(4, "Fede Chingotto")]}}


class FakeSession:
    def __init__(self, routes):
        self.routes, self.headers, self.calls = routes, {}, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        path = url.replace(padelapi.BASE_URL, "")
        resp = self.routes[path]
        if callable(resp):
            resp = resp(params or {})
        return resp if isinstance(resp, FakeResponse) else FakeResponse(resp)


def client(routes):
    return PadelApiClient(token="t", session=FakeSession(routes), requests_per_minute=10_000)


def test_score_to_text():
    assert score_to_text([{"team_1": "6", "team_2": "4"}, {"team_1": "6(3)", "team_2": "7"}]) \
        == "6-4 6-7"
    assert score_to_text("hidden_free_plan") is None and score_to_text(None) is None


def test_token_required(monkeypatch):
    monkeypatch.delenv("PADEL_API_TOKEN", raising=False)
    with pytest.raises(padelapi.PadelApiError):
        PadelApiClient()


def test_sync_matches_maps_and_paginates(db):
    tourn = {"id": 9, "name": "Madrid P1", "level": "p1", "status": "finished",
             "format": "pairs", "start_date": "2026-05-04"}
    pages = {1: page([match(100), match(101, status="walkover")], more=True, last=2),
             2: page([match(102, winner="team_2", rnd=2,
                             score=[{"team_1": "3", "team_2": "6"}, {"team_1": "4", "team_2": "6"}]),
                      match(103, status="finished", winner="hidden_free_plan",
                            score="hidden_free_plan"),
                      match(104, status="scheduled", winner=None, score=[],
                            played_at="2999-01-01")], current=2, last=2)}
    c = client({"/tournaments": page([tourn]),
                "/tournaments/9/matches": lambda p: pages[p["page"]]})
    counts = sync_matches(db, c, since="2026-01-01")
    assert counts["played"] == 2 and counts["fixture"] == 1
    assert counts["not_played"] == 1 and counts["hidden_free_plan"] == 1
    df = db.matches_df()
    final = df[df.source_key == "padelapi:100"].iloc[0]
    assert (final.winner, final.score, final["round"], final.category, final.gender) == \
        ("A", "6-4 7-6", "F", "p1", "M")
    assert df[df.source_key == "padelapi:102"].iloc[0].winner == "B"
    # re-sync is idempotent
    sync_matches(db, client({"/tournaments": page([tourn]),
                             "/tournaments/9/matches": lambda p: pages[p["page"]]}))
    assert len(db.matches_df()) == 3
    assert db.stats()["players"] == 4


def test_rate_limit_retry(db, monkeypatch):
    monkeypatch.setattr(padelapi.time, "sleep", lambda s: None)
    responses = iter([FakeResponse({}, 429, {"Retry-After": "1"}), FakeResponse(page([]))])
    c = client({"/rankings": lambda p: next(responses)})
    assert sync_rankings(db, c, categories=("men",)) == 0
    assert c.requests_made == 2


def test_sync_rankings(db):
    rows = [{"id": 1, "name": "Agustín Tapia", "nationality": "AR", "ranking": 1,
             "points": 15000, "date": "2026-11-02"}]
    c = client({"/rankings": page(rows)})
    assert sync_rankings(db, c, categories=("men",)) == 1
    r = db.rankings_df().iloc[0]
    assert (r.position, r.points, r.date, r.source) == (1, 15000, "2026-11-02", "padelapi")
