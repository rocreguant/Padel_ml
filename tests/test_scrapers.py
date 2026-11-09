from padel_ml.scrapers import import_matches, import_odds, parse_ranking_html, parse_results_html

from .conftest import FIXTURES


def test_parse_ranking_table():
    rows = parse_ranking_html((FIXTURES / "ranking_table.html").read_text())
    assert [r.position for r in rows] == [1, 2, 3]
    assert rows[0].name == "Agustín Tapia"
    assert rows[0].country == "ARG"
    assert rows[0].points == 14230


def test_parse_ranking_cards():
    rows = parse_ranking_html((FIXTURES / "ranking_cards.html").read_text())
    assert [(r.position, r.name, r.points) for r in rows] == [
        (1, "Ariana Sánchez", 12500), (2, "Paula Josemaría", 12500)]


def test_parse_results():
    ms = parse_results_html((FIXTURES / "results.html").read_text())
    assert len(ms) == 3
    final = ms[0]
    assert final.team_a == ("Agustín Tapia", "Arturo Coello")
    assert final.score == "6-4 7-6" and final.date == "2026-05-10" and final.round == "Final"
    assert ms[1].score == "3-6 6-4 4-6" and ms[1].date == "2026-05-09"
    assert ms[2].score is None  # upcoming


def test_csv_import_and_odds(db, tmp_path):
    m = tmp_path / "m.csv"
    m.write_text("date,team_a_1,team_a_2,team_b_1,team_b_2,score,tournament\n"
                 "2026-01-01,A1,A2,B1,B2,6-1 6-1,T\n"
                 "2026-02-01,A1,A2,B1,B2,,T\n")
    assert import_matches(db, m) == 2
    o = tmp_path / "o.csv"
    o.write_text("date,team_a_1,team_a_2,team_b_1,team_b_2,odds_a,odds_b\n"
                 "2026-02-01,B1,B2,A1,A2,\"3,10\",1.35\n"   # teams swapped, comma decimal
                 "2026-02-01,X,Y,Z,W,,\n")                  # unfilled template row
    assert import_odds(db, o) == (1, 1)
    odds = db.latest_odds_df("bet365").iloc[0]
    assert (odds.odds_a, odds.odds_b) == (1.35, 3.10)
