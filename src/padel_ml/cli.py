"""Command line interface: `padel-ml <command>`.

Typical workflow
  padel-ml scrape-rankings                       # players + FIP ranking
  padel-ml scrape-results URL --tournament X --date YYYY-MM-DD
  padel-ml import-matches history.csv           # bulk history (most reliable)
  padel-ml train
  padel-ml odds-template odds.csv               # fill with bet365.es prices
  padel-ml import-odds odds.csv
  padel-ml value-bets                           # what to bet on
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from . import betting, demo
from .config import get_settings
from .db import Database


def _db(args) -> Database:
    return Database(args.db or get_settings().db_path)


def _rules(args) -> betting.BetRules:
    s = get_settings()
    return betting.BetRules(
        bankroll=args.bankroll if args.bankroll is not None else s.bankroll,
        kelly_fraction=args.kelly if args.kelly is not None else s.kelly_fraction,
        min_edge=args.min_edge if args.min_edge is not None else s.min_edge,
        max_stake_pct=args.max_stake if args.max_stake is not None else s.max_stake_pct,
        min_prob=args.min_prob, max_odds=args.max_odds, shrink=args.shrink,
    )


def _team_names(db: Database) -> pd.DataFrame:
    return pd.read_sql_query(
        """SELECT m.id AS match_id, m.date, m.tournament, m.round, m.winner,
                  pa1.name || ' / ' || pa2.name AS team_a,
                  pb1.name || ' / ' || pb2.name AS team_b
           FROM matches m
           JOIN players pa1 ON pa1.id=m.a1 JOIN players pa2 ON pa2.id=m.a2
           JOIN players pb1 ON pb1.id=m.b1 JOIN players pb2 ON pb2.id=m.b2""", db.conn)


# ------------------------------------------------------------------ commands
def cmd_init(args) -> None:
    db = _db(args)
    print(f"database ready at {db.path}: {db.stats()}")


def cmd_stats(args) -> None:
    print(json.dumps(_db(args).stats(), indent=2))


def cmd_scrape_rankings(args) -> None:
    from .scrapers import Fetcher, scrape_rankings
    n = scrape_rankings(_db(args), Fetcher(use_cache=not args.no_cache),
                        tuple(args.gender))
    print(f"stored {n} ranking rows")


def cmd_scrape_results(args) -> None:
    from .scrapers import Fetcher, scrape_results
    db, f = _db(args), Fetcher(use_cache=not args.no_cache)
    total = 0
    for url in args.urls:
        total += scrape_results(db, url, args.tournament, args.date, args.category,
                                args.gender, f)
    print(f"stored {total} matches")


def cmd_import_matches(args) -> None:
    from .scrapers import import_matches
    print(f"imported {import_matches(_db(args), args.csv)} matches")


def cmd_import_players(args) -> None:
    from .scrapers import import_players
    print(f"imported {import_players(_db(args), args.csv)} players")


def cmd_import_odds(args) -> None:
    from .scrapers import import_odds
    ok, skipped = import_odds(_db(args), args.csv, default_bookmaker=args.bookmaker)
    print(f"imported {ok} odds rows ({skipped} skipped)")


def cmd_odds_template(args) -> None:
    from .odds import export_odds_template
    n = export_odds_template(_db(args), args.out)
    print(f"wrote {n} upcoming fixtures to {args.out} - fill odds_a/odds_b from bet365.es")


def cmd_train(args) -> None:
    from .model import train
    rep = train(_db(args), get_settings().model_dir, holdout_frac=args.holdout)
    print(f"trained on {rep.n_train_matches} matches; chosen model: {rep.chosen}")
    print(pd.DataFrame(rep.validation).T.round(4).to_string())


def cmd_predict(args) -> None:
    from .model import predict_upcoming
    db = _db(args)
    pred = predict_upcoming(db, get_settings().model_dir)
    if pred.empty:
        print("no upcoming fixtures in the database")
        return
    out = pred.merge(_team_names(db).drop(columns=["date"]), on="match_id")
    out["favourite"] = out.apply(lambda r: r.team_a if r.prob_a >= .5 else r.team_b, axis=1)
    out["fav_prob"] = out.prob_a.where(out.prob_a >= .5, 1 - out.prob_a).round(3)
    print(out[["date", "tournament", "team_a", "team_b", "favourite", "fav_prob"]]
          .to_string(index=False))


def cmd_value_bets(args) -> None:
    from .model import predict_upcoming
    db = _db(args)
    pred = predict_upcoming(db, get_settings().model_dir)
    odds = db.latest_odds_df(args.bookmaker)
    df = (pred.merge(odds[["match_id", "odds_a", "odds_b"]], on="match_id")
          .merge(_team_names(db).drop(columns=["date", "winner"]), on="match_id"))
    if df.empty:
        print(f"no upcoming fixtures with {args.bookmaker} odds - run odds-template/import-odds")
        return
    res = betting.evaluate(df[["match_id", "date", "tournament", "team_a", "team_b",
                               "prob_a", "odds_a", "odds_b"]], _rules(args))
    show = res if args.all else res[res["bet"]]
    cols = ["date", "tournament", "pick_team", "odds", "model_prob", "bookie_prob", "ev",
            "stake"]
    if show.empty:
        print("no value bets found with the current thresholds")
    else:
        print(show[cols + (["bet"] if args.all else [])].to_string(index=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(args.out, index=False)
        print(f"full table written to {args.out}")


def cmd_backtest(args) -> None:
    from .model import holdout_predictions
    db = _db(args)
    hp = holdout_predictions(db, args.model, holdout_frac=args.holdout)
    odds = db.latest_odds_df(args.bookmaker)
    df = hp.merge(odds[["match_id", "odds_a", "odds_b"]], on="match_id")
    if df.empty:
        print("no historical odds on the hold-out matches - import past bet365 odds first")
        return
    rules = _rules(args)
    print(f"hold-out matches with odds: {len(df)}")
    print("model    :", betting.backtest(df, rules))
    print("elo only :", betting.backtest(df.assign(prob_a=df["elo_prob"]), rules))


def cmd_demo(args) -> None:
    db = _db(args)
    print("generated SYNTHETIC demo data:", demo.populate(db, seed=args.seed))


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="padel-ml", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", help="SQLite path (default data/padel.db or $PADEL_ML_DB)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_)
        sp.set_defaults(fn=fn)
        return sp

    def bet_opts(sp):
        sp.add_argument("--bookmaker", default="bet365")
        sp.add_argument("--bankroll", type=float)
        sp.add_argument("--kelly", type=float, help="fraction of Kelly stake (default 0.25)")
        sp.add_argument("--min-edge", type=float, help="minimum expected value (default 0.05)")
        sp.add_argument("--max-stake", type=float, help="max stake as share of bankroll")
        sp.add_argument("--min-prob", type=float, default=0.20)
        sp.add_argument("--max-odds", type=float, default=6.0)
        sp.add_argument("--shrink", type=float, default=0.0,
                        help="blend model probability towards the market (0..1)")

    add("init", cmd_init, "create the database")
    add("stats", cmd_stats, "row counts")

    sp = add("scrape-rankings", cmd_scrape_rankings, "scrape FIP rankings (players + points)")
    sp.add_argument("--gender", nargs="+", default=["M", "F"], choices=["M", "F"])
    sp.add_argument("--no-cache", action="store_true")

    sp = add("scrape-results", cmd_scrape_results, "scrape a tournament results/draw page")
    sp.add_argument("urls", nargs="+")
    sp.add_argument("--tournament", required=True)
    sp.add_argument("--date", default=date.today().isoformat(),
                    help="fallback match date when the page doesn't show one")
    sp.add_argument("--category")
    sp.add_argument("--gender", choices=["M", "F"])
    sp.add_argument("--no-cache", action="store_true")

    for name, fn, h in (("import-matches", cmd_import_matches, "load matches/fixtures CSV"),
                        ("import-players", cmd_import_players, "load players CSV")):
        add(name, fn, h).add_argument("csv")
    sp = add("import-odds", cmd_import_odds, "load bookmaker odds CSV")
    sp.add_argument("csv")
    sp.add_argument("--bookmaker", default="bet365")
    add("odds-template", cmd_odds_template, "export upcoming fixtures to fill with odds") \
        .add_argument("out", nargs="?", default="data/odds_to_fill.csv")

    sp = add("train", cmd_train, "train and validate the winner model")
    sp.add_argument("--holdout", type=float, default=0.2)
    add("predict", cmd_predict, "predict upcoming fixtures")

    sp = add("value-bets", cmd_value_bets, "compare predictions with odds; list bets")
    bet_opts(sp)
    sp.add_argument("--all", action="store_true", help="show non-bets too")
    sp.add_argument("--out", help="write the full table to CSV")

    sp = add("backtest", cmd_backtest, "simulate the betting rules on past matches")
    bet_opts(sp)
    sp.add_argument("--model", default="logreg", choices=["logreg", "gbm"])
    sp.add_argument("--holdout", type=float, default=0.3)

    sp = add("demo", cmd_demo, "fill the DB with synthetic data to try the pipeline")
    sp.add_argument("--seed", type=int, default=7)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        args.fn(args)
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
