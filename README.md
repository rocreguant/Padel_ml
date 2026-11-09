# padel_ml

Predict the winners of professional padel matches and compare the predictions with
bookmaker odds (bet365.es) to find **value bets**: matches where the model rates a
pair's chances higher than the odds imply.

```
 ┌───────────── 1. data ─────────────┐   ┌──── 2. model ────┐   ┌──── 3. betting ────┐
 Padel API    ┐                        │  chronological    │   │ bet365 odds (CSV)  │
 results pages ├─► SQLite (players,     ├─► feature replay  ├──►│ remove margin      ├─► bets
 CSV history  ─┘   rankings, matches,   │  Elo + LogReg/GBM │   │ EV, fractional     │
                   odds, predictions)   │  time-split eval  │   │ Kelly stake        │
                                        └───────────────────┘   └────────────────────┘
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest              # 17 tests, no network needed
```

## Try it in 30 seconds (synthetic data)

```bash
export PADEL_ML_DB=data/demo.db
padel-ml demo           # 64 fake players, ~600 simulated matches, fake odds
padel-ml train          # compares Elo baseline, logistic regression, gradient boosting
padel-ml predict
padel-ml value-bets
padel-ml backtest
```

The demo data is **made up**. It only shows that the plumbing works.

## Real workflow

### 1. Build the database

**Main source: [Padel API](https://padelapi.org).** It's a public REST API with Premier
Padel and FIP results (every round from 2023 onward), World Padel Tour and Padel Pro
Tour finals back to 2006, and weekly rankings. Its terms allow storing the data for
your own use.

| Plan | Price | What you get |
|---|---|---|
| Free | €0 | 50k requests/month (10/min), current rankings, **scores from the last 6 months only** |
| Plus | €19/month | full historical archive, 60 requests/min |

```bash
export PADEL_API_TOKEN=...            # from https://padelapi.org/register
padel-ml sync-matches --since 2023-01-01           # Premier Padel: major,p1,p2,finals
padel-ml sync-matches --since 2023-01-01 --levels fip_platinum,fip_gold
padel-ml sync-rankings                             # run weekly to build ranking history
padel-ml sync-matches --since 2026-11-01           # re-run to pick up new results/fixtures
```

Syncing can be repeated safely because every match is keyed by its Padel API id.
Fixtures that haven't been played yet are stored as upcoming matches, ready for
predictions and odds. On the free plan, older matches come back with hidden scores and
are skipped, and the sync tells you how many. At 10 requests/min, a full Premier Padel
season (about 25 tournaments × 3–4 pages) takes roughly 10 minutes.

Other sources:

| Command | Source | Status |
|---|---|---|
| `padel-ml import-matches history.csv` | any CSV you assemble | works; format in `examples/` |
| `padel-ml import-players players.csv` | CSV | extra player attributes (hand, height, birth date) |
| `padel-ml scrape-results URL ...` | a tournament page with results in its HTML | generic parser; premierpadel.com loads results with JavaScript, so it finds nothing there |
| `padel-ml scrape-rankings` | padelfip.com | **blocked**: the site shows bots a Cloudflare challenge (HTTP 403) |

Names are matched accent- and case-insensitively ("Agustín TAPIA" and "agustin tapia"
are the same player).

**Run `sync-rankings` regularly (e.g. weekly).** Ranking features are strictly
"as of" the match date. A ranking scraped today is never used for a match played
last year, because that would leak the future into training. The ranking features
become useful once a history of snapshots has built up.

### 2. Train and predict

```bash
padel-ml train      # needs ≥ 50 completed matches; several thousand is far better
padel-ml predict    # P(win) for every upcoming fixture
```

Features (all as "pair A minus pair B" differences, computed only from earlier matches):

- individual Elo (mean of both players, plus the weaker player), with a straight-sets bonus
- Elo of the exact pairing (padel pairs change often, so chemistry matters)
- recent form and share of games won over the last 10 matches (shrunk towards 50 %)
- experience, and matches played together as a pair
- head-to-head between the two pairs
- FIP ranking points and position as of the match date
- days of rest

Training validates on the most recent 20 % of matches, keeps whichever of logistic
regression or gradient boosting has the lowest log-loss (shown next to a pure Elo
baseline), then refits on all data. Every match is also used with the teams swapped,
so the order the source lists the teams in (often the winner first) cannot bias the
model.

### 3. Compare with bet365 and choose bets

bet365 has no public API, loads its prices with JavaScript behind bot protection, and
its terms forbid automated collection, so **this project does not scrape bet365.es**.
Collecting the odds is a manual step that takes a couple of minutes:

```bash
padel-ml odds-template data/odds_today.csv   # one row per upcoming fixture
# open bet365.es and type the decimal odds into odds_a / odds_b ("1,72" also works)
padel-ml import-odds data/odds_today.csv
padel-ml value-bets --bankroll 200
```

Example output:

```
      date   tournament                       pick_team  odds  model_prob  bookie_prob     ev  stake
2026-07-06 Demo Masters Demo Player 13 / Demo Player 63  3.23      0.4080       0.2917 0.3178   3.56
```

How a bet is chosen:

- **bookie_prob**: bet365's implied probability with its margin removed.
- **ev** = `model_prob × odds − 1`, the expected profit per €1 staked.
- A match is a **bet** when `ev ≥ --min-edge` (default 5 %), `model_prob ≥ --min-prob`
  (20 %) and `odds ≤ --max-odds` (6.0).
- **stake** is a fractional Kelly stake (`--kelly 0.25`), capped at `--max-stake` (5 %) of
  the bankroll.
- `--shrink 0.3` blends 30 % of the market probability into the model's. This is a good idea
  while the model is young.
- `--all` shows every match side and `--out file.csv` saves the full table.

If you also record bet365 odds for matches before they are played, `padel-ml backtest`
replays the betting rules on held-out past matches using a model trained only on
earlier data, and compares the result with a pure-Elo strategy. **Don't bet real money
until the backtest is profitable over a few hundred bets.**

To plug in a legitimate odds feed later, subclass `padel_ml.odds.OddsProvider`.

## Configuration

These environment variables are optional:

| Variable | Default | Meaning |
|---|---|---|
| `PADEL_API_TOKEN` | – | Padel API token (required for `sync-*`) |
| `PADEL_API_RPM` | `10` | Padel API requests per minute (60 on paid plans) |
| `PADEL_ML_DB` | `data/padel.db` | database path |
| `PADEL_ML_DATA_DIR` | `data` | cache, models, reports |
| `PADEL_ML_DELAY` | `2.0` | seconds between requests to the same host |
| `PADEL_ML_USER_AGENT` | `padel_ml/0.1 …` | scraper user agent |
| `PADEL_ML_BANKROLL` / `PADEL_ML_KELLY` / `PADEL_ML_MIN_EDGE` / `PADEL_ML_MAX_STAKE` | 100 / 0.25 / 0.05 / 0.05 | betting rules |

## Project layout

```
src/padel_ml/
  db.py            SQLite schema and access
  scrapers/        Padel API client, CSV import, HTML scrapers, polite fetcher
  features.py      chronological Elo/form/h2h/ranking features
  model.py         training, validation, prediction
  odds/            odds sources (manual bet365 CSV; pluggable provider)
  betting.py       margin removal, EV, Kelly, backtest
  demo.py          synthetic data generator
  cli.py           `padel-ml` command
tests/             unit and end-to-end tests (no network needed)
```

## Disclaimer

This is a statistical tool, not financial advice. Bookmakers price padel well and
limit accounts that win consistently, and a model with little data is overconfident.
Bet only money you can afford to lose. In Spain gambling is for adults (18+) and is
regulated by the DGOJ. For help, see [jugarbien.es](https://www.jugarbien.es).
