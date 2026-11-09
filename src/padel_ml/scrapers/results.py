"""Match-results scraper for tournament draw/result pages (Premier Padel, FIP Tour, ...).

Pass the URL of a tournament's results/draw page. The parser looks for repeated
"match" blocks that contain two teams of two players and a per-set score, which is
how both premierpadel.com and padelfip.com render their draws. Markup is not stable
across seasons, so the selectors are kept in one place (`MATCH_SELECTORS`, ...).

Pages that build the draw with JavaScript after load will return no matches with
plain HTTP; for those, export the results to CSV and use `padel-ml import-matches`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Tag

from ..db import Database
from .base import Fetcher

log = logging.getLogger(__name__)

MATCH_SELECTORS = ["[class*=match]", "[class*=partido]", "[class*=game-card]"]
TEAM_SELECTORS = ["[class*=team]", "[class*=pair]", "[class*=couple]", "[class*=pareja]"]
NAME_SELECTORS = ["[class*=name]", "[class*=player]", "a"]
SCORE_SELECTORS = ["[class*=score]", "[class*=set]", "[class*=result]"]
DATE_ATTRS = ["data-date", "datetime"]

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})|(\d{2})[/.](\d{2})[/.](\d{4})")


@dataclass
class ScrapedMatch:
    team_a: tuple[str, str]
    team_b: tuple[str, str]
    games_a: list[int] = field(default_factory=list)
    games_b: list[int] = field(default_factory=list)
    date: str | None = None
    round: str | None = None

    @property
    def score(self) -> str | None:
        if not self.games_a or len(self.games_a) != len(self.games_b):
            return None
        return " ".join(f"{a}-{b}" for a, b in zip(self.games_a, self.games_b))


def _first(el: Tag, selectors: list[str]) -> list[Tag]:
    for s in selectors:
        found = el.select(s)
        if found:
            return found
    return []


def _names(team: Tag) -> list[str]:
    names: list[str] = []
    for el in _first(team, NAME_SELECTORS):
        t = el.get_text(" ", strip=True)
        if t and sum(c.isalpha() for c in t) >= 3 and t not in names and not el.select(
                ",".join(NAME_SELECTORS)):
            names.append(t)
    if len(names) < 2:  # fallback: text lines with letters
        names = [s for s in team.stripped_strings if sum(c.isalpha() for c in s) >= 3][:2]
    return names[:2]


def _games(team: Tag) -> list[int]:
    out = []
    for el in _first(team, SCORE_SELECTORS):
        if el.select(",".join(SCORE_SELECTORS)):
            continue  # container, not a leaf score cell
        t = el.get_text(strip=True)
        m = re.match(r"^(\d{1,2})", t)
        if m:
            out.append(int(m.group(1)))
    return out


def _parse_date(el: Tag) -> str | None:
    for node in [el, *el.parents]:
        if not isinstance(node, Tag):
            continue
        for attr in DATE_ATTRS:
            v = node.get(attr)
            if v:
                m = _DATE_RE.search(v)
                if m:
                    return m.group(1) or f"{m.group(4)}-{m.group(3)}-{m.group(2)}"
    return None


def parse_results_html(html: str) -> list[ScrapedMatch]:
    soup = BeautifulSoup(html, "lxml")
    blocks: list[Tag] = []
    for sel in MATCH_SELECTORS:
        blocks = [b for b in soup.select(sel) if len(_first(b, TEAM_SELECTORS)) >= 2]
        # keep only innermost blocks (drop wrappers that contain another match block)
        ids = {id(b) for b in blocks}
        blocks = [b for b in blocks
                  if not any(id(d) in ids for d in b.find_all(True))]
        if blocks:
            break
    out: list[ScrapedMatch] = []
    seen: set[tuple] = set()
    for b in blocks:
        teams = [t for t in _first(b, TEAM_SELECTORS)
                 if not t.select(",".join(TEAM_SELECTORS))][:2]
        if len(teams) < 2:
            continue
        na, nb = _names(teams[0]), _names(teams[1])
        if len(na) < 2 or len(nb) < 2:
            continue
        key = (tuple(na), tuple(nb))
        if key in seen:
            continue
        seen.add(key)
        rnd = b.get("data-round")
        out.append(ScrapedMatch((na[0], na[1]), (nb[0], nb[1]), _games(teams[0]),
                                _games(teams[1]), _parse_date(b), rnd))
    return out


def scrape_results(db: Database, url: str, tournament: str, date_fallback: str,
                   category: str | None = None, gender: str | None = None,
                   fetcher: Fetcher | None = None, source: str = "web") -> int:
    """Scrape one tournament page and store its matches.

    Matches without a score are stored as upcoming fixtures.
    """
    fetcher = fetcher or Fetcher()
    matches = parse_results_html(fetcher.get(url))
    if not matches:
        log.warning("no matches parsed from %s (JS-rendered page or new markup?)", url)
    for m in matches:
        db.upsert_match(
            date=m.date or date_fallback, team_a=m.team_a, team_b=m.team_b, score=m.score,
            tournament=tournament, category=category, round=m.round, gender=gender,
            source=source,
        )
    return len(matches)
