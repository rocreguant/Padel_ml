"""International Padel Federation (FIP) rankings scraper - padelfip.com.

The official FIP ranking (which also governs Premier Padel seeding) is published as
an HTML page. Site markup changes from time to time, so the parser is deliberately
tolerant: it first looks for a <table>, then for repeated "card" elements, and
extracts position / name / country / points from each row.

If the parser stops finding rows, open the page in a browser, inspect a ranking row
and adjust `ROW_SELECTORS` / `parse_ranking_html` - the rest of the pipeline does not
need to change.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date

from bs4 import BeautifulSoup, Tag

from ..db import Database
from .base import Fetcher

log = logging.getLogger(__name__)

RANKING_URLS = {
    "M": "https://www.padelfip.com/ranking-male/",
    "F": "https://www.padelfip.com/ranking-female/",
}

# CSS selectors tried in order for "card" style layouts.
ROW_SELECTORS = [
    "[class*=ranking] [class*=player]",
    "[class*=player-ranking]",
    "[class*=ranking-item]",
    "[class*=slider__item]",
    "li[class*=player]",
]

_NUM_RE = re.compile(r"\d[\d.,]*")
_COUNTRY_RE = re.compile(r"^[A-Z]{3}$")


@dataclass
class RankingRow:
    position: int
    name: str
    points: float | None = None
    country: str | None = None


def _to_float(s: str) -> float | None:
    """'14.230' / '14,230' -> 14230 (thousands separators); '12,5' -> 12.5."""
    s = s.strip()
    if re.fullmatch(r"\d{1,3}([.,]\d{3})+", s):
        s = s.replace(".", "").replace(",", "")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _row_from_cells(cells: list[str]) -> RankingRow | None:
    cells = [c.strip() for c in cells if c and c.strip()]
    if len(cells) < 2:
        return None
    pos = None
    if re.fullmatch(r"\d{1,4}", cells[0]):
        pos = int(cells[0])
    country = next((c for c in cells if _COUNTRY_RE.match(c)), None)
    # name: longest cell made mostly of letters
    letter_cells = [c for c in cells if sum(ch.isalpha() for ch in c) >= max(3, 0.6 * len(c))
                    and c != country]
    if pos is None or not letter_cells:
        return None
    name = max(letter_cells, key=len)
    points = None
    for c in reversed(cells):
        if c in (cells[0], name, country):
            continue
        m = _NUM_RE.fullmatch(c.replace(" ", "").replace("pts", "").replace("Pts", ""))
        if m:
            points = _to_float(m.group())
            break
    return RankingRow(pos, name, points, country)


def _cell_texts(el: Tag) -> list[str]:
    return [t for t in el.stripped_strings]


def parse_ranking_html(html: str) -> list[RankingRow]:
    soup = BeautifulSoup(html, "lxml")
    rows: list[RankingRow] = []

    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
            r = _row_from_cells(cells)
            if r:
                rows.append(r)
    if not rows:
        for sel in ROW_SELECTORS:
            for el in soup.select(sel):
                r = _row_from_cells(_cell_texts(el))
                if r:
                    rows.append(r)
            if rows:
                break

    # de-duplicate by position keeping first occurrence
    seen, out = set(), []
    for r in rows:
        if r.position not in seen:
            seen.add(r.position)
            out.append(r)
    return sorted(out, key=lambda r: r.position)


def scrape_rankings(db: Database, fetcher: Fetcher | None = None,
                    genders: tuple[str, ...] = ("M", "F")) -> int:
    fetcher = fetcher or Fetcher()
    today = date.today().isoformat()
    total = 0
    for g in genders:
        url = RANKING_URLS[g]
        try:
            html = fetcher.get(url)
        except Exception as e:  # noqa: BLE001 - keep scraping other sources
            log.error("FIP ranking %s failed: %s", g, e)
            continue
        rows = parse_ranking_html(html)
        if not rows:
            log.warning("FIP ranking %s: no rows parsed - markup may have changed (%s)", g, url)
        for r in rows:
            pid = db.upsert_player(r.name, gender=g, country=r.country, source="fip")
            db.add_ranking(pid, today, r.position, r.points, "fip")
            total += 1
        log.info("FIP ranking %s: %d players", g, len(rows))
    return total
