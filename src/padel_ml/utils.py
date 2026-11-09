"""Small helpers shared across modules."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


def normalize_name(name: str) -> str:
    """Canonical key for a player name: no accents, lowercase, single spaces.

    "Agustín TAPIA " -> "agustin tapia"; "Tapia, Agustín" -> "agustin tapia".
    """
    name = name.strip()
    if "," in name:  # "Surname, Name" -> "Name Surname"
        last, first = [p.strip() for p in name.split(",", 1)]
        name = f"{first} {last}"
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = re.sub(r"[^a-zA-Z0-9 \-']", " ", name).lower()
    return re.sub(r"\s+", " ", name).strip()


@dataclass(frozen=True)
class ParsedScore:
    sets_a: int
    sets_b: int
    games_a: int
    games_b: int
    retired: bool = False

    @property
    def winner(self) -> str | None:
        if self.sets_a > self.sets_b:
            return "A"
        if self.sets_b > self.sets_a:
            return "B"
        return None


_SET_RE = re.compile(r"(\d+)\s*[-/]\s*(\d+)(?:\s*\(\d+\))?")


def parse_score(score: str | None) -> ParsedScore | None:
    """Parse '6-4 3-6 7-6(5)' style scores (team A listed first)."""
    if not score:
        return None
    sets = _SET_RE.findall(score)
    if not sets:
        return None
    sa = sb = ga = gb = 0
    for a, b in sets:
        a, b = int(a), int(b)
        ga += a
        gb += b
        if a > b:
            sa += 1
        elif b > a:
            sb += 1
    retired = bool(re.search(r"\b(ret|retired|abandon|wo|w/o)\b", score, re.I))
    return ParsedScore(sa, sb, ga, gb, retired)
