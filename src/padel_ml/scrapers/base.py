"""Polite HTTP fetching: robots.txt, rate limiting, on-disk cache, retries."""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from urllib import robotparser
from urllib.parse import urlparse

import requests

from ..config import Settings, get_settings

log = logging.getLogger(__name__)


class RobotsDisallowed(RuntimeError):
    pass


class Fetcher:
    def __init__(self, settings: Settings | None = None, use_cache: bool = True):
        self.s = settings or get_settings()
        self.use_cache = use_cache
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.s.user_agent,
            "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
        })
        self._robots: dict[str, robotparser.RobotFileParser | None] = {}
        self._last_request: dict[str, float] = {}

    # -------------------------------------------------------------- robots
    def allowed(self, url: str) -> bool:
        parts = urlparse(url)
        base = f"{parts.scheme}://{parts.netloc}"
        if base not in self._robots:
            rp = robotparser.RobotFileParser()
            try:
                r = self.session.get(base + "/robots.txt", timeout=self.s.request_timeout_s)
                if r.status_code >= 400:
                    rp = None  # no robots.txt -> allowed
                else:
                    rp.parse(r.text.splitlines())
            except requests.RequestException as e:
                log.warning("could not read robots.txt for %s: %s", base, e)
                rp = None
            self._robots[base] = rp
        rp = self._robots[base]
        return True if rp is None else rp.can_fetch(self.s.user_agent, url)

    # -------------------------------------------------------------- cache
    def _cache_path(self, url: str) -> Path:
        h = hashlib.sha256(url.encode()).hexdigest()[:32]
        return self.s.cache_dir / f"{h}.html"

    def _cached(self, url: str) -> str | None:
        p = self._cache_path(url)
        if self.use_cache and p.exists():
            age_h = (time.time() - p.stat().st_mtime) / 3600
            if age_h < self.s.cache_ttl_hours:
                return p.read_text(encoding="utf-8")
        return None

    # -------------------------------------------------------------- fetch
    def get(self, url: str, retries: int = 3) -> str:
        cached = self._cached(url)
        if cached is not None:
            return cached
        if not self.allowed(url):
            raise RobotsDisallowed(f"robots.txt disallows {url}")
        host = urlparse(url).netloc
        wait = self.s.request_delay_s - (time.time() - self._last_request.get(host, 0))
        if wait > 0:
            time.sleep(wait)
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                r = self.session.get(url, timeout=self.s.request_timeout_s)
                self._last_request[host] = time.time()
                if r.status_code == 429 or r.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
                r.raise_for_status()
                p = self._cache_path(url)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(r.text, encoding="utf-8")
                return r.text
            except requests.RequestException as e:
                last_exc = e
                backoff = 2 ** (attempt + 1)
                log.warning("GET %s failed (%s); retrying in %ss", url, e, backoff)
                time.sleep(backoff)
        raise RuntimeError(f"failed to fetch {url}") from last_exc
