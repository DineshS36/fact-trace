"""
Zero-Burn Cache Layer
=====================
Every SerpApi call is routed through an SQLite cache so that
frontend tweaking, agent-loop debugging, and test runs never
burn live API credits.

Design notes:
- Thread-safe: each call opens its own short-lived connection.
- Deterministic keys: params are JSON-serialised with sorted keys.
- Transparent: callers get the same dict whether cached or live.
"""

import json
import logging
import os
import sqlite3
from pathlib import Path

from serpapi import GoogleSearch

logger = logging.getLogger(__name__)

DB_PATH: str = os.getenv("SERP_CACHE_DB", "serp_cache.db")

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS cache (
    cache_key  TEXT PRIMARY KEY,
    data       TEXT    NOT NULL,
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
)
"""


def _get_connection() -> sqlite3.Connection:
    """Return a new connection with WAL mode for concurrent reads."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_SCHEMA_SQL)
    return conn


def _make_key(params: dict) -> str:
    """Deterministic cache key from search parameters."""
    # Strip the api_key so the cache key is user-agnostic
    filtered = {k: v for k, v in params.items() if k != "api_key"}
    return json.dumps(filtered, sort_keys=True)


def execute_search(params: dict) -> dict:
    """
    Execute a SerpApi search with transparent caching.

    On cache hit  → returns stored result, zero API cost.
    On cache miss → fetches live, stores result, returns it.

    Args:
        params: SerpApi search parameters (engine, q, num, etc.).
                The ``api_key`` is injected from the environment.

    Returns:
        Raw dict from SerpApi.

    Raises:
        RuntimeError: If SERPAPI_API_KEY is unset and cache misses.
    """
    key = _make_key(params)

    # ── Cache lookup ────────────────────────────────────────
    with _get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT data FROM cache WHERE cache_key = ?", (key,))
        row = cur.fetchone()
        if row:
            logger.debug("Cache HIT for key=%s", key[:80])
            return json.loads(row[0])

    # ── Cache miss → live fetch ─────────────────────────────
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "SERPAPI_API_KEY is not set and no cached result exists "
            "for this query. Set the key in your .env file."
        )

    logger.info("Cache MISS — fetching live from SerpApi: %s", key[:80])
    live_params = {**params, "api_key": api_key}
    search = GoogleSearch(live_params)
    results: dict = search.get_dict()

    # Persist for future runs
    with _get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO cache (cache_key, data) VALUES (?, ?)",
            (key, json.dumps(results)),
        )

    return results


def clear_cache() -> int:
    """
    Drop all cached entries. Returns the number of rows deleted.
    Useful for integration-test teardown or manual refresh.
    """
    with _get_connection() as conn:
        cur = conn.execute("DELETE FROM cache")
        deleted = cur.rowcount
    logger.info("Cache cleared: %d entries removed", deleted)
    return deleted


def cache_stats() -> dict:
    """Return basic cache statistics for observability."""
    with _get_connection() as conn:
        cur = conn.execute("SELECT COUNT(*), SUM(LENGTH(data)) FROM cache")
        count, total_bytes = cur.fetchone()
    return {
        "entries": count or 0,
        "total_bytes": total_bytes or 0,
        "db_path": str(Path(DB_PATH).resolve()),
    }
