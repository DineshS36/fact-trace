"""
Multi-Engine Search Dispatcher
===============================
Dispatches fact-checking queries across multiple SerpApi surfaces
(Google organic, Google News, Google Scholar) to gather diverse
evidence from different source types.

All calls route through the cache layer — zero credit burn during dev.
"""

import logging
from typing import Any

from .cache import execute_search

logger = logging.getLogger(__name__)


# ── Individual Engine Wrappers ──────────────────────────────


def search_organic(query: str, num: int = 5) -> dict[str, Any]:
    """
    Standard Google web search.
    Returns top ``num`` organic results.
    """
    return execute_search({"engine": "google", "q": query, "num": num})


def search_news(query: str) -> dict[str, Any]:
    """
    Google News search — surfaces recent press coverage and reporting.
    """
    return execute_search({"engine": "google_news", "q": query})


def search_scholar(query: str) -> dict[str, Any]:
    """
    Google Scholar search — surfaces academic papers, citations,
    and institutional publications for evidence grounding.
    """
    return execute_search({"engine": "google_scholar", "q": query})


# ── Aggregated Multi-Engine Search ──────────────────────────


def search_all_engines(query: str, num_organic: int = 5) -> dict[str, Any]:
    """
    Run the query against all three engines and return a unified
    evidence bundle.

    Returns:
        {
            "organic": { ... },
            "news":    { ... },
            "scholar": { ... },
        }

    Each value is the raw SerpApi response dict for that engine.
    Failures on individual engines are captured as error strings
    rather than crashing the whole pipeline.
    """
    results: dict[str, Any] = {}

    engines = [
        ("organic", lambda: search_organic(query, num=num_organic)),
        ("news", lambda: search_news(query)),
        ("scholar", lambda: search_scholar(query)),
    ]

    for name, fetch in engines:
        try:
            results[name] = fetch()
        except Exception as exc:
            logger.warning("Engine '%s' failed for query '%s': %s", name, query, exc)
            results[name] = {"error": str(exc)}

    return results


# ── Evidence Extraction Helpers ─────────────────────────────


def extract_snippets(raw_results: dict[str, Any], max_per_engine: int = 5) -> list[dict]:
    """
    Flatten multi-engine results into a list of evidence snippets
    suitable for LLM consumption.

    Each snippet dict contains:
        - source_engine: str
        - title: str
        - snippet: str
        - link: str
    """
    snippets: list[dict] = []

    # Organic results
    organic = raw_results.get("organic", {})
    for item in organic.get("organic_results", [])[:max_per_engine]:
        snippets.append({
            "source_engine": "google",
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "link": item.get("link", ""),
        })

    # News results
    news = raw_results.get("news", {})
    for item in news.get("news_results", [])[:max_per_engine]:
        snippets.append({
            "source_engine": "google_news",
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "link": item.get("link", ""),
        })

    # Scholar results
    scholar = raw_results.get("scholar", {})
    for item in scholar.get("organic_results", [])[:max_per_engine]:
        snippets.append({
            "source_engine": "google_scholar",
            "title": item.get("title", ""),
            "snippet": item.get("snippet", item.get("publication_info", {}).get("summary", "")),
            "link": item.get("link", ""),
        })

    return snippets
