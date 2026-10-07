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


# ── Domain Authority Filter ─────────────────────────────────

BANNED_DOMAINS = {
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "quora.com",
    "reddit.com",
    "pinterest.com",
    "twitter.com",
    "x.com",
}


def filter_reputable_sources(results: list[dict], max_sources: int = 5) -> list[dict]:
    """Filter out low-signal/social forum domains and drop noisy sources."""
    curated = []
    for item in results:
        link = (item.get("link") or "").lower()
        if any(banned in link for banned in BANNED_DOMAINS):
            continue
        curated.append(item)
        if len(curated) >= max_sources:
            break
    return curated


# ── Evidence Extraction Helpers ─────────────────────────────


def extract_rich_metadata(raw_results: dict[str, Any]) -> dict[str, Any]:
    """
    Extract high-value SerpApi metadata across surfaces:
    - knowledge_graph entity info
    - related_questions (investigation angles)
    """
    organic_data = raw_results.get("organic", {})
    knowledge_graph = organic_data.get("knowledge_graph", {})
    
    # Extract clean knowledge graph info if present
    kg_entity = {}
    if knowledge_graph:
        kg_entity = {
            "title": knowledge_graph.get("title", ""),
            "type": knowledge_graph.get("type", ""),
            "description": knowledge_graph.get("description", ""),
            "source": knowledge_graph.get("source", {}).get("name", "") if isinstance(knowledge_graph.get("source"), dict) else "",
        }

    # Extract related questions / angles
    related_questions_raw = organic_data.get("related_questions", [])
    related_queries = []
    for q in related_questions_raw[:3]:
        question_text = q.get("question")
        if question_text and question_text not in related_queries:
            related_queries.append(question_text)

    return {
        "knowledge_graph": kg_entity,
        "related_queries": related_queries,
    }


def extract_snippets(raw_results: dict[str, Any], max_per_engine: int = 5) -> list[dict]:
    """
    Flatten multi-engine results into a curated list of evidence snippets
    filtered for domain authority and enriched with SerpApi metadata.
    """
    snippets: list[dict] = []

    # Organic results (filtered)
    organic = raw_results.get("organic", {})
    organic_items = filter_reputable_sources(
        organic.get("organic_results", []),
        max_sources=max_per_engine
    )
    for item in organic_items:
        snippets.append({
            "source_engine": "google",
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "link": item.get("link", ""),
            "highlighted_words": item.get("snippet_highlighted_words", []),
        })

    # News results (filtered)
    news = raw_results.get("news", {})
    news_items = filter_reputable_sources(
        news.get("news_results", []),
        max_sources=max_per_engine
    )
    for item in news_items:
        snippets.append({
            "source_engine": "google_news",
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "link": item.get("link", ""),
            "highlighted_words": item.get("snippet_highlighted_words", []),
        })

    # Scholar results (peer-reviewed / academic)
    scholar = raw_results.get("scholar", {})
    scholar_items = filter_reputable_sources(
        scholar.get("organic_results", []),
        max_sources=max_per_engine
    )
    for item in scholar_items:
        snippets.append({
            "source_engine": "google_scholar",
            "title": item.get("title", ""),
            "snippet": item.get("snippet", item.get("publication_info", {}).get("summary", "")),
            "link": item.get("link", ""),
            "highlighted_words": item.get("snippet_highlighted_words", []),
        })

    return snippets
