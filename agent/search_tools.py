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
from urllib.parse import urlparse

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


def search_all_engines(
    query: str,
    engines_to_query: list[str] | None = None,
    num_organic: int = 5,
) -> dict[str, Any]:
    """
    Run the query against selected engines and return an evidence bundle.
    If engines_to_query is None, queries all engines ("organic", "news", "scholar").
    """
    results: dict[str, Any] = {}

    available = {
        "organic": lambda: search_organic(query, num=num_organic),
        "google": lambda: search_organic(query, num=num_organic),
        "news": lambda: search_news(query),
        "google_news": lambda: search_news(query),
        "scholar": lambda: search_scholar(query),
        "google_scholar": lambda: search_scholar(query),
    }

    # Normalize target engine names
    targets = engines_to_query or ["organic", "news", "scholar"]
    chosen_engines = set()
    for t in targets:
        normalized = t.lower()
        if normalized in ("organic", "google"):
            chosen_engines.add("organic")
        elif normalized in ("news", "google_news"):
            chosen_engines.add("news")
        elif normalized in ("scholar", "google_scholar"):
            chosen_engines.add("scholar")

    if not chosen_engines:
        chosen_engines = {"organic"}

    for name in chosen_engines:
        fetch = available[name]
        try:
            results[name] = fetch()
        except Exception as exc:
            logger.warning("Engine '%s' failed for query '%s': %s", name, query, exc)
            results[name] = {"error": str(exc)}

    return results


# ── Tiered Domain Authority Filter ──────────────────────────

# Social forum & low-signal domains to drop completely
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

# Explicitly elevated authority clusters
TIER_1_DOMAINS = {
    # Government & Scientific Indexers
    "nih.gov", "ncbi.nlm.nih.gov", "cdc.gov", "fda.gov", "who.int", "nature.com",
    "sciencedirect.com", "thelancet.com", "jamanetwork.com", "cell.com", "arxiv.org",
    # Primary News Wires & Financial Records
    "reuters.com", "apnews.com", "bloomberg.com", "sec.gov", "nasa.gov", "weather.gov"
}

TIER_2_DOMAINS = {
    "bbc.com", "nytimes.com", "washingtonpost.com", "wsj.com", "theguardian.com",
    "thehindu.com", "economist.com", "ft.com", "scientificamerican.com"
}


def extract_clean_domain(link: str) -> str:
    """Extracts a clean, canonical domain name from any URL, discarding breadcrumb/view noise."""
    if not link:
        return "web"
    try:
        netloc = urlparse(link).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        if ":" in netloc:
            netloc = netloc.split(":")[0]
        return netloc if netloc else "web"
    except Exception:
        return "web"


def calculate_domain_authority(url: str, is_scholar: bool = False) -> tuple[int, str]:
    """Returns (tier_score: 1-3, label: 'High' | 'Medium' | 'Low')."""
    if not url:
        if is_scholar:
            return (1, "Tier-1 Authoritative (Peer-Reviewed Scholar)")
        return (3, "Standard Web")

    try:
        domain = urlparse(url).netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        if ":" in domain:
            domain = domain.split(":")[0]
    except Exception:
        domain = ""

    if (
        any(domain.endswith(t1) or domain == t1 for t1 in TIER_1_DOMAINS)
        or domain.endswith(".gov")
        or domain.endswith(".edu")
        or is_scholar
    ):
        label = (
            "Tier-1 Authoritative (Peer-Reviewed Scholar)"
            if is_scholar and not any(domain.endswith(t1) or domain == t1 for t1 in TIER_1_DOMAINS) and not (domain.endswith(".gov") or domain.endswith(".edu"))
            else "Tier-1 Authoritative (Gov/Academia/Wire)"
        )
        return (1, label)
    elif any(domain.endswith(t2) or domain == t2 for t2 in TIER_2_DOMAINS):
        return (2, "Tier-2 Reputable (Major Institutional Press)")
    return (3, "Standard Web")


def filter_and_rank_sources(results: list[dict], top_k: int = 5, is_scholar: bool = False) -> list[dict]:
    """Sorts results so high-authority domains are prioritized at the top of context."""
    valid_results = []
    for r in results:
        link = r.get("link", "")
        # Drop social/forum noise
        if any(banned in link.lower() for banned in BANNED_DOMAINS):
            continue
        tier, label = calculate_domain_authority(link, is_scholar=is_scholar)
        r_copy = dict(r)
        r_copy["authority_tier"] = tier
        r_copy["authority_label"] = label
        valid_results.append(r_copy)

    # Sort primarily by Tier (Tier 1 first), secondarily keeping original relevance
    valid_results.sort(key=lambda x: x["authority_tier"])
    return valid_results[:top_k]


def filter_reputable_sources(results: list[dict], max_sources: int = 5) -> list[dict]:
    """Filter out low-signal/social forum domains and drop noisy sources (backward-compatible wrapper)."""
    return filter_and_rank_sources(results, top_k=max_sources)


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
    filtered for domain authority and enriched with SerpApi date, highlights, and source.
    High-authority sources (Tier 1/2) are ranked first.
    """
    snippets: list[dict] = []

    # Organic results (filtered & ranked by authority)
    organic = raw_results.get("organic", {})
    organic_items = filter_and_rank_sources(
        organic.get("organic_results", []),
        top_k=max_per_engine,
        is_scholar=False,
    )
    for item in organic_items:
        link_url = item.get("link", "")
        snippets.append({
            "source_engine": "google",
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "link": link_url,
            "date": item.get("date", "Unknown date"),
            "source_domain": extract_clean_domain(link_url),
            "authority_tier": item.get("authority_tier", 3),
            "authority_label": item.get("authority_label", "Standard Web"),
            "highlighted_words": item.get("snippet_highlighted_words", []),
        })

    # News results (filtered & ranked by authority)
    news = raw_results.get("news", {})
    news_items = filter_and_rank_sources(
        news.get("news_results", []),
        top_k=max_per_engine,
        is_scholar=False,
    )
    for item in news_items:
        link_url = item.get("link", "")
        snippets.append({
            "source_engine": "google_news",
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "link": link_url,
            "date": item.get("date", "Recent"),
            "source_domain": extract_clean_domain(link_url),
            "authority_tier": item.get("authority_tier", 3),
            "authority_label": item.get("authority_label", "Standard Web"),
            "highlighted_words": item.get("snippet_highlighted_words", []),
        })

    # Scholar results (peer-reviewed / academic, elevated as Tier-1 authority)
    scholar = raw_results.get("scholar", {})
    scholar_items = filter_and_rank_sources(
        scholar.get("organic_results", []),
        top_k=max_per_engine,
        is_scholar=True,
    )
    for item in scholar_items:
        pub_info = item.get("publication_info", {})
        pub_summary = pub_info.get("summary", "") if isinstance(pub_info, dict) else ""
        snippets.append({
            "source_engine": "google_scholar",
            "title": item.get("title", ""),
            "snippet": item.get("snippet", pub_summary),
            "link": item.get("link", ""),
            "date": pub_summary if pub_summary else "Peer-reviewed",
            "source_domain": "Google Scholar / Academic",
            "authority_tier": item.get("authority_tier", 1),
            "authority_label": item.get("authority_label", "Tier-1 Authoritative (Peer-Reviewed Scholar)"),
            "highlighted_words": item.get("snippet_highlighted_words", []),
        })

    return snippets
