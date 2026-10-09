"""
Retrieval Agent
===============
Executes discrete search queries across selected SerpApi surfaces
with Zero-Burn SQLite caching, budget-aware credit management,
and raw payload tracking for developer inspection.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.parse import urlparse
from pydantic import BaseModel, Field

from .cache import execute_search
from .planner import Plan
from .search_tools import (
    POP_CULTURE_KEYWORDS,
    extract_clean_domain,
    extract_rich_metadata,
    extract_serpapi_ai_overview,
    search_news,
    search_organic,
    search_scholar,
)

logger = logging.getLogger(__name__)


class RawEvidenceBundle(BaseModel):
    """Raw evidence gathered across search queries and surfaces."""

    raw_snippets: list[dict] = Field(default_factory=list)
    engines_used: list[str] = Field(default_factory=list)
    queries_executed: list[str] = Field(default_factory=list)
    knowledge_graph: dict[str, Any] = Field(default_factory=dict)
    related_queries: list[str] = Field(default_factory=list)
    raw_serpapi_payload: dict[str, Any] = Field(default_factory=dict)
    serpapi_ai_overview: str | None = None


class RetrievalAgent:
    """
    Executes search queries produced by the Planner Agent concurrently via ThreadPoolExecutor,
    handling single-surface routing (3 credits) or deep multi-surface routing (6 credits),
    strictly suppressing Google Scholar for pop culture/entertainment queries,
    and capturing SerpApi's native Google AI Overview.
    """

    def __init__(self, max_snippets_per_call: int = 5):
        self.max_snippets = max_snippets_per_call

    def retrieve(self, plan: Plan, credit_mode: str = "balanced") -> RawEvidenceBundle:
        """
        Execute queries concurrently across worker threads.
        - 'balanced' mode: Dispatches 1 targeted engine per query (exactly 3 API calls max).
        - 'deep' mode: Dispatches multiple engines per query (exhaustive corroboration).
        """
        all_snippets: list[dict] = []
        engines_used: set[str] = set()
        queries_executed: list[str] = []
        kg_data: dict[str, Any] = {}
        related_queries_list: list[str] = []
        raw_payloads: dict[str, Any] = {}
        native_ai_overview: str | None = None

        c_lower = plan.claim.lower()
        is_pop_culture = (
            any(k in c_lower for k in POP_CULTURE_KEYWORDS)
            or plan.claim_type == "entertainment_media"
        )
        is_scientific = plan.claim_type == "scientific" and not is_pop_culture

        # Build list of distinct search tasks
        tasks: list[dict[str, str]] = []
        for idx, query in enumerate(plan.queries):
            queries_executed.append(query)

            if credit_mode == "balanced":
                # Route each query to a dedicated optimal surface
                if idx == 0:
                    engine_name = "google_scholar" if is_scientific else "google"
                elif idx == 1:
                    engine_name = "google"
                else:
                    engine_name = "google_news" if not is_scientific else "google"

                engines_to_run = [engine_name]
            else:
                engines_to_run = [
                    e for e in plan.target_surfaces
                    if not (is_pop_culture and "scholar" in e)
                ]
                if not engines_to_run:
                    engines_to_run = ["google", "google_news"]

            for eng in engines_to_run:
                # Strict pop-culture guard: never route to scholar
                target_eng = "google" if (is_pop_culture and "scholar" in eng) else eng
                tasks.append({"query": query, "engine": target_eng})

        # Worker for ThreadPoolExecutor
        def _fetch_single(task: dict[str, str]) -> tuple[str, str, dict[str, Any]]:
            eng = task["engine"]
            q = task["query"]
            if eng in ("google_scholar", "scholar"):
                raw = search_scholar(q)
            elif eng in ("google_news", "news"):
                raw = search_news(q)
            else:
                raw = search_organic(q, num=self.max_snippets)
            return eng, q, raw

        # Execute all searches concurrently in parallel threads
        results_list: list[tuple[str, str, dict[str, Any]]] = []
        if tasks:
            with ThreadPoolExecutor(max_workers=min(len(tasks), 8)) as executor:
                future_to_task = {
                    executor.submit(_fetch_single, t): t for t in tasks
                }
                for future in as_completed(future_to_task):
                    t_info = future_to_task[future]
                    try:
                        eng, q, raw = future.result()
                        results_list.append((eng, q, raw))
                    except Exception as exc:
                        logger.warning(
                            "Parallel retrieval failed for engine '%s' on query '%s': %s",
                            t_info["engine"],
                            t_info["query"],
                            exc,
                        )

        # Process gathered search responses
        for eng, query, raw in results_list:
            raw_payloads[f"{eng}: {query}"] = raw

            # Harvester for SerpApi Native Google AI Overview
            if not native_ai_overview:
                native_ai_overview = extract_serpapi_ai_overview(raw)

            if eng in ("google_scholar", "scholar"):
                engines_used.add("google_scholar")
                items = raw.get("organic_results", [])
                for item in items[:self.max_snippets]:
                    pub_info = item.get("publication_info", {})
                    pub_summary = pub_info.get("summary", "") if isinstance(pub_info, dict) else ""
                    link_url = item.get("link", "")
                    all_snippets.append({
                        "source_engine": "google_scholar",
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", pub_summary),
                        "link": link_url,
                        "date": pub_summary if pub_summary else "Peer-reviewed",
                        "source_domain": extract_clean_domain(link_url) if link_url else "scholar.google.com",
                        "highlighted_words": item.get("snippet_highlighted_words", []),
                    })

            elif eng in ("google_news", "news"):
                engines_used.add("google_news")
                items = raw.get("news_results", [])
                for item in items[:self.max_snippets]:
                    link_url = item.get("link", "")
                    all_snippets.append({
                        "source_engine": "google_news",
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", ""),
                        "link": link_url,
                        "date": item.get("date", "Recent"),
                        "source_domain": extract_clean_domain(link_url),
                        "highlighted_words": item.get("snippet_highlighted_words", []),
                    })

            else:
                engines_used.add("google")
                meta = extract_rich_metadata({"organic": raw})
                if not kg_data and meta.get("knowledge_graph") and meta["knowledge_graph"].get("title"):
                    kg_data = meta["knowledge_graph"]
                for rq in meta.get("related_queries", []):
                    if rq not in related_queries_list:
                        related_queries_list.append(rq)

                items = raw.get("organic_results", [])
                for item in items[:self.max_snippets]:
                    link_url = item.get("link", "")
                    all_snippets.append({
                        "source_engine": "google",
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", ""),
                        "link": link_url,
                        "date": item.get("date", "Unknown date"),
                        "source_domain": extract_clean_domain(link_url),
                        "highlighted_words": item.get("snippet_highlighted_words", []),
                    })

        return RawEvidenceBundle(
            raw_snippets=all_snippets,
            engines_used=sorted(list(engines_used)),
            queries_executed=queries_executed,
            knowledge_graph=kg_data,
            related_queries=related_queries_list[:4],
            raw_serpapi_payload=raw_payloads,
            serpapi_ai_overview=native_ai_overview,
        )
