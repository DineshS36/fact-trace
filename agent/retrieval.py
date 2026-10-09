"""
Retrieval Agent
===============
Executes discrete search queries across selected SerpApi surfaces
with Zero-Burn SQLite caching, budget-aware credit management,
and raw payload tracking for developer inspection.
"""

import logging
from typing import Any
from urllib.parse import urlparse
from pydantic import BaseModel, Field

from .cache import execute_search
from .planner import Plan
from .search_tools import (
    extract_clean_domain,
    extract_rich_metadata,
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


class RetrievalAgent:
    """
    Executes search queries produced by the Planner Agent without concatenating,
    handling single-surface routing (3 credits) or deep multi-surface routing (6 credits).
    """

    def __init__(self, max_snippets_per_call: int = 5):
        self.max_snippets = max_snippets_per_call

    def retrieve(self, plan: Plan, credit_mode: str = "balanced") -> RawEvidenceBundle:
        """
        Execute queries.
        - 'balanced' mode: Dispatches 1 targeted engine per query (exactly 3 API calls max).
        - 'deep' mode: Dispatches multiple engines per query (exhaustive corroboration).
        """
        all_snippets: list[dict] = []
        engines_used: set[str] = set()
        queries_executed: list[str] = []
        kg_data: dict[str, Any] = {}
        related_queries_list: list[str] = []
        raw_payloads: dict[str, Any] = {}

        is_scientific = plan.claim_type == "scientific"

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
                engines_to_run = plan.target_surfaces

            for eng in engines_to_run:
                try:
                    if eng in ("google_scholar", "scholar"):
                        raw = search_scholar(query)
                        raw_payloads[f"scholar: {query[:40]}"] = {
                            "search_metadata": raw.get("search_metadata", {}),
                            "results_count": len(raw.get("organic_results", [])),
                            "sample_result": raw.get("organic_results", [{}])[0] if raw.get("organic_results") else {},
                        }
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
                        raw = search_news(query)
                        raw_payloads[f"news: {query[:40]}"] = {
                            "search_metadata": raw.get("search_metadata", {}),
                            "results_count": len(raw.get("news_results", [])),
                            "sample_result": raw.get("news_results", [{}])[0] if raw.get("news_results") else {},
                        }
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
                        raw = search_organic(query, num=self.max_snippets)
                        raw_payloads[f"google: {query[:40]}"] = {
                            "search_metadata": raw.get("search_metadata", {}),
                            "results_count": len(raw.get("organic_results", [])),
                            "sample_result": raw.get("organic_results", [{}])[0] if raw.get("organic_results") else {},
                        }
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

                except Exception as exc:
                    logger.warning("Retrieval failed for engine '%s' on query '%s': %s", eng, query, exc)

        return RawEvidenceBundle(
            raw_snippets=all_snippets,
            engines_used=sorted(list(engines_used)),
            queries_executed=queries_executed,
            knowledge_graph=kg_data,
            related_queries=related_queries_list[:4],
            raw_serpapi_payload=raw_payloads,
        )
