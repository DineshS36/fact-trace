"""
Core Fact-Verification Agent
=============================
Orchestrates multi-engine evidence retrieval, tiered domain authority filtering,
LLM query decomposition with search operator injection, and AI Overview synthesis
with grounded citations ([1], [2]).

Architecture:
    Claim → Query Decomposer → Multi-Engine Search (cached) → Tier-1/2 Ranking → AI Overview → VerificationResult

Supported LLM backends (set via LLM_PROVIDER env var):
    - "gemini"  → Google Gemini API  (default)
    - "openai"  → OpenAI API
    - "groq"    → Groq API
"""

import json
import logging
import os
from enum import Enum
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .search_tools import (
    calculate_domain_authority,
    extract_rich_metadata,
    extract_snippets,
    filter_and_rank_sources,
    search_all_engines,
)

load_dotenv()
logger = logging.getLogger(__name__)


# ── Data Models ─────────────────────────────────────────────


class Verdict(str, Enum):
    """Discrete verdicts the agent can assign to a claim."""

    TRUE = "True"
    MOSTLY_TRUE = "Mostly True"
    HALF_TRUE = "Half True"
    CONTESTED = "Contested"
    MOSTLY_FALSE = "Mostly False"
    FALSE = "False"
    UNVERIFIABLE = "Unverifiable"


class SourceEvidence(BaseModel):
    """A single piece of evidence used in the verdict."""

    index: int = 1
    title: str
    snippet: str
    url: str
    source_engine: str
    stance: str = Field(
        default="neutral",
        description="How this source relates to the claim: supports | refutes | neutral",
    )
    date: str = "Unknown date"
    source_domain: str = ""
    authority_tier: int = 3
    authority_label: str = "Standard Web"
    highlighted_words: list[str] = Field(default_factory=list)


class KnowledgeGraphEntity(BaseModel):
    """Deep SerpApi knowledge graph entity if resolved."""

    title: str = ""
    type: str = ""
    description: str = ""
    source: str = ""


class VerificationResult(BaseModel):
    """Structured output of a fact-verification run."""

    claim: str
    verdict: Verdict
    confidence_score: float = Field(ge=0.0, le=1.0)
    explanation: str
    executive_overview: str = ""
    consensus_summary: str = ""
    divergence_detected: bool = False
    divergence_notes: str = ""
    evidence: list[SourceEvidence] = Field(default_factory=list)
    search_queries_used: list[str] = Field(default_factory=list)
    engines_queried: list[str] = Field(default_factory=list)
    knowledge_graph: KnowledgeGraphEntity | None = None
    related_queries: list[str] = Field(default_factory=list)


# ── LLM Client Factory ─────────────────────────────────────

_DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-3.5-flash-lite",
}


def _call_llm(
    system_prompt: str,
    user_prompt: str = "",
    response_mime_type: str = "application/json",
) -> str:
    """
    Route prompt to the configured LLM provider and return raw response text.
    Handles single-prompt and system+user prompts across Gemini, OpenAI, and Groq.

    Raises:
        RuntimeError: If provider is unsupported or API key is missing.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower().strip()
    model = os.getenv(
        f"{provider.upper()}_MODEL", _DEFAULT_MODELS.get(provider, "")
    )

    if provider == "openai":
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in .env")
        client = OpenAI(api_key=api_key)
        messages = []
        if system_prompt and user_prompt:
            messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
        else:
            messages.append({"role": "user", "content": system_prompt or user_prompt})

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
        }
        if response_mime_type == "application/json":
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    if provider == "groq":
        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set in .env")
        client = Groq(api_key=api_key)
        messages = []
        if system_prompt and user_prompt:
            messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
        else:
            messages.append({"role": "user", "content": system_prompt or user_prompt})

        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
        }
        if response_mime_type == "application/json":
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    if provider == "gemini":
        from google import genai
        from google.genai import types

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in .env")
        client = genai.Client(api_key=api_key)
        contents = f"{system_prompt}\n\n{user_prompt}".strip() if user_prompt else system_prompt

        candidate_models = [model]
        for fallback_mod in ("gemini-3.5-flash-lite", "gemini-3.8-flash"):
            if fallback_mod not in candidate_models:
                candidate_models.append(fallback_mod)

        last_exc = None
        for mod in candidate_models:
            try:
                resp = client.models.generate_content(
                    model=mod,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        response_mime_type=response_mime_type,
                    ),
                )
                if resp.text:
                    return resp.text
            except Exception as exc:
                last_exc = exc
                logger.warning("Gemini model '%s' failed (%s), trying fallback...", mod, exc)
                continue

        if last_exc:
            raise last_exc
        return ""

    raise RuntimeError(
        f"Unsupported LLM_PROVIDER='{provider}'. Use 'openai', 'groq', or 'gemini'."
    )


# ── Query Decomposer with Search Operator Injection ─────────


def generate_targeted_queries(claim: str, claim_type: str = "general") -> list[str]:
    """
    Converts a claim into 3 distinct, high-precision Google search queries equipped
    with search operators and targeted keywords rather than naive string appending.
    """
    prompt = f"""You are a Principal Search Engineer. Convert the following claim into 3 distinct, high-precision Google search queries designed to find authoritative, primary documentation.

Claim: "{claim}"
Type: {claim_type}

Rules:
1. Query 1 must target primary institutions or high-authority bodies using targeted keywords (e.g., using terms like "clinical trial", "statute", "official release", or site operators).
2. Query 2 must target scientific consensus, meta-analyses, or official reports.
3. Query 3 must search for the original counter-evidence, origin of the myth, or debunking reporting.
4. Do NOT use overly restrictive operators if they risk zero results, but prioritize precise terminology over conversational sentences.

Return ONLY a valid JSON array of 3 strings: ["query 1", "query 2", "query 3"]
"""
    try:
        response = _call_llm(prompt, response_mime_type="application/json")
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        parsed = json.loads(cleaned)
        if isinstance(parsed, list) and len(parsed) > 0:
            return [str(q).strip() for q in parsed if str(q).strip()][:3]
        elif isinstance(parsed, dict) and "queries" in parsed and isinstance(parsed["queries"], list):
            return [str(q).strip() for q in parsed["queries"] if str(q).strip()][:3]

        return [claim, f"{claim} systematic review", f"{claim} official analysis"]
    except Exception as exc:
        logger.warning("Query decomposer fallback due to error: %s", exc)
        return [claim, f"{claim} systematic review", f"{claim} official analysis"]


# ── Dynamic Engine Dispatcher ──────────────────────────────


def select_engines(claim_type: str) -> list[str]:
    """Determine which SerpApi engines to query based on claim categorization."""
    mapping = {
        "scientific": ["google", "google_scholar"],
        "temporal_news": ["google", "google_news"],
        "commercial_saas": ["google", "google_news"],
        "general_factoid": ["google"],
    }
    return mapping.get(claim_type, ["google", "google_scholar"])


def classify_claim_type(claim: str) -> str:
    """Heuristic / keyword classification to choose optimal search surfaces."""
    c = claim.lower()
    # Scientific / medical / nutritional terms
    if any(k in c for k in [
        "creatine", "supplement", "dose", "protein", "health", "cancer", "vaccine",
        "disease", "clinical", "optical", "scientific", "physics", "chemical", "celsius",
        "pressure", "freez", "boil", "temperature", "hair loss", "dna", "cell"
    ]):
        return "scientific"

    # Temporal / recent / release terms
    if any(k in c for k in [
        "launched", "released", "announced", "2024", "2025", "2026", "yesterday",
        "last month", "recently", "sora", "openai", "acquisition", "died", "resigned"
    ]):
        return "temporal_news"

    # Software / commercial products
    if any(k in c for k in ["pricing", "cost", "features", "saas", "api", "subscription"]):
        return "commercial_saas"

    return "scientific"  # Default to rigorous multi-surface search


# ── AI Overview Prompt Builder ─────────────────────────────


def _build_overview_prompt(claim: str, sources: list[dict]) -> str:
    """
    Format sources into numbered references [1], [2] and prompt the LLM
    to generate an epistemic AI Overview with grounded inline citations.
    """
    formatted_sources_list = []
    for idx, s in enumerate(sources, 1):
        tier_label = s.get("authority_label", f"Tier {s.get('authority_tier', 3)}")
        formatted_sources_list.append(
            f"[{idx}] Title: {s.get('title', 'Unknown')}\n"
            f"    Domain/Source: {s.get('source_domain', '')} ({tier_label})\n"
            f"    Engine: {s.get('source_engine', 'google')}\n"
            f"    Published: {s.get('date', 'Unknown date')}\n"
            f"    URL: {s.get('link', '')}\n"
            f"    Excerpt: {s.get('snippet', '')}"
        )
    formatted_sources_text = "\n\n".join(formatted_sources_list)

    return f"""You are the core intelligence engine of FactTrace, an Epistemic Audit Agent modeled after Google AI Overviews.

Analyze the user's claim against the provided evidence snippets.

CLAIM: "{claim}"

EVIDENCE SOURCES:
{formatted_sources_text if formatted_sources_text else "No sources available."}

TASK:
1. Synthesize an "Executive AI Overview" explaining the factual truth, nuances, and context.
2. CRITICAL: Every major factual claim or assertion MUST cite the source index in square brackets, e.g., "Clinical trials show no direct link between creatine and DHT levels [1][3], though a single 2009 study on rugby players noted a temporary fluctuation [2]."
3. Explicitly detect if mainstream search snippets diverge from academic papers or wire reports.
4. Provide a confidence score from 0.0 to 1.0 calibrated to source reliability.

Return strictly JSON matching the schema:
{{
  "verdict": "True | Mostly True | Contested | Mostly False | False | Unverifiable",
  "confidence": 0.0 to 1.0,
  "executive_overview": "Comprehensive synthesized response with [1], [2] citations...",
  "consensus_summary": "1-sentence summary of expert/empirical consensus",
  "divergence_detected": true,
  "divergence_notes": "Explanation of where web vs academic/news split, if applicable",
  "source_stances": [
     {{"index": 1, "stance": "supports | refutes | neutral"}}
  ]
}}"""


# ── Main Verifier Class ────────────────────────────────────


class FactVerifier:
    """
    Orchestrates the end-to-end fact verification pipeline with dynamic engine dispatching,
    LLM query decomposition, tiered authority filtering, and grounded AI Overviews.

    Usage::

        verifier = FactVerifier()
        result = verifier.verify("Creatine causes hair loss in men.")
        print(result.verdict, result.confidence_score, result.executive_overview)
    """

    def __init__(self, max_snippets_per_engine: int = 5):
        self._max_snippets = max_snippets_per_engine

    def _generate_search_queries(self, claim: str, claim_type: str = "scientific") -> list[str]:
        """
        Expand a claim into targeted search queries via the LLM Search Planner.
        """
        return generate_targeted_queries(claim, claim_type)

    def verify(self, claim: str) -> VerificationResult:
        """
        Run the full verification pipeline for a claim.

        Steps:
            1. Categorize claim & select optimal SerpApi engines.
            2. Generate 3 operator-rich search queries via LLM planner.
            3. Fetch evidence from targeted engines (cached).
            4. Filter and rank evidence by domain authority tiers.
            5. Synthesize Google AI Overview with grounded [1], [2] citations.
            6. Parse structured VerificationResult.
        """
        claim_type = classify_claim_type(claim)
        target_engines = select_engines(claim_type)
        logger.info("Claim classified as '%s'. Targeting engines: %s", claim_type, target_engines)

        # 1. Query decomposition & search operator injection
        queries = self._generate_search_queries(claim, claim_type)
        logger.info("Planner generated %d queries: %s", len(queries), queries)

        # 2 & 3. Multi-engine search + snippet extraction
        all_snippets: list[dict] = []
        raw_results_list: list[dict[str, Any]] = []
        engines_used: set[str] = set()

        for query in queries:
            raw = search_all_engines(
                query,
                engines_to_query=target_engines,
                num_organic=self._max_snippets,
            )
            raw_results_list.append(raw)
            for eng_name, eng_data in raw.items():
                if isinstance(eng_data, dict) and "error" not in eng_data:
                    engines_used.add(eng_name)
            all_snippets.extend(
                extract_snippets(raw, max_per_engine=self._max_snippets)
            )

        # Extract deep SerpApi metadata (knowledge graph, related questions)
        kg_data = None
        related_queries_list: list[str] = []
        for raw in raw_results_list:
            meta = extract_rich_metadata(raw)
            if not kg_data and meta.get("knowledge_graph") and meta["knowledge_graph"].get("title"):
                kg_dict = meta["knowledge_graph"]
                kg_data = KnowledgeGraphEntity(
                    title=kg_dict.get("title", ""),
                    type=kg_dict.get("type", ""),
                    description=kg_dict.get("description", ""),
                    source=kg_dict.get("source", ""),
                )
            for rq in meta.get("related_queries", []):
                if rq not in related_queries_list:
                    related_queries_list.append(rq)

        # Deduplicate snippets by URL while maintaining priority order
        seen_urls: set[str] = set()
        unique_snippets: list[dict] = []
        for s in all_snippets:
            url = s.get("link", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_snippets.append(s)

        # Re-sort unique snippets by authority tier (Tier 1 first)
        unique_snippets.sort(key=lambda x: x.get("authority_tier", 3))

        logger.info("Collected %d unique ranked evidence snippets", len(unique_snippets))

        if not unique_snippets:
            return VerificationResult(
                claim=claim,
                verdict=Verdict.UNVERIFIABLE,
                confidence_score=0.0,
                explanation="No reputable evidence could be retrieved for this claim under domain authority filtering.",
                executive_overview="No primary or reputable sources qualified under domain authority filters to verify this claim.",
                consensus_summary="Insufficient evidence to determine consensus.",
                divergence_detected=False,
                divergence_notes="",
                evidence=[],
                search_queries_used=queries,
                engines_queried=sorted(list(engines_used)),
                knowledge_graph=kg_data,
                related_queries=related_queries_list[:4],
            )

        # 4. LLM Epistemic AI Overview with inline citations
        overview_prompt = _build_overview_prompt(claim, unique_snippets)
        llm_response = _call_llm(overview_prompt, response_mime_type="application/json")

        # 5. Parse structured output
        return self._parse_llm_response(
            claim=claim,
            queries=queries,
            raw_response=llm_response,
            unique_snippets=unique_snippets,
            engines_queried=sorted(list(engines_used)),
            knowledge_graph=kg_data,
            related_queries=related_queries_list[:4],
        )

    def _parse_llm_response(
        self,
        claim: str,
        queries: list[str],
        raw_response: str,
        unique_snippets: list[dict],
        engines_queried: list[str] | None = None,
        knowledge_graph: KnowledgeGraphEntity | None = None,
        related_queries: list[str] | None = None,
    ) -> VerificationResult:
        """
        Parse the LLM's JSON response into a VerificationResult with grounded citations.
        Falls back to UNVERIFIABLE if parsing fails.
        """
        engines_queried = engines_queried or []
        related_queries = related_queries or []

        try:
            # Strip markdown code fences if wrapped
            cleaned = raw_response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            cleaned = cleaned.strip()

            data: dict[str, Any] = json.loads(cleaned)

            # Build stances lookup by source index
            stances_by_idx: dict[int, str] = {}
            for st_item in data.get("source_stances", []):
                idx = st_item.get("index")
                if idx is not None:
                    stances_by_idx[int(idx)] = st_item.get("stance", "neutral")

            # Map unique snippets to SourceEvidence
            evidence: list[SourceEvidence] = []
            for idx, s in enumerate(unique_snippets, 1):
                stance = stances_by_idx.get(idx, s.get("stance", "neutral"))
                evidence.append(
                    SourceEvidence(
                        index=idx,
                        title=s.get("title", ""),
                        snippet=s.get("snippet", ""),
                        url=s.get("link", ""),
                        source_engine=s.get("source_engine", "unknown"),
                        stance=stance,
                        date=s.get("date", "Unknown date"),
                        source_domain=s.get("source_domain", ""),
                        authority_tier=s.get("authority_tier", 3),
                        authority_label=s.get("authority_label", "Standard Web"),
                        highlighted_words=s.get("highlighted_words", []),
                    )
                )

            # Map verdict safely
            raw_verdict = str(data.get("verdict", "Unverifiable")).strip()
            verdict_map = {
                "true": Verdict.TRUE,
                "mostly true": Verdict.MOSTLY_TRUE,
                "half true": Verdict.HALF_TRUE,
                "contested": Verdict.CONTESTED,
                "mostly false": Verdict.MOSTLY_FALSE,
                "false": Verdict.FALSE,
                "unverifiable": Verdict.UNVERIFIABLE,
            }
            verdict = verdict_map.get(raw_verdict.lower(), Verdict.UNVERIFIABLE)

            # Overview & explanation
            executive_overview = data.get("executive_overview") or data.get("explanation") or ""
            consensus_summary = data.get("consensus_summary", "")
            divergence_detected = bool(data.get("divergence_detected", False))
            divergence_notes = data.get("divergence_notes", "")
            confidence = float(data.get("confidence", data.get("confidence_score", 0.0)))

            return VerificationResult(
                claim=claim,
                verdict=verdict,
                confidence_score=max(0.0, min(1.0, confidence)),
                explanation=executive_overview,
                executive_overview=executive_overview,
                consensus_summary=consensus_summary,
                divergence_detected=divergence_detected,
                divergence_notes=divergence_notes,
                evidence=evidence,
                search_queries_used=queries,
                engines_queried=engines_queried,
                knowledge_graph=knowledge_graph,
                related_queries=related_queries,
            )

        except Exception as exc:
            logger.error("Failed to parse LLM response: %s\nRaw: %s", exc, raw_response[:500])
            fallback_evidence = [
                SourceEvidence(
                    index=idx,
                    title=s.get("title", ""),
                    snippet=s.get("snippet", ""),
                    url=s.get("link", ""),
                    source_engine=s.get("source_engine", "unknown"),
                    stance="neutral",
                    date=s.get("date", "Unknown date"),
                    source_domain=s.get("source_domain", ""),
                    authority_tier=s.get("authority_tier", 3),
                    authority_label=s.get("authority_label", "Standard Web"),
                    highlighted_words=s.get("highlighted_words", []),
                )
                for idx, s in enumerate(unique_snippets, 1)
            ]
            return VerificationResult(
                claim=claim,
                verdict=Verdict.UNVERIFIABLE,
                confidence_score=0.0,
                explanation=f"LLM analysis could not be parsed. Raw response: {raw_response[:300]}",
                executive_overview=f"LLM analysis parsing encountered an error. Raw output: {raw_response[:300]}",
                consensus_summary="Unverified due to parsing failure.",
                divergence_detected=False,
                divergence_notes="",
                evidence=fallback_evidence,
                search_queries_used=queries,
                engines_queried=engines_queried,
                knowledge_graph=knowledge_graph,
                related_queries=related_queries,
            )


# ── Interactive Follow-up Investigation Helper ─────────────


def verify_follow_up(
    claim: str,
    follow_up_question: str,
    previous_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Run an epistemic audit on a follow-up query in context of the previous audit.
    Conducts targeted search on the follow-up, ranks evidence by authority, and synthesizes
    an AI Overview with grounded citations.
    """
    claim_type = classify_claim_type(f"{claim} {follow_up_question}")
    target_engines = select_engines(claim_type)

    # Decompose follow-up query into targeted search operators
    targeted_queries = generate_targeted_queries(f"{claim}: {follow_up_question}", claim_type)

    all_snippets: list[dict] = []
    engines_used: set[str] = set()
    for query in targeted_queries:
        raw = search_all_engines(query, engines_to_query=target_engines, num_organic=3)
        for eng_name, eng_data in raw.items():
            if isinstance(eng_data, dict) and "error" not in eng_data:
                engines_used.add(eng_name)
        all_snippets.extend(extract_snippets(raw, max_per_engine=3))

    seen_urls: set[str] = set()
    unique_snippets: list[dict] = []
    for s in all_snippets:
        url = s.get("link", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_snippets.append(s)

    unique_snippets.sort(key=lambda x: x.get("authority_tier", 3))

    # Format new sources
    formatted_sources = []
    for idx, s in enumerate(unique_snippets, 1):
        formatted_sources.append(
            f"[{idx}] Title: {s.get('title')}\n"
            f"    Domain: {s.get('source_domain')} ({s.get('authority_label')})\n"
            f"    Engine: {s.get('source_engine')}\n"
            f"    Published: {s.get('date')}\n"
            f"    URL: {s.get('link')}\n"
            f"    Excerpt: {s.get('snippet')}"
        )
    formatted_sources_text = "\n\n".join(formatted_sources)

    prev_overview = previous_result.get("executive_overview") or previous_result.get("explanation", "")
    prev_verdict = previous_result.get("verdict", "")

    prompt = f"""You are FactTrace in Google AI Follow-up Mode.

ORIGINAL CLAIM: "{claim}"
PREVIOUS AUDIT VERDICT: {prev_verdict}
PREVIOUS EXECUTIVE OVERVIEW:
{prev_overview}

FOLLOW-UP QUESTION / CHALLENGE:
"{follow_up_question}"

FOLLOW-UP EVIDENCE SOURCES:
{formatted_sources_text if formatted_sources_text else "No new sources retrieved."}

TASK:
1. Address the follow-up question or challenge directly and objectively.
2. Ground all factual assertions using bracketed citation indices [1], [2] matching the follow-up sources.
3. State clearly whether this challenges or reinforces the initial audit verdict.

Return strictly JSON:
{{
  "follow_up_overview": "Direct answer to the follow-up with [1], [2] bracketed citations...",
  "status": "reinforces | challenges | clarifies",
  "key_finding": "1-sentence summary of the follow-up conclusion"
}}
"""
    try:
        resp = _call_llm(prompt, response_mime_type="application/json")
        cleaned = resp.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        parsed = json.loads(cleaned.strip())
    except Exception as exc:
        logger.error("Error in follow-up response: %s", exc)
        parsed = {
            "follow_up_overview": f"Investigated: '{follow_up_question}'. Based on previous audit context and corroboration sources.",
            "status": "clarifies",
            "key_finding": "Investigation completed.",
        }

    return {
        "follow_up_question": follow_up_question,
        "answer": parsed.get("follow_up_overview", ""),
        "status": parsed.get("status", "clarifies"),
        "key_finding": parsed.get("key_finding", ""),
        "sources": [
            {
                "index": idx,
                "title": s.get("title", ""),
                "snippet": s.get("snippet", ""),
                "link": s.get("link", ""),
                "engine": s.get("source_engine", ""),
                "date": s.get("date", ""),
                "domain": s.get("source_domain", ""),
                "authority_tier": s.get("authority_tier", 3),
                "authority_label": s.get("authority_label", "Standard Web"),
                "highlights": s.get("highlighted_words", []),
            }
            for idx, s in enumerate(unique_snippets, 1)
        ],
        "search_queries_used": targeted_queries,
    }


# ── Convenience Entrypoint ──────────────────────────────────


def verify_claim(claim: str) -> dict[str, Any]:
    """
    Convenience function returning a clean dict representation
    tailored for frontend and demo consumption with Google AI Mode fields.
    """
    verifier = FactVerifier()
    res = verifier.verify(claim)

    return {
        "claim": res.claim,
        "verdict": res.verdict.value,
        "confidence": res.confidence_score,
        "explanation": res.explanation,
        "executive_overview": res.executive_overview or res.explanation,
        "consensus_summary": res.consensus_summary,
        "divergence_detected": res.divergence_detected,
        "divergence_notes": res.divergence_notes,
        "engines_queried": res.engines_queried,
        "knowledge_graph": res.knowledge_graph.model_dump() if res.knowledge_graph else None,
        "related_queries": res.related_queries,
        "sources": [
            {
                "index": ev.index,
                "title": ev.title,
                "snippet": ev.snippet,
                "link": ev.url,
                "engine": ev.source_engine,
                "stance": ev.stance,
                "date": ev.date,
                "domain": ev.source_domain,
                "authority_tier": ev.authority_tier,
                "authority_label": ev.authority_label,
                "highlighted_words": ev.highlighted_words,
                "highlights": ev.highlighted_words,
            }
            for ev in res.evidence
        ],
        "search_queries_used": res.search_queries_used,
    }
