"""
Core Fact-Verification Agent
=============================
Orchestrates multi-engine evidence retrieval and LLM-powered
analysis to score factual claims.

Architecture:
    Claim → search_all_engines → extract_snippets → LLM analysis → VerificationResult

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

from .search_tools import extract_rich_metadata, extract_snippets, search_all_engines

load_dotenv()
logger = logging.getLogger(__name__)


# ── Data Models ─────────────────────────────────────────────


class Verdict(str, Enum):
    """Discrete verdicts the agent can assign to a claim."""

    TRUE = "True"
    MOSTLY_TRUE = "Mostly True"
    HALF_TRUE = "Half True"
    MOSTLY_FALSE = "Mostly False"
    FALSE = "False"
    UNVERIFIABLE = "Unverifiable"


class SourceEvidence(BaseModel):
    """A single piece of evidence used in the verdict."""

    title: str
    snippet: str
    url: str
    source_engine: str
    stance: str = Field(
        description="How this source relates to the claim: supports | refutes | neutral"
    )
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


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Route the prompt to the configured LLM provider and return
    the raw response text.

    Raises:
        RuntimeError: If the provider is unsupported or the key is missing.
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
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""

    if provider == "groq":
        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set in .env")
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""

    if provider == "gemini":
        from google import genai
        from google.genai import types

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in .env")
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model,
            contents=f"{system_prompt}\n\n{user_prompt}",
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )
        return resp.text or ""

    raise RuntimeError(
        f"Unsupported LLM_PROVIDER='{provider}'. Use 'openai', 'groq', or 'gemini'."
    )


# ── Prompt Templates ────────────────────────────────────────

_SYSTEM_PROMPT = """You are FactTrace, a rigorous fact-verification analyst.

Your task:
1. Analyse the CLAIM against the provided EVIDENCE snippets.
2. Determine a verdict and confidence score.
3. Cite which evidence supports or refutes the claim.

IMPORTANT RULES:
- Be precise and cite sources.
- If evidence is insufficient, say "Unverifiable" rather than guessing.
- Confidence score (0.0–1.0) reflects how strongly evidence supports your verdict.
- Respond ONLY with valid JSON matching this schema (no markdown fences):

{
  "verdict": "True | Mostly True | Half True | Mostly False | False | Unverifiable",
  "confidence_score": 0.0 to 1.0,
  "explanation": "2-4 sentence analysis",
  "evidence": [
    {
      "title": "source title",
      "snippet": "relevant excerpt",
      "url": "source url",
      "source_engine": "google | google_news | google_scholar",
      "stance": "supports | refutes | neutral"
    }
  ]
}"""


def _build_user_prompt(claim: str, snippets: list[dict]) -> str:
    evidence_text = json.dumps(snippets, indent=2)
    return f"CLAIM: {claim}\n\nEVIDENCE:\n{evidence_text}"


# ── Main Verifier Class ────────────────────────────────────


class FactVerifier:
    """
    Orchestrates the end-to-end fact verification pipeline.

    Usage::

        verifier = FactVerifier()
        result = verifier.verify("The Great Wall of China is visible from space.")
        print(result.verdict, result.confidence_score)
    """

    def __init__(self, max_snippets_per_engine: int = 5):
        self._max_snippets = max_snippets_per_engine

    def _generate_search_queries(self, claim: str) -> list[str]:
        """
        Expand a single claim into multiple search queries for
        better evidence coverage.

        Currently uses a simple heuristic; can be upgraded to
        LLM-generated query expansion.
        """
        queries = [claim]

        # Add a "fact check" variant to surface dedicated fact-check sites
        if "fact check" not in claim.lower():
            queries.append(f"{claim} fact check")

        return queries

    def verify(self, claim: str) -> VerificationResult:
        """
        Run the full verification pipeline for a claim.

        Steps:
            1. Generate search queries from the claim.
            2. Fetch evidence from all engines (cached).
            3. Extract & flatten snippets.
            4. Send claim + evidence to LLM for analysis.
            5. Parse LLM output into VerificationResult.

        Args:
            claim: The factual claim to verify.

        Returns:
            A structured VerificationResult.
        """
        # 1. Query expansion
        queries = self._generate_search_queries(claim)
        logger.info("Verifying claim with %d queries: %s", len(queries), queries)

        # 2 & 3. Multi-engine search + snippet extraction
        all_snippets: list[dict] = []
        raw_results_list: list[dict[str, Any]] = []
        engines_used: set[str] = set()

        for query in queries:
            raw = search_all_engines(query, num_organic=self._max_snippets)
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

        # Map URLs to highlight words
        url_to_highlights: dict[str, list[str]] = {}
        for s in all_snippets:
            url = s.get("link", "")
            if url and s.get("highlighted_words"):
                url_to_highlights[url] = s.get("highlighted_words", [])

        # Deduplicate by URL
        seen_urls: set[str] = set()
        unique_snippets: list[dict] = []
        for s in all_snippets:
            url = s.get("link", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_snippets.append(s)

        logger.info("Collected %d unique evidence snippets", len(unique_snippets))

        if not unique_snippets:
            return VerificationResult(
                claim=claim,
                verdict=Verdict.UNVERIFIABLE,
                confidence_score=0.0,
                explanation="No reputable evidence could be retrieved for this claim.",
                evidence=[],
                search_queries_used=queries,
                engines_queried=sorted(list(engines_used)),
                knowledge_graph=kg_data,
                related_queries=related_queries_list[:4],
            )

        # 4. LLM analysis
        user_prompt = _build_user_prompt(claim, unique_snippets)
        llm_response = _call_llm(_SYSTEM_PROMPT, user_prompt)

        # 5. Parse structured output
        return self._parse_llm_response(
            claim=claim,
            queries=queries,
            raw_response=llm_response,
            engines_queried=sorted(list(engines_used)),
            knowledge_graph=kg_data,
            related_queries=related_queries_list[:4],
            url_to_highlights=url_to_highlights,
        )

    def _parse_llm_response(
        self,
        claim: str,
        queries: list[str],
        raw_response: str,
        engines_queried: list[str] | None = None,
        knowledge_graph: KnowledgeGraphEntity | None = None,
        related_queries: list[str] | None = None,
        url_to_highlights: dict[str, list[str]] | None = None,
    ) -> VerificationResult:
        """
        Parse the LLM's JSON response into a VerificationResult.
        Falls back to UNVERIFIABLE if parsing fails.
        """
        engines_queried = engines_queried or []
        related_queries = related_queries or []
        url_to_highlights = url_to_highlights or {}

        try:
            # Strip markdown code fences if the LLM wraps them
            cleaned = raw_response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            cleaned = cleaned.strip()

            data: dict[str, Any] = json.loads(cleaned)

            evidence = [
                SourceEvidence(
                    title=e.get("title", ""),
                    snippet=e.get("snippet", ""),
                    url=e.get("url", e.get("link", "")),
                    source_engine=e.get("source_engine", "unknown"),
                    stance=e.get("stance", "neutral"),
                    highlighted_words=url_to_highlights.get(e.get("url", e.get("link", "")), []),
                )
                for e in data.get("evidence", [])
            ]

            return VerificationResult(
                claim=claim,
                verdict=Verdict(data["verdict"]),
                confidence_score=float(data["confidence_score"]),
                explanation=data["explanation"],
                evidence=evidence,
                search_queries_used=queries,
                engines_queried=engines_queried,
                knowledge_graph=knowledge_graph,
                related_queries=related_queries,
            )

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.error("Failed to parse LLM response: %s\nRaw: %s", exc, raw_response[:500])
            return VerificationResult(
                claim=claim,
                verdict=Verdict.UNVERIFIABLE,
                confidence_score=0.0,
                explanation=f"LLM analysis could not be parsed. Raw response: {raw_response[:300]}",
                evidence=[],
                search_queries_used=queries,
                engines_queried=engines_queried,
                knowledge_graph=knowledge_graph,
                related_queries=related_queries,
            )


def verify_claim(claim: str) -> dict[str, Any]:
    """
    Convenience function returning a clean dict representation
    tailored for frontend and demo consumption.
    """
    verifier = FactVerifier()
    res = verifier.verify(claim)

    return {
        "claim": res.claim,
        "verdict": res.verdict.value,
        "confidence": res.confidence_score,
        "explanation": res.explanation,
        "engines_queried": res.engines_queried,
        "knowledge_graph": res.knowledge_graph.model_dump() if res.knowledge_graph else None,
        "related_queries": res.related_queries,
        "sources": [
            {
                "title": ev.title,
                "snippet": ev.snippet,
                "link": ev.url,
                "engine": ev.source_engine,
                "stance": ev.stance,
                "highlighted_words": ev.highlighted_words,
            }
            for ev in res.evidence
        ],
        "search_queries_used": res.search_queries_used,
    }
