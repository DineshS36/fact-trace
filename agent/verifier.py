"""
FactTrace Epistemic Orchestrator
=================================
Orchestrates the 4 specialized agents:
1. Planner Agent        → Deconstructs claim, extracts core concepts, generates 3 isolated operator queries.
2. Retrieval Agent      → Executes discrete queries across surfaces via Zero-Burn SQLite Cache.
3. Epistemic Critic     → Purges dead videos, CAPTCHAs, journal ads; enforces domain diversity & hybrid ranking.
4. Synthesis Agent      → Synthesizes calibrated AI Overview with bracketed citations ([1], [2]).
Equipped with AgentTelemetry for stage latency tracking and raw SerpApi payload inspection.
"""

import logging
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field

from .critic import CriticAuditReport, EpistemicCriticAgent
from .llm_client import call_llm
from .planner import Plan, PlannerAgent
from .retrieval import RetrievalAgent
from .synthesis import SynthesisAgent
from .telemetry import AgentTelemetry

logger = logging.getLogger(__name__)


# ── Data Models ─────────────────────────────────────────────


class Verdict(str, Enum):
    """Discrete verdicts assigned to a claim."""

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
    relevance_score: float = 0.0
    final_score: float = 0.0
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
    core_concepts: list[str] = Field(default_factory=list)
    purged_sources_count: int = 0
    purged_reasons: dict[str, int] = Field(default_factory=dict)
    credit_mode: str = "balanced"
    telemetry: dict[str, Any] = Field(default_factory=dict)
    raw_serpapi_payload: dict[str, Any] = Field(default_factory=dict)


# Backward-compatible LLM caller
def _call_llm(system_prompt: str, user_prompt: str = "", response_mime_type: str = "application/json") -> str:
    return call_llm(system_prompt, user_prompt, response_mime_type)


def generate_targeted_queries(claim: str, claim_type: str = "general") -> list[str]:
    """Backward-compatible helper that delegates to PlannerAgent."""
    planner = PlannerAgent()
    plan = planner.plan(claim)
    return plan.queries


# ── Main Verifier Class (Multi-Agent Orchestrator) ──────────


class FactVerifier:
    """
    Unified Orchestrator coordinating the 4-agent epistemic pipeline:
    Planner ➔ Retrieval ➔ Critic ➔ Synthesis
    """

    def __init__(self, max_snippets_per_engine: int = 5):
        self.planner = PlannerAgent()
        self.retriever = RetrievalAgent(max_snippets_per_call=max_snippets_per_engine)
        self.critic = EpistemicCriticAgent(min_relevance_threshold=0.20, top_k=7, max_per_domain=2)
        self.synthesizer = SynthesisAgent()

    def verify(self, claim: str, credit_mode: str = "balanced") -> VerificationResult:
        """
        Execute the full 4-stage multi-agent verification pipeline with telemetry tracking:
        1. Planner Agent    → Deconstructs claim into 3 distinct operator queries and concepts.
        2. Retrieval Agent  → Gathers raw multi-surface evidence via Zero-Burn Cache.
        3. Critic Agent     → Purges dead videos, CAPTCHAs, enforces domain diversity & hybrid ranking.
        4. Synthesis Agent  → Generates Google AI Overview with bracketed citations [1], [2].
        """
        telemetry = AgentTelemetry()
        logger.info("Initiating Epistemic Audit for claim: '%s' (mode=%s)", claim, credit_mode)

        # Stage 1: Planning
        telemetry.start("planner")
        plan: Plan = self.planner.plan(claim)
        telemetry.stop("planner")
        logger.info("Planner generated 3 distinct queries: %s", plan.queries)

        # Stage 2: Retrieval
        telemetry.start("retrieval")
        raw_bundle = self.retriever.retrieve(plan, credit_mode=credit_mode)
        telemetry.stop("retrieval")
        logger.info(
            "Retriever collected %d raw snippets across engines: %s",
            len(raw_bundle.raw_snippets),
            raw_bundle.engines_used,
        )

        # Stage 3: Epistemic Criticism (Dead-video/CAPTCHA Purge + Domain Diversity + Hybrid Ranking)
        telemetry.start("critic")
        critic_report: CriticAuditReport = self.critic.audit(raw_bundle, plan)
        telemetry.stop("critic")
        logger.info(
            "Critic approved %d sources (purged %d noisy/CAPTCHA/off-topic/duplicate items)",
            len(critic_report.curated_evidence),
            critic_report.purged_count,
        )

        # Stage 4: Synthesis
        telemetry.start("synthesis")
        synthesis = self.synthesizer.synthesize(claim, plan, critic_report)
        telemetry.stop("synthesis")

        # Knowledge Graph conversion
        kg_data = None
        if raw_bundle.knowledge_graph and raw_bundle.knowledge_graph.get("title"):
            kg_data = KnowledgeGraphEntity(
                title=raw_bundle.knowledge_graph.get("title", ""),
                type=raw_bundle.knowledge_graph.get("type", ""),
                description=raw_bundle.knowledge_graph.get("description", ""),
                source=raw_bundle.knowledge_graph.get("source", ""),
            )

        # Map curated evidence into SourceEvidence with stance
        evidence: list[SourceEvidence] = []
        for s in critic_report.curated_evidence:
            stance = synthesis.source_stances.get(s.index, "neutral")
            evidence.append(
                SourceEvidence(
                    index=s.index,
                    title=s.title,
                    snippet=s.snippet,
                    url=s.url,
                    source_engine=s.source_engine,
                    stance=stance,
                    date=s.date,
                    source_domain=s.source_domain,
                    authority_tier=s.authority_tier,
                    authority_label=s.authority_label,
                    relevance_score=s.relevance_score,
                    final_score=s.final_score,
                    highlighted_words=s.highlighted_words,
                )
            )

        # Map verdict string to enum safely
        verdict_map = {
            "true": Verdict.TRUE,
            "mostly true": Verdict.MOSTLY_TRUE,
            "half true": Verdict.HALF_TRUE,
            "contested": Verdict.CONTESTED,
            "mostly false": Verdict.MOSTLY_FALSE,
            "false": Verdict.FALSE,
            "unverifiable": Verdict.UNVERIFIABLE,
        }
        verdict = verdict_map.get(synthesis.verdict.strip().lower(), Verdict.UNVERIFIABLE)

        return VerificationResult(
            claim=claim,
            verdict=verdict,
            confidence_score=max(0.0, min(1.0, synthesis.confidence)),
            explanation=synthesis.executive_overview,
            executive_overview=synthesis.executive_overview,
            consensus_summary=synthesis.consensus_summary,
            divergence_detected=synthesis.divergence_detected,
            divergence_notes=synthesis.divergence_notes,
            evidence=evidence,
            search_queries_used=plan.queries,
            engines_queried=raw_bundle.engines_used,
            knowledge_graph=kg_data,
            related_queries=raw_bundle.related_queries,
            core_concepts=plan.core_concepts,
            purged_sources_count=critic_report.purged_count,
            purged_reasons=critic_report.purged_reasons,
            credit_mode=credit_mode,
            telemetry=telemetry.to_dict(),
            raw_serpapi_payload=raw_bundle.raw_serpapi_payload,
        )


# ── Interactive Follow-up Investigation ─────────────────────


def verify_follow_up(
    claim: str,
    follow_up_question: str,
    previous_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Run an epistemic audit on a follow-up query in context of the previous audit.
    Conducts targeted planning, retrieval, and synthesis.
    """
    verifier = FactVerifier()
    combined_claim = f"{claim}: {follow_up_question}"

    # Plan follow-up
    plan = verifier.planner.plan(combined_claim)
    # Retrieve
    raw_bundle = verifier.retriever.retrieve(plan, credit_mode="balanced")
    # Critic audit
    critic_report = verifier.critic.audit(raw_bundle, plan)

    prev_overview = previous_result.get("executive_overview") or previous_result.get("explanation", "")
    prev_verdict = previous_result.get("verdict", "")

    # Synthesis
    parsed = verifier.synthesizer.synthesize_follow_up(
        claim=claim,
        follow_up_question=follow_up_question,
        previous_overview=prev_overview,
        previous_verdict=prev_verdict,
        critic_report=critic_report,
    )

    return {
        "follow_up_question": follow_up_question,
        "answer": parsed.get("follow_up_overview", ""),
        "status": parsed.get("status", "clarifies"),
        "key_finding": parsed.get("key_finding", ""),
        "sources": [
            {
                "index": s.index,
                "title": s.title,
                "snippet": s.snippet,
                "link": s.url,
                "engine": s.source_engine,
                "date": s.date,
                "domain": s.source_domain,
                "authority_tier": s.authority_tier,
                "authority_label": s.authority_label,
                "relevance_score": s.relevance_score,
                "highlights": s.highlighted_words,
            }
            for s in critic_report.curated_evidence
        ],
        "search_queries_used": plan.queries,
    }


# ── Convenience Entrypoint ──────────────────────────────────


def verify_claim(claim: str, credit_mode: str = "balanced") -> dict[str, Any]:
    """
    Convenience function returning a clean dict representation
    tailored for frontend and demo consumption.
    """
    verifier = FactVerifier()
    res = verifier.verify(claim, credit_mode=credit_mode)

    return {
        "claim": res.claim,
        "verdict": res.verdict.value,
        "confidence": res.confidence_score,
        "explanation": res.explanation,
        "executive_overview": res.executive_overview,
        "consensus_summary": res.consensus_summary,
        "divergence_detected": res.divergence_detected,
        "divergence_notes": res.divergence_notes,
        "engines_queried": res.engines_queried,
        "knowledge_graph": res.knowledge_graph.model_dump() if res.knowledge_graph else None,
        "related_queries": res.related_queries,
        "core_concepts": res.core_concepts,
        "purged_sources_count": res.purged_sources_count,
        "purged_reasons": res.purged_reasons,
        "credit_mode": res.credit_mode,
        "telemetry": res.telemetry,
        "raw_serpapi_payload": res.raw_serpapi_payload,
        "search_queries_used": res.search_queries_used,
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
                "relevance_score": ev.relevance_score,
                "final_score": ev.final_score,
                "highlighted_words": ev.highlighted_words,
                "highlights": ev.highlighted_words,
            }
            for ev in res.evidence
        ],
    }
