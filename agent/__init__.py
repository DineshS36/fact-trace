"""
FactTrace Multi-Agent Package
=============================
Multi-engine fact verification agent with SerpApi search,
Zero-Burn SQLite caching, and a 4-stage Epistemic Multi-Agent pipeline:
1. PlannerAgent
2. RetrievalAgent
3. EpistemicCriticAgent
4. SynthesisAgent
"""

from .critic import CriticAuditReport, CuratedEvidence, EpistemicCriticAgent
from .llm_client import call_llm
from .planner import Plan, PlannerAgent
from .retrieval import RawEvidenceBundle, RetrievalAgent
from .synthesis import SynthesisAgent, SynthesisResult
from .verifier import (
    FactVerifier,
    SourceEvidence,
    Verdict,
    VerificationResult,
    generate_targeted_queries,
    verify_claim,
    verify_follow_up,
)

__all__ = [
    "PlannerAgent",
    "Plan",
    "RetrievalAgent",
    "RawEvidenceBundle",
    "EpistemicCriticAgent",
    "CriticAuditReport",
    "CuratedEvidence",
    "SynthesisAgent",
    "SynthesisResult",
    "FactVerifier",
    "VerificationResult",
    "SourceEvidence",
    "Verdict",
    "call_llm",
    "generate_targeted_queries",
    "verify_claim",
    "verify_follow_up",
]
