"""
FactTrace Agent Package
=======================
Multi-engine fact verification agent with SerpApi search,
SQLite caching, and LLM-powered evidence analysis.
"""

from .verifier import (
    FactVerifier,
    VerificationResult,
    generate_targeted_queries,
    verify_claim,
    verify_follow_up,
)

__all__ = [
    "FactVerifier",
    "VerificationResult",
    "generate_targeted_queries",
    "verify_claim",
    "verify_follow_up",
]
