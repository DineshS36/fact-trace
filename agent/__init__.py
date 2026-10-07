"""
FactTrace Agent Package
=======================
Multi-engine fact verification agent with SerpApi search,
SQLite caching, and LLM-powered evidence analysis.
"""

from .verifier import FactVerifier, VerificationResult, verify_claim

__all__ = ["FactVerifier", "VerificationResult", "verify_claim"]
