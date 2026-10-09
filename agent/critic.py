"""
Epistemic Critic Agent
======================
Protects the evidence stream from the "Domain Authority Trap", purging CAPTCHAs,
academic journal ads, cookie walls, and off-topic high-authority domains (e.g. NOAA
marine mammal reports on physics queries). Implements hybrid relevance-authority ranking.
"""

import logging
import re
from typing import Any
from urllib.parse import urlparse
from pydantic import BaseModel, Field

from .planner import Plan
from .retrieval import RawEvidenceBundle

logger = logging.getLogger(__name__)

# ── Boilerplate, CAPTCHA, and Marketing Patterns to Purge ───

BLOCKED_SNIPPET_PATTERNS = [
    "javascript is disabled",
    "verify that you're not a robot",
    "enable javascript",
    "access to this page has been denied",
    "checking your browser",
    "security check to access",
    "cloudflare",
    "cookie policy",
    "terms of service",
    "privacy policy",
    "impact factor",
    "publish in scientific reports",
    "publish in",
    "submit your manuscript",
    "call for papers",
    "subscription required",
    "sign in to continue",
    "all rights reserved",
    "online scientific calculator",
    "free online calculator",
    "free graphing calculator",
    "desmos",
]

# Social / forum domains to drop completely
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

# Authority clusters
TIER_1_DOMAINS = {
    "nih.gov", "ncbi.nlm.nih.gov", "cdc.gov", "fda.gov", "who.int", "nature.com",
    "sciencedirect.com", "thelancet.com", "jamanetwork.com", "cell.com", "arxiv.org",
    "reuters.com", "apnews.com", "bloomberg.com", "sec.gov", "nasa.gov", "weather.gov"
}

TIER_2_DOMAINS = {
    "bbc.com", "nytimes.com", "washingtonpost.com", "wsj.com", "theguardian.com",
    "thehindu.com", "economist.com", "ft.com", "scientificamerican.com"
}


class CuratedEvidence(BaseModel):
    """An audited, relevant evidence snippet passed to the Synthesis Agent."""

    index: int
    title: str
    snippet: str
    url: str
    source_engine: str
    date: str
    source_domain: str
    authority_tier: int
    authority_label: str
    relevance_score: float
    final_score: float
    highlighted_words: list[str] = Field(default_factory=list)


class CriticAuditReport(BaseModel):
    """Output of the Epistemic Critic Agent."""

    curated_evidence: list[CuratedEvidence] = Field(default_factory=list)
    purged_count: int = 0
    purged_reasons: dict[str, int] = Field(default_factory=dict)
    divergence_detected: bool = False
    divergence_notes: str = ""


def is_valid_content(snippet: str, title: str, url: str) -> tuple[bool, str]:
    """Check if content is genuine evidence or scraper-block/ad/banned domain."""
    url_lower = url.lower()
    if any(banned in url_lower for banned in BANNED_DOMAINS):
        return False, "banned_social_domain"

    combined = f"{title} {snippet}".lower()
    for pat in BLOCKED_SNIPPET_PATTERNS:
        if pat in combined:
            return False, f"boilerplate_or_captcha: {pat}"

    if len(snippet.strip()) < 20:
        return False, "insufficient_length"

    return True, "valid"


def calculate_domain_tier(url: str, is_scholar: bool = False) -> tuple[int, str, float]:
    """
    Returns (tier_int, label, weight_multiplier).
    Tier 1 = 1.0 weight, Tier 2 = 0.65 weight, Tier 3 = 0.35 weight.
    """
    if not url:
        return (1, "Tier-1 Authoritative (Peer-Reviewed Scholar)", 1.0) if is_scholar else (3, "Standard Web", 0.35)

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
        return (1, label, 1.0)

    if any(domain.endswith(t2) or domain == t2 for t2 in TIER_2_DOMAINS):
        return (2, "Tier-2 Reputable (Major Institutional Press)", 0.65)

    return (3, "Standard Web", 0.35)


def calculate_relevance_score(snippet: str, title: str, core_concepts: list[str]) -> float:
    """
    Computes topical relevance score (0.0 to 1.0) based on core concept matches.
    Prevents off-topic .gov pages (e.g. NOAA marine mammals) from ranking for physics queries.
    """
    if not core_concepts:
        return 0.5  # Neutral default if no concepts defined

    text = f"{title} {snippet}".lower()
    matched_concepts = 0

    for concept in core_concepts:
        c_clean = concept.strip().lower()
        if not c_clean:
            continue
        # Check full concept phrase
        if c_clean in text:
            matched_concepts += 1
            continue
        # Check individual words in multi-word concepts
        sub_words = [w for w in c_clean.split() if len(w) > 3]
        if sub_words and any(w in text for w in sub_words):
            matched_concepts += 0.5

    return min(1.0, matched_concepts / max(len(core_concepts), 1))


class EpistemicCriticAgent:
    """
    Filters out noise, evaluates topical relevance, applies domain authority weighting,
    and resolves evidence conflicts.
    """

    def __init__(self, min_relevance_threshold: float = 0.20, top_k: int = 7):
        self.min_relevance = min_relevance_threshold
        self.top_k = top_k

    def audit(self, raw_bundle: RawEvidenceBundle, plan: Plan) -> CriticAuditReport:
        """
        Evaluate raw retrieved evidence snippets and return curated, topically grounded sources.
        """
        valid_candidates: list[dict] = []
        purged_reasons: dict[str, int] = {}
        purged_count = 0

        seen_urls = set()

        for raw_item in raw_bundle.raw_snippets:
            url = raw_item.get("link", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            snippet = raw_item.get("snippet", "")
            title = raw_item.get("title", "")
            is_scholar = raw_item.get("source_engine") == "google_scholar"

            # 1. Boilerplate & CAPTCHA Purge
            is_valid, reason = is_valid_content(snippet, title, url)
            if not is_valid:
                purged_count += 1
                purged_reasons[reason] = purged_reasons.get(reason, 0) + 1
                continue

            # 2. Semantic Relevance Gating
            rel_score = calculate_relevance_score(snippet, title, plan.core_concepts)
            if rel_score < self.min_relevance:
                purged_count += 1
                reason = "off_topic_insufficient_relevance"
                purged_reasons[reason] = purged_reasons.get(reason, 0) + 1
                continue

            # 3. Domain Tier Calculation
            tier, label, auth_weight = calculate_domain_tier(url, is_scholar=is_scholar)

            # 4. Hybrid Scoring Formula: 70% Topical Relevance, 30% Domain Authority
            final_score = (rel_score * 0.70) + (auth_weight * 0.30)

            valid_candidates.append({
                "title": title,
                "snippet": snippet,
                "url": url,
                "source_engine": raw_item.get("source_engine", "google"),
                "date": raw_item.get("date", "Unknown date"),
                "source_domain": raw_item.get("source_domain", ""),
                "authority_tier": tier,
                "authority_label": label,
                "relevance_score": round(rel_score, 3),
                "final_score": round(final_score, 3),
                "highlighted_words": raw_item.get("highlighted_words", []),
            })

        # Sort primarily by hybrid final_score descending
        valid_candidates.sort(key=lambda x: x["final_score"], reverse=True)

        curated = [
            CuratedEvidence(
                index=idx,
                title=c["title"],
                snippet=c["snippet"],
                url=c["url"],
                source_engine=c["source_engine"],
                date=c["date"],
                source_domain=c["source_domain"],
                authority_tier=c["authority_tier"],
                authority_label=c["authority_label"],
                relevance_score=c["relevance_score"],
                final_score=c["final_score"],
                highlighted_words=c["highlighted_words"],
            )
            for idx, c in enumerate(valid_candidates[:self.top_k], 1)
        ]

        logger.info(
            "CriticAgent audited %d raw snippets: %d approved, %d purged",
            len(raw_bundle.raw_snippets),
            len(curated),
            purged_count,
        )

        return CriticAuditReport(
            curated_evidence=curated,
            purged_count=purged_count,
            purged_reasons=purged_reasons,
        )
