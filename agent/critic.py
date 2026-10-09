"""
Epistemic Critic Agent
======================
Protects the evidence stream from visible scraping flaws:
1. Dead video embeds & trash placeholders ("Video unavailable", "404 Not Found", CAPTCHAs).
2. Domain Monopolization & Duplicate Flooding via Domain Diversity Capping (max 2 per domain).
3. Off-topic domain authority traps (e.g. NOAA marine quotas for physics).
4. Cleans raw breadcrumbs into canonical host domains.
"""

import logging
import re
from typing import Any
from urllib.parse import urlparse
from pydantic import BaseModel, Field

from .planner import Plan
from .retrieval import RawEvidenceBundle

logger = logging.getLogger(__name__)

# ── Boilerplate, CAPTCHA, and Dead-Video Blacklist ──────────

TRASH_TITLES_AND_SNIPPETS = {
    "video unavailable",
    "private video",
    "this video is unavailable",
    "javascript is disabled",
    "please enable cookies",
    "verify you are human",
    "verify that you're not a robot",
    "page not found",
    "404 not found",
    "access denied",
    "access to this page has been denied",
    "security check to access",
    "checking your browser",
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
}

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


def extract_clean_domain(link: str) -> str:
    """Extracts a clean, canonical host domain from any URL."""
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
    """Check if content is genuine evidence or trash/dead video/CAPTCHA."""
    url_lower = url.lower()
    if any(banned in url_lower for banned in BANNED_DOMAINS):
        return False, "banned_social_domain"

    combined = f"{title} {snippet}".lower()
    for trash in TRASH_TITLES_AND_SNIPPETS:
        if trash in combined:
            return False, f"trash_or_dead_embed: {trash}"

    if len(snippet.strip()) < 15:
        return False, "insufficient_length"

    return True, "valid"


def calculate_domain_tier(url: str, is_scholar: bool = False) -> tuple[int, str, float]:
    """
    Returns (tier_int, label, weight_multiplier).
    Tier 1 = 1.0 weight, Tier 2 = 0.65 weight, Tier 3 = 0.35 weight.
    """
    if not url:
        return (1, "Tier-1 Authoritative (Peer-Reviewed Scholar)", 1.0) if is_scholar else (3, "Standard Web", 0.35)

    domain = extract_clean_domain(url)

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


def enforce_domain_diversity(sources: list[dict], max_per_domain: int = 2) -> list[dict]:
    """Limits results to at most max_per_domain per host domain to prevent flooding."""
    domain_counts: dict[str, int] = {}
    diverse_sources: list[dict] = []
    for s in sources:
        dom = s.get("source_domain", "") or extract_clean_domain(s.get("url", ""))
        current_count = domain_counts.get(dom, 0)
        if current_count < max_per_domain:
            diverse_sources.append(s)
            domain_counts[dom] = current_count + 1
    return diverse_sources


class EpistemicCriticAgent:
    """
    Filters out noise, evaluates topical relevance, applies domain authority weighting,
    purges dead embeds, and enforces domain diversity.
    """

    def __init__(self, min_relevance_threshold: float = 0.20, top_k: int = 7, max_per_domain: int = 2):
        self.min_relevance = min_relevance_threshold
        self.top_k = top_k
        self.max_per_domain = max_per_domain

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

            # 1. Dead Video & Trash Placeholder Purge
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

            # 3. Domain Tier & Canonical Host Calculation
            tier, label, auth_weight = calculate_domain_tier(url, is_scholar=is_scholar)
            clean_dom = extract_clean_domain(url)

            # 4. Hybrid Scoring Formula: 70% Topical Relevance, 30% Domain Authority
            final_score = (rel_score * 0.70) + (auth_weight * 0.30)

            valid_candidates.append({
                "title": title,
                "snippet": snippet,
                "url": url,
                "source_engine": raw_item.get("source_engine", "google"),
                "date": raw_item.get("date", "Unknown date"),
                "source_domain": clean_dom,
                "authority_tier": tier,
                "authority_label": label,
                "relevance_score": round(rel_score, 3),
                "final_score": round(final_score, 3),
                "highlighted_words": raw_item.get("highlighted_words", []),
            })

        # Sort primarily by hybrid final_score descending
        valid_candidates.sort(key=lambda x: x["final_score"], reverse=True)

        # 5. Enforce Domain Diversity Cap (Max 2 results per domain to prevent YouTube flooding)
        diverse_candidates = enforce_domain_diversity(valid_candidates, max_per_domain=self.max_per_domain)
        if len(valid_candidates) > len(diverse_candidates):
            domain_purged = len(valid_candidates) - len(diverse_candidates)
            purged_count += domain_purged
            purged_reasons["domain_diversity_cap_exceeded"] = domain_purged

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
            for idx, c in enumerate(diverse_candidates[:self.top_k], 1)
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
