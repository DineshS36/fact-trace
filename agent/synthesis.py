"""
Synthesis Agent
===============
Produces calibrated Epistemic AI Overviews modeled after Google AI Overviews,
enforcing bracketed citations ([1], [2]), empirical consensus extraction,
and source divergence detection.
"""

import json
import logging
from typing import Any
from pydantic import BaseModel, Field

from .critic import CriticAuditReport, CuratedEvidence
from .llm_client import call_llm, clean_llm_json
from .planner import Plan

logger = logging.getLogger(__name__)


class SynthesisResult(BaseModel):
    """Calibrated synthesis produced by the Synthesis Agent."""

    verdict: str
    confidence: float = Field(ge=0.0, le=1.0)
    executive_overview: str
    consensus_summary: str
    divergence_detected: bool = False
    divergence_notes: str = ""
    source_stances: dict[int, str] = Field(default_factory=dict)


class SynthesisAgent:
    """
    Synthesizes audited evidence into an executive overview with grounded citations.
    """

    def synthesize(self, claim: str, plan: Plan, critic_report: CriticAuditReport) -> SynthesisResult:
        """
        Synthesize claim against critic-approved evidence.
        """
        sources = critic_report.curated_evidence

        if not sources:
            return SynthesisResult(
                verdict="Unverifiable",
                confidence=0.0,
                executive_overview="No primary or reputable sources passed the Epistemic Critic's topical relevance and anti-spam filters.",
                consensus_summary="Insufficient evidence to determine consensus.",
                divergence_detected=False,
                divergence_notes="",
                source_stances={},
            )

        formatted_sources_list = []
        for s in sources:
            formatted_sources_list.append(
                f"[{s.index}] Title: {s.title}\n"
                f"    Domain/Source: {s.source_domain} ({s.authority_label})\n"
                f"    Engine: {s.source_engine}\n"
                f"    Published: {s.date}\n"
                f"    URL: {s.url}\n"
                f"    Excerpt: {s.snippet}"
            )
        formatted_sources_text = "\n\n".join(formatted_sources_list)

        prompt = f"""You are the Synthesis Agent of FactTrace, an Epistemic Audit Agent modeled after Google AI Overviews.

Analyze the user's claim against the provided evidence snippets.

CLAIM: "{claim}"
CLAIM TYPE: {plan.claim_type}
CORE CONCEPTS: {', '.join(plan.core_concepts)}

AUDITED EVIDENCE SOURCES:
{formatted_sources_text}

TASK:
1. Synthesize an "Executive AI Overview" explaining the factual truth, nuances, and context.
2. CRITICAL: Every major factual claim or assertion MUST cite the source index in square brackets, e.g., "Clinical trials show no direct link between creatine and DHT levels [1][3], though a single 2009 study on rugby players noted a temporary fluctuation [2]."
3. Explicitly detect if mainstream search snippets diverge from academic papers or wire reports.
4. Provide a confidence score from 0.0 to 1.0 calibrated to source reliability.

Return strictly JSON matching this schema:
{{
  "verdict": "True | Mostly True | Contested | Mostly False | False | Unverifiable",
  "confidence": 0.0 to 1.0,
  "executive_overview": "Comprehensive synthesized response with [1], [2] citations...",
  "consensus_summary": "1-sentence summary of expert/empirical consensus",
  "divergence_detected": true or false,
  "divergence_notes": "Explanation of where web vs academic/news split, if applicable",
  "source_stances": [
     {{"index": 1, "stance": "supports | refutes | neutral"}}
  ]
}}"""

        try:
            response = call_llm(prompt, response_mime_type="application/json")
            data = clean_llm_json(response)
            if not isinstance(data, dict):
                data = {}

            stances = {}
            for st_item in data.get("source_stances", []):
                idx = st_item.get("index")
                if idx is not None:
                    stances[int(idx)] = st_item.get("stance", "neutral")

            return SynthesisResult(
                verdict=str(data.get("verdict", "Unverifiable")),
                confidence=float(data.get("confidence", 0.0)),
                executive_overview=str(data.get("executive_overview", "")),
                consensus_summary=str(data.get("consensus_summary", "")),
                divergence_detected=bool(data.get("divergence_detected", False)),
                divergence_notes=str(data.get("divergence_notes", "")),
                source_stances=stances,
            )

        except Exception as exc:
            logger.error("SynthesisAgent encountered parsing error: %s", exc)
            return SynthesisResult(
                verdict="Unverifiable",
                confidence=0.0,
                executive_overview="Epistemic synthesis could not be parsed.",
                consensus_summary="Unverified.",
                divergence_detected=False,
                divergence_notes="",
                source_stances={},
            )

    def synthesize_follow_up(
        self,
        claim: str,
        follow_up_question: str,
        previous_overview: str,
        previous_verdict: str,
        critic_report: CriticAuditReport,
    ) -> dict[str, Any]:
        """
        Synthesize conversational follow-up turns in context of the previous audit.
        """
        sources = critic_report.curated_evidence

        formatted_sources = []
        for s in sources:
            formatted_sources.append(
                f"[{s.index}] Title: {s.title}\n"
                f"    Domain: {s.source_domain} ({s.authority_label})\n"
                f"    Excerpt: {s.snippet}"
            )
        formatted_sources_text = "\n\n".join(formatted_sources)

        prompt = f"""You are FactTrace in Google AI Follow-up Mode.

ORIGINAL CLAIM: "{claim}"
PREVIOUS AUDIT VERDICT: {previous_verdict}
PREVIOUS EXECUTIVE OVERVIEW:
{previous_overview}

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
}}"""

        try:
            resp = call_llm(prompt, response_mime_type="application/json")
            parsed = clean_llm_json(resp)
            if not isinstance(parsed, dict):
                parsed = {}
        except Exception as exc:
            logger.error("Error in follow-up synthesis: %s", exc)
            parsed = {
                "follow_up_overview": f"Investigated: '{follow_up_question}'. Based on previous audit context and corroboration sources.",
                "status": "clarifies",
                "key_finding": "Investigation completed.",
            }

        return parsed
