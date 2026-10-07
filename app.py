"""
FactTrace — Streamlit UI
=========================
Interactive fact-verification dashboard powered by multi-engine
SerpApi search and LLM analysis.

Run:
    streamlit run app.py
"""

import logging
import streamlit as st
from dotenv import load_dotenv

from agent import FactVerifier, VerificationResult
from agent.cache import cache_stats
from agent.verifier import Verdict

# ── Bootstrap ───────────────────────────────────────────────

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(levelname)-5s  %(message)s",
)

# ── Page Config ─────────────────────────────────────────────

st.set_page_config(
    page_title="FactTrace — AI Fact Verifier",
    page_icon="🔍",
    layout="wide",
)

# ── Verdict Styling ─────────────────────────────────────────

_VERDICT_COLORS: dict[Verdict, str] = {
    Verdict.TRUE: "#22c55e",
    Verdict.MOSTLY_TRUE: "#84cc16",
    Verdict.HALF_TRUE: "#eab308",
    Verdict.MOSTLY_FALSE: "#f97316",
    Verdict.FALSE: "#ef4444",
    Verdict.UNVERIFIABLE: "#94a3b8",
}

_VERDICT_ICONS: dict[Verdict, str] = {
    Verdict.TRUE: "✅",
    Verdict.MOSTLY_TRUE: "🟢",
    Verdict.HALF_TRUE: "🟡",
    Verdict.MOSTLY_FALSE: "🟠",
    Verdict.FALSE: "❌",
    Verdict.UNVERIFIABLE: "❓",
}

_STANCE_ICONS = {
    "supports": "🟩",
    "refutes": "🟥",
    "neutral": "⬜",
}

# ── Custom CSS ──────────────────────────────────────────────

st.markdown(
    """
    <style>
    .verdict-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
    .verdict-label {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }
    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        margin-top: 0.5rem;
    }
    .source-card {
        padding: 0.75rem 1rem;
        border-left: 4px solid #6366f1;
        margin-bottom: 0.5rem;
        border-radius: 0 8px 8px 0;
        background: rgba(99, 102, 241, 0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ──────────────────────────────────────────────────

st.title("🔍 FactTrace")
st.caption("AI-powered multi-source fact verification")

# ── Input ───────────────────────────────────────────────────

claim = st.text_area(
    "Enter a factual claim to verify",
    placeholder="e.g. The Great Wall of China is visible from space with the naked eye.",
    height=100,
)

col_btn, col_cache = st.columns([1, 3])
with col_btn:
    verify_clicked = st.button("🚀  Verify Claim", type="primary", use_container_width=True)
with col_cache:
    stats = cache_stats()
    st.caption(f"Cache: {stats['entries']} entries · {stats['total_bytes'] / 1024:.1f} KB")

# ── Verification Logic ─────────────────────────────────────

if verify_clicked and claim.strip():
    with st.spinner("Searching across Google, News, and Scholar…"):
        verifier = FactVerifier()
        result: VerificationResult = verifier.verify(claim.strip())

    # ── Verdict Display ─────────────────────────────────────
    color = _VERDICT_COLORS.get(result.verdict, "#94a3b8")
    icon = _VERDICT_ICONS.get(result.verdict, "❓")

    st.markdown(
        f"""
        <div class="verdict-box" style="border: 2px solid {color};">
            <div class="verdict-label" style="color: {color};">
                {icon}  {result.verdict.value}
            </div>
            <div style="color: #9ca3af; font-size: 0.9rem;">
                Confidence: {result.confidence_score:.0%}
            </div>
            <div class="confidence-bar"
                 style="background: linear-gradient(90deg, {color} {result.confidence_score * 100}%, #374151 {result.confidence_score * 100}%);">
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Explanation ─────────────────────────────────────────
    st.subheader("📝 Analysis")
    st.write(result.explanation)

    # ── Evidence Sources ────────────────────────────────────
    if result.evidence:
        st.subheader("📚 Evidence Sources")
        for ev in result.evidence:
            stance_icon = _STANCE_ICONS.get(ev.stance, "⬜")
            st.markdown(
                f"""
                <div class="source-card">
                    <strong>{stance_icon} {ev.title}</strong><br/>
                    <span style="color: #9ca3af; font-size: 0.85rem;">
                        {ev.source_engine} · {ev.stance}
                    </span><br/>
                    <span>{ev.snippet}</span><br/>
                    <a href="{ev.url}" target="_blank" style="font-size: 0.8rem;">🔗 {ev.url[:80]}{'…' if len(ev.url) > 80 else ''}</a>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Search Queries Used ─────────────────────────────────
    with st.expander("🔎 Search queries used"):
        for q in result.search_queries_used:
            st.code(q, language=None)

elif verify_clicked:
    st.warning("Please enter a claim to verify.")
