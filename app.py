"""
FactTrace — Epistemic Audit Agent Dashboard
===========================================
High-density Streamlit UI demonstrating multi-engine corroboration
powered by SerpApi and Gemini with deep entity extraction and
domain authority filtering.
"""

import logging
import streamlit as st
from dotenv import load_dotenv

from agent.cache import cache_stats
from agent.verifier import verify_claim

# ── Bootstrap ───────────────────────────────────────────────

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(levelname)-5s  %(message)s",
)

st.set_page_config(
    page_title="FactTrace | Epistemic Audit Agent",
    page_icon="🛡️",
    layout="wide",
)

# ── Custom Styling ──────────────────────────────────────────

st.markdown(
    """
    <style>
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    .entity-pill {
        display: inline-block;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }
    .kw-tag {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-right: 0.3rem;
        margin-bottom: 0.3rem;
    }
    .source-container {
        border-left: 3px solid #6366f1;
        padding-left: 1rem;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ──────────────────────────────────────────────────

col_title, col_stats = st.columns([3, 1])
with col_title:
    st.title("🛡️ FactTrace: Epistemic Audit Agent")
    st.caption("Autonomous multi-engine corroboration powered by SerpApi & Gemini")
with col_stats:
    stats = cache_stats()
    st.markdown(
        f"""
        <div class="metric-card" style="text-align: right; padding: 0.6rem 1rem;">
            <div style="font-size: 0.8rem; color: #94a3b8;">Zero-Burn Cache</div>
            <div style="font-weight: 700; font-size: 1.1rem; color: #10b981;">{stats['entries']} hits stored</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Benchmark Prompts ───────────────────────────────────────

st.markdown("##### ⚡ Benchmark Demonstration Claims")
b1, b2, b3 = st.columns(3)

benchmarks = {
    "creatine": "Does dietary creatine intake cause hair loss in healthy adults?",
    "sora": "OpenAI launched Sora publicly to all free users in early 2024.",
    "space_pen": "NASA spent millions developing a space pen while the Soviets just used a pencil.",
}

if "input_claim" not in st.session_state:
    st.session_state["input_claim"] = benchmarks["creatine"]

with b1:
    if st.button("🔬 Myth: Creatine & Hair Loss", use_container_width=True):
        st.session_state["input_claim"] = benchmarks["creatine"]
with b2:
    if st.button("⏱️ Temporal: OpenAI Sora 2024", use_container_width=True):
        st.session_state["input_claim"] = benchmarks["sora"]
with b3:
    if st.button("🚀 Attribution: NASA Space Pen", use_container_width=True):
        st.session_state["input_claim"] = benchmarks["space_pen"]

# ── Input Area ──────────────────────────────────────────────

claim = st.text_input(
    "Enter a claim to investigate:",
    value=st.session_state["input_claim"],
    placeholder="e.g., Does creatine cause hair loss?",
)

run_audit = st.button("Run Epistemic Audit", type="primary", use_container_width=False)

if run_audit and claim.strip():
    with st.spinner("Dispatching multi-engine searches & auditing evidence across Google, News & Scholar..."):
        result = verify_claim(claim.strip())

    verdict_raw = result["verdict"].upper()
    confidence = result["confidence"]

    # ── Overview Section ────────────────────────────────────
    col1, col2 = st.columns([1, 2.5])

    with col1:
        badge_color = (
            "green"
            if "TRUE" in verdict_raw and "MOSTLY" not in verdict_raw and "FALSE" not in verdict_raw
            else "red"
            if "FALSE" in verdict_raw
            else "orange"
        )
        st.markdown(f"### Verdict: :{badge_color}[{result['verdict']}]")
        st.metric("Confidence Calibration", f"{confidence * 100:.1f}%")
        
        engines = result.get("engines_queried", [])
        if engines:
            st.caption(f"**Engines Corroborated:** {', '.join(engines)}")
        
        # Knowledge Graph deep field display
        kg = result.get("knowledge_graph")
        if kg and kg.get("title"):
            st.markdown(
                f"""
                <div class="entity-pill">✓ Entity Verified: {kg.get('title')}</div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.5rem;">
                    {kg.get('type')}: {kg.get('description')[:120]}
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("### Executive Analysis")
        st.write(result["explanation"])

        # Related investigation angles from SerpApi
        related = result.get("related_queries", [])
        if related:
            st.markdown("**Related Investigation Angles:**")
            for rq in related[:2]:
                st.markdown(f"- *{rq}*")

    st.divider()

    # ── Source Corroboration Matrix ─────────────────────────
    st.markdown("### Source Corroboration Matrix")
    st.caption("Filtered via Domain Authority Filter (banned social/forums dropped) & annotated with SerpApi highlights.")

    sources = result.get("sources", [])
    if not sources:
        st.info("No primary sources qualified under the domain authority filter.")
    else:
        for src in sources:
            stance = src.get("stance", "neutral").lower()
            status_icon = "🟩" if stance == "supports" else "🟥" if stance == "refutes" else "🟨"
            engine_name = src.get("engine", "google")
            
            with st.expander(f"{status_icon} {src.get('title', 'Source')} ({engine_name})"):
                st.write(f"**Snippet:** {src.get('snippet')}")
                
                # Highlighted words deep field
                hw = src.get("highlighted_words", [])
                if hw:
                    kw_html = " ".join([f'<span class="kw-tag">{w}</span>' for w in hw[:5]])
                    st.markdown(f"**SerpApi Highlight Matches:** {kw_html}", unsafe_allow_html=True)
                
                st.markdown(f"🔗 [Direct Source Link]({src.get('link')})")

elif run_audit:
    st.warning("Please enter a claim to investigate.")
