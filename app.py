"""
FactTrace — Epistemic Audit Agent Dashboard
===========================================
High-density Streamlit UI demonstrating multi-engine corroboration
powered by SerpApi and Gemini with deep entity extraction, tiered domain
authority ranking, grounded AI Overviews, and interactive follow-up investigations.
"""

import logging
import streamlit as st
from dotenv import load_dotenv

from agent.cache import cache_stats
from agent.verifier import verify_claim, verify_follow_up

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
    .consensus-card {
        background: rgba(16, 185, 129, 0.08);
        border-left: 4px solid #10b981;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin: 0.8rem 0;
        font-size: 0.95rem;
    }
    .ai-overview-container {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 1.2rem;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    .followup-box {
        background: rgba(99, 102, 241, 0.05);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.8rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session State Initialization ────────────────────────────

if "audit_history" not in st.session_state:
    st.session_state.audit_history = []

if "current_result" not in st.session_state:
    st.session_state.current_result = None

benchmarks = {
    "creatine": "High-dose creatine supplementation directly causes hair loss in healthy athletes.",
    "sora": "OpenAI launched Sora publicly to all free users in early 2024.",
    "space_pen": "NASA spent millions developing a space pen while the Soviets just used a pencil.",
}

if "input_claim" not in st.session_state:
    st.session_state["input_claim"] = benchmarks["creatine"]

# ── Header ──────────────────────────────────────────────────

col_title, col_stats = st.columns([3, 1])
with col_title:
    st.title("🛡️ FactTrace: Epistemic Audit Agent")
    st.caption("Google AI Mode Epistemic Corroboration Engine powered by SerpApi & Gemini")
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

with b1:
    if st.button("🔬 Optimal Demo: Creatine & Hair Loss", use_container_width=True):
        st.session_state["input_claim"] = benchmarks["creatine"]
        st.session_state.current_result = None
        st.session_state.audit_history = []
        st.rerun()
with b2:
    if st.button("⏱️ Temporal: OpenAI Sora 2024", use_container_width=True):
        st.session_state["input_claim"] = benchmarks["sora"]
        st.session_state.current_result = None
        st.session_state.audit_history = []
        st.rerun()
with b3:
    if st.button("🚀 Attribution: NASA Space Pen", use_container_width=True):
        st.session_state["input_claim"] = benchmarks["space_pen"]
        st.session_state.current_result = None
        st.session_state.audit_history = []
        st.rerun()

# ── Input Area ──────────────────────────────────────────────

claim = st.text_input(
    "Enter a claim to investigate:",
    value=st.session_state["input_claim"],
    placeholder="e.g., High-dose creatine supplementation directly causes hair loss in healthy athletes.",
)

col_btn, _ = st.columns([1, 4])
with col_btn:
    run_audit = st.button("Run Epistemic Audit", type="primary", use_container_width=True)

if run_audit and claim.strip():
    with st.spinner("Decomposing queries with search operators, dispatching multi-engine searches & auditing evidence..."):
        result = verify_claim(claim.strip())
        st.session_state.current_result = result
        st.session_state.audit_history = []
elif run_audit:
    st.warning("Please enter a claim to investigate.")

# ── Results Render ──────────────────────────────────────────

result = st.session_state.current_result

if result:
    verdict_raw = str(result["verdict"]).upper()
    confidence = result["confidence"]

    # ── Overview Section ────────────────────────────────────
    col1, col2 = st.columns([1, 2.6])

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

        # Targeted queries used by search planner
        planner_queries = result.get("search_queries_used", [])
        if planner_queries:
            with st.expander("🔍 Injected Search Operators"):
                for q in planner_queries:
                    st.code(q, language="text")

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
        st.markdown("### 🤖 Google AI Mode Overview")
        overview_text = result.get("executive_overview") or result.get("explanation", "")
        st.markdown(
            f"""<div class="ai-overview-container">{overview_text}</div>""",
            unsafe_allow_html=True,
        )

        if result.get("consensus_summary"):
            st.markdown(
                f"""<div class="consensus-card"><strong>📌 Consensus:</strong> {result['consensus_summary']}</div>""",
                unsafe_allow_html=True,
            )

        if result.get("divergence_detected"):
            st.warning(f"⚠️ **Source Divergence Detected:** {result.get('divergence_notes')}")

        # Related investigation angles from SerpApi
        related = result.get("related_queries", [])
        if related:
            st.markdown("**Related Investigation Angles:**")
            for rq in related[:2]:
                st.markdown(f"- *{rq}*")

    st.divider()

    # ── Grounded Evidence Matrix ────────────────────────────
    st.markdown("### 📚 Grounded Evidence Matrix")
    st.caption("Ranked by Tier-1/2 Domain Authority (.gov, .edu, Nature, wires elevated; social/forums dropped) & annotated with bracketed citations.")

    sources = result.get("sources", [])
    if not sources:
        st.info("No primary sources qualified under the domain authority filter.")
    else:
        for idx, src in enumerate(sources, 1):
            tier = src.get("authority_tier", 3)
            tier_badge = (
                "🏛️ Primary Authority (Tier 1)"
                if tier == 1
                else "📰 Established Press (Tier 2)"
                if tier == 2
                else "🌐 General Source (Tier 3)"
            )
            stance = src.get("stance", "neutral").lower()
            status_icon = "🟩" if stance == "supports" else "🟥" if stance == "refutes" else "🟨"
            engine_name = src.get("engine", "google")
            pub_date = src.get("date", "N/A")
            domain_name = src.get("domain", "")

            with st.expander(f"[{idx}] {status_icon} {src.get('title', 'Source')} — {tier_badge}"):
                domain_str = f" | Domain: `{domain_name}`" if domain_name else ""
                st.write(f"**Authority:** {src.get('authority_label', 'Standard')}")
                st.caption(f"Engine: `{engine_name}` | Published: {pub_date}{domain_str} | Stance: `{stance.capitalize()}`")
                st.write(f"**Verified Excerpt:** {src.get('snippet')}")

                # Highlighted words deep field
                hw = src.get("highlights") or src.get("highlighted_words", [])
                if hw:
                    kw_html = " ".join([f'<span class="kw-tag">{w}</span>' for w in hw[:5]])
                    st.markdown(f"**SerpApi Highlight Matches:** {kw_html}", unsafe_allow_html=True)

                st.markdown(f"🔗 [Inspect Raw URL]({src.get('link')})")

    # ── Conversational Follow-up History ────────────────────
    if st.session_state.audit_history:
        st.divider()
        st.markdown("### 💬 Interactive Deep-Dive Follow-ups")
        for turn_idx, turn in enumerate(st.session_state.audit_history, 1):
            with st.container():
                st.markdown(
                    f"""
                    <div class="followup-box">
                        <div style="color: #818cf8; font-weight: 600; font-size: 0.9rem; margin-bottom: 0.3rem;">
                            Turn {turn_idx}: Follow-up Question
                        </div>
                        <div style="font-size: 1.05rem; font-weight: 500; margin-bottom: 0.6rem;">
                            "{turn.get('follow_up_question')}"
                        </div>
                        <div style="line-height: 1.5; margin-bottom: 0.5rem;">
                            {turn.get('answer')}
                        </div>
                        <div style="font-size: 0.85rem; color: #10b981;">
                            <strong>Key Finding:</strong> {turn.get('key_finding')} ({turn.get('status', 'clarifies').capitalize()} audit)
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if turn.get("sources"):
                    with st.expander(f"Inspect {len(turn['sources'])} Follow-up Grounding Sources"):
                        for f_idx, f_src in enumerate(turn["sources"], 1):
                            st.markdown(
                                f"**[{f_idx}] {f_src.get('title')}** ({f_src.get('authority_label')})\n\n"
                                f"*{f_src.get('snippet')}*\n\n"
                                f"[Inspect Link]({f_src.get('link')})\n"
                            )

    # ── Google AI Mode Follow-up Bar ────────────────────────
    st.divider()
    follow_up = st.chat_input("Ask a follow-up or challenge this audit (e.g. 'What about the 2009 rugby study?')...")
    if follow_up:
        with st.spinner(f"Investigating follow-up: '{follow_up}' in context of previous audit..."):
            follow_up_res = verify_follow_up(
                claim=result["claim"],
                follow_up_question=follow_up.strip(),
                previous_result=result,
            )
            st.session_state.audit_history.append(follow_up_res)
            st.rerun()
