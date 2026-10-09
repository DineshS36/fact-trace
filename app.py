"""
FactTrace — Epistemic Audit Multi-Agent Dashboard
=================================================
High-density Streamlit UI demonstrating an autonomous 4-stage Multi-Agent
architecture (Planner ➔ Retrieval ➔ Critic ➔ Synthesis) powered by
SerpApi and Gemini with semantic relevance gating, CAPTCHA purging,
domain diversity capping, telemetry breakdown, and raw SerpApi inspector.
"""

import logging
import time
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
    page_title="FactTrace | Multi-Agent Epistemic Audit",
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
        padding: 0.8rem 1rem;
        margin-bottom: 0.8rem;
    }
    .entity-pill {
        display: inline-block;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .agent-pipeline-bar {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 1rem;
        flex-wrap: wrap;
    }
    .agent-step {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #a5b4fc;
        padding: 0.3rem 0.7rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
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
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 12px;
        padding: 1.2rem;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    .critic-badge {
        display: inline-block;
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-bottom: 0.6rem;
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
    "kayal": "who is helping Kayal right now in Sun TV serial",
    "void": "is the void will kill us what happened when it touch our earth",
    "creatine": "High-dose creatine supplementation directly causes hair loss in healthy athletes.",
    "sora": "OpenAI launched Sora publicly to all free users in early 2024.",
}

if "input_claim" not in st.session_state:
    st.session_state["input_claim"] = benchmarks["kayal"]

# ── Header ──────────────────────────────────────────────────

col_title, col_stats = st.columns([3, 1])
with col_title:
    st.title("🛡️ FactTrace: Epistemic Multi-Agent")
    st.caption("Autonomous 4-Agent Pipeline: Planner ➔ Retrieval ➔ Epistemic Critic ➔ Synthesis")
with col_stats:
    stats = cache_stats()
    st.markdown(
        f"""
        <div class="metric-card" style="text-align: right;">
            <div style="font-size: 0.75rem; color: #94a3b8;">Zero-Burn Cache</div>
            <div style="font-weight: 700; font-size: 1.1rem; color: #10b981;">{stats['entries']} hits stored</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Multi-Agent Architecture Bar ────────────────────────────

st.markdown(
    """
    <div class="agent-pipeline-bar">
        <div class="agent-step">1️⃣ Planner Agent (Decomposition & Recency Heuristic)</div>
        <div class="agent-step">2️⃣ Retrieval Agent (Cached SerpApi & Payload Store)</div>
        <div class="agent-step">3️⃣ Critic Agent (Domain Diversity Cap & Anti-Scraping Purge)</div>
        <div class="agent-step">4️⃣ Synthesis Agent (AI Overview & Grounded [1] Citations)</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Benchmark Prompts ───────────────────────────────────────

st.markdown("##### ⚡ Benchmark Demonstration Claims")
b1, b2, b3, b4 = st.columns(4)

with b1:
    if st.button("📺 Serial: Kayal Sun TV Current Plot", use_container_width=True):
        st.session_state["input_claim"] = benchmarks["kayal"]
        st.session_state.current_result = None
        st.session_state.audit_history = []
        st.rerun()
with b2:
    if st.button("🌌 Physics: False Vacuum Decay", use_container_width=True):
        st.session_state["input_claim"] = benchmarks["void"]
        st.session_state.current_result = None
        st.session_state.audit_history = []
        st.rerun()
with b3:
    if st.button("🔬 Optimal: Creatine & Hair Loss", use_container_width=True):
        st.session_state["input_claim"] = benchmarks["creatine"]
        st.session_state.current_result = None
        st.session_state.audit_history = []
        st.rerun()
with b4:
    if st.button("⏱️ Temporal: OpenAI Sora 2024", use_container_width=True):
        st.session_state["input_claim"] = benchmarks["sora"]
        st.session_state.current_result = None
        st.session_state.audit_history = []
        st.rerun()

# ── Input Area & Credit Mode ────────────────────────────────

claim = st.text_input(
    "Enter a claim to investigate:",
    value=st.session_state["input_claim"],
    placeholder="e.g., who is helping Kayal right now in Sun TV serial",
)

col_ctrl1, col_ctrl2 = st.columns([1.5, 3.5])
with col_ctrl1:
    credit_mode_choice = st.radio(
        "SerpApi Execution Budget:",
        options=["⚡ Balanced (Strictly 3 Credits)", "🔬 Deep Multi-Surface (6 Credits)"],
        horizontal=True,
    )
    mode_slug = "balanced" if "3 Credits" in credit_mode_choice else "deep"

with col_ctrl2:
    run_audit = st.button("Run Epistemic Multi-Agent Audit", type="primary", use_container_width=False)

if run_audit and claim.strip():
    with st.spinner("Executing Multi-Agent Pipeline: Planning ➔ Retrieving ➔ Purging Noise ➔ Synthesizing Overview..."):
        result = verify_claim(claim.strip(), credit_mode=mode_slug)
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
            st.caption(f"**Engines Corroborated:** {', '.join(engines)} (Mode: `{result.get('credit_mode', 'balanced')}`)")

        # Critic Agent Purge Stats
        purged_count = result.get("purged_sources_count", 0)
        reasons = result.get("purged_reasons", {})
        reasons_text = f" (Purged: {reasons})" if reasons else ""
        st.markdown(
            f"""<div class="critic-badge">🛡️ Critic Agent: Purged {purged_count} noisy/duplicate items</div>""",
            unsafe_allow_html=True,
        )

        # Core Concepts identified by Planner
        concepts = result.get("core_concepts", [])
        if concepts:
            st.markdown("**Gated Concepts:** " + " ".join([f'<span class="kw-tag">{c}</span>' for c in concepts]), unsafe_allow_html=True)

        # 3 Distinct Search Operator Queries (Verified Unconcatenated)
        planner_queries = result.get("search_queries_used", [])
        if planner_queries:
            with st.expander(f"🔍 Planner Agent: {len(planner_queries)} Isolated Queries"):
                labels = ["Primary Authority", "Consensus / Recap", "Counter-Hypothesis / Myth"]
                for i, q in enumerate(planner_queries):
                    label = labels[i] if i < len(labels) else f"Query {i+1}"
                    st.caption(f"**Query {i+1} ({label}):**")
                    st.code(q, language="text")

        # Knowledge Graph
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
                f"""<div class="consensus-card"><strong>📌 Empirical Consensus:</strong> {result['consensus_summary']}</div>""",
                unsafe_allow_html=True,
            )

        if result.get("divergence_detected"):
            st.warning(f"⚠️ **Source Divergence Detected:** {result.get('divergence_notes')}")

        # Related investigation angles
        related = result.get("related_queries", [])
        if related:
            st.markdown("**Related Investigation Angles:**")
            for rq in related[:2]:
                st.markdown(f"- *{rq}*")

    # ── Production Telemetry Bar ────────────────────────────
    telemetry = result.get("telemetry", {})
    if telemetry:
        st.markdown("##### ⏱️ Agent Telemetry & Stage Latency Breakdown")
        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
        col_t1.metric("⚡ Total Latency", f"{telemetry.get('total_latency', 0.0):.2f}s")
        col_t2.metric("🧠 Planner Latency", f"{telemetry.get('planner_latency', 0.0):.2f}s")
        col_t3.metric("🌐 SerpApi Dispatch", f"{telemetry.get('retrieval_latency', 0.0):.2f}s")
        col_t4.metric("🛡️ Critic & Synthesis", f"{(telemetry.get('critic_latency', 0.0) + telemetry.get('synthesis_latency', 0.0)):.2f}s")

    st.divider()

    # ── Grounded Evidence Matrix ────────────────────────────
    st.markdown("### 📚 Grounded Evidence Matrix (Critic-Audited)")
    st.caption("Clean host domains, duplicate-capped (max 2/domain), and ranked via Hybrid Score (70% Topical Relevance + 30% Domain Authority).")

    sources = result.get("sources", [])
    if not sources:
        st.info("No primary sources qualified under topical relevance gating and domain authority filtering.")
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
            clean_dom = src.get("domain", "web")
            rel_score = src.get("relevance_score", 0.0)

            with st.expander(f"[{src.get('index', idx)}] {status_icon} {src.get('title', 'Source')} — {tier_badge}"):
                st.caption(
                    f"**Host Domain:** `{clean_dom}` | **Engine:** `{engine_name}` | "
                    f"**Published:** {pub_date} | **Stance:** `{stance.capitalize()}` | **Relevance:** `{rel_score*100:.0f}%`"
                )
                st.write(f"**Verified Excerpt:** {src.get('snippet')}")

                hw = src.get("highlights") or src.get("highlighted_words", [])
                if hw:
                    kw_html = " ".join([f'<span class="kw-tag">{w}</span>' for w in hw[:5]])
                    st.markdown(f"**SerpApi Matches:** {kw_html}", unsafe_allow_html=True)

                st.markdown(f"🔗 [Inspect Direct URL]({src.get('link')})")

    # ── SerpApi Developer Inspector Drawer & Dossier Export ──
    st.divider()
    col_dossier, col_export = st.columns([2, 1])

    with col_dossier:
        with st.expander("🛠️ Developer Mode: Inspect Raw SerpApi JSON Payload"):
            st.caption("Inspect live structured responses returned from SerpApi engines:")
            raw_payload = result.get("raw_serpapi_payload", {})
            if raw_payload:
                st.json(raw_payload)
            else:
                st.info("No raw SerpApi payload captured for this run.")

    with col_export:
        def generate_markdown_report(res: dict) -> str:
            md = f"# FactTrace Epistemic Audit Report\n\n"
            md += f"**Claim:** {res['claim']}\n\n"
            md += f"**Verdict:** {res['verdict']} | **Confidence:** {res['confidence']*100:.1f}%\n\n"
            if res.get("consensus_summary"):
                md += f"**Empirical Consensus:** {res['consensus_summary']}\n\n"
            md += f"## Executive Analysis\n{res['executive_overview']}\n\n"
            md += f"## Evidence Citations\n"
            for i, s in enumerate(res.get('sources', []), 1):
                md += f"- **[{i}] {s['title']}** (`{s['domain']}`) - *{s['stance']}* (Relevance: {s.get('relevance_score', 0)*100:.0f}%)\n  {s['link']}\n"
            if res.get("search_queries_used"):
                md += f"\n## Injected Search Operators\n"
                for q in res["search_queries_used"]:
                    md += f"- `{q}`\n"
            telem = res.get("telemetry", {})
            if telem:
                md += f"\n## Telemetry\n- Total Latency: {telem.get('total_latency')}s\n- Planner: {telem.get('planner_latency')}s\n- Retrieval: {telem.get('retrieval_latency')}s\n"
            return md

        st.download_button(
            label="📥 Export Audit Dossier (.md)",
            data=generate_markdown_report(result),
            file_name=f"facttrace_audit_{int(time.time())}.md",
            mime="text/markdown",
            use_container_width=True,
        )

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
                                f"**[{f_src.get('index', f_idx)}] {f_src.get('title')}** (`{f_src.get('domain', 'web')}`)\n\n"
                                f"*{f_src.get('snippet')}*\n\n"
                                f"[Inspect Direct Link]({f_src.get('link')})\n"
                            )

    # ── Google AI Mode Follow-up Bar ────────────────────────
    st.divider()
    follow_up = st.chat_input("Ask a follow-up or challenge this audit (e.g. 'Who is the main antagonist in this episode?')...")
    if follow_up:
        with st.spinner(f"Investigating follow-up: '{follow_up}' in context of previous audit..."):
            follow_up_res = verify_follow_up(
                claim=result["claim"],
                follow_up_question=follow_up.strip(),
                previous_result=result,
            )
            st.session_state.audit_history.append(follow_up_res)
            st.rerun()
