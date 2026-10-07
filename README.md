# FactTrace 🔍

AI-powered multi-source fact verification agent that cross-references claims against Google Search, Google News, and Google Scholar, then uses LLM reasoning to deliver a structured verdict with confidence scoring.

## Architecture

```
Claim → Query Expansion → Multi-Engine Search (cached) → Evidence Extraction → LLM Analysis → Verdict
```

| Layer | File | Purpose |
|-------|------|---------|
| **Cache** | `agent/cache.py` | SQLite caching — zero API burn during dev |
| **Search** | `agent/search_tools.py` | SerpApi multi-engine dispatcher |
| **Verifier** | `agent/verifier.py` | Orchestration, LLM analysis, scoring |
| **UI** | `app.py` | Streamlit dashboard |

## Quick Start

```bash
# 1. Clone & enter
git clone https://github.com/DineshS36/fact-trace.git
cd fact-trace

# 2. Create virtual env
python -m venv venv
venv\Scripts\activate  # Windows

# 3. Install deps
pip install -r requirements.txt

# 4. Configure keys
#    Edit .env with your SERPAPI_API_KEY and LLM provider key

# 5. Run
streamlit run app.py
```

## Configuration

Set keys in `.env`:

| Variable | Required | Description |
|----------|----------|-------------|
| `SERPAPI_API_KEY` | Yes | [SerpApi](https://serpapi.com) key |
| `LLM_PROVIDER` | No | `gemini` (default), `openai`, or `groq` |
| `GEMINI_API_KEY` | If gemini | Google Gemini API key |
| `OPENAI_API_KEY` | If openai | OpenAI API key |
| `GROQ_API_KEY` | If groq | Groq API key |

## Verdict Scale

| Verdict | Meaning |
|---------|---------|
| ✅ True | Strong evidence confirms the claim |
| 🟢 Mostly True | Claim is accurate with minor caveats |
| 🟡 Half True | Partially accurate, missing context |
| 🟠 Mostly False | Contains some truth but is misleading |
| ❌ False | Evidence contradicts the claim |
| ❓ Unverifiable | Insufficient evidence to judge |

## License

See [LICENSE](./LICENSE).
