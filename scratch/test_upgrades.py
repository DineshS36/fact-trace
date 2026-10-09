import sys
import time

sys.path.insert(0, r"c:\Desktop\facttrace")

from agent.llm_client import clean_llm_json
from agent.search_tools import (
    extract_serpapi_ai_overview,
    select_engines,
)
from agent.planner import PlannerAgent
from agent.verifier import verify_claim

print("=== 1. TESTING BULLETPROOF LLM JSON PARSER ===")
# Test 1: Clean dict
assert clean_llm_json('{"key": "value"}') == {"key": "value"}

# Test 2: Markdown wrapped dict
wrapped_json = """```json
{
  "verdict": "False",
  "confidence": 0.95
}
```"""
res_wrapped = clean_llm_json(wrapped_json)
assert res_wrapped["verdict"] == "False"
assert res_wrapped["confidence"] == 0.95

# Test 3: Markdown wrapped with preamble commentary
commentary_json = """Here is your epistemic audit result:
```json
{
  "verdict": "Mostly True",
  "queries": ["q1", "q2"]
}
```
Let me know if you need anything else!"""
res_commentary = clean_llm_json(commentary_json)
assert res_commentary["verdict"] == "Mostly True"
assert res_commentary["queries"] == ["q1", "q2"]

# Test 4: JSON array
arr_json = '```json\n["query 1", "query 2", "query 3"]\n```'
res_arr = clean_llm_json(arr_json)
assert res_arr == ["query 1", "query 2", "query 3"]
print("[OK] clean_llm_json passed all markdown and regex extraction tests!")

print("\n=== 2. TESTING POP-CULTURE SCHOLAR SUPPRESSION ===")
pop_claim = "Gudako is the official female protagonist of Fate/Grand Order."
engines_pop_bal = select_engines(pop_claim, mode="balanced")
engines_pop_deep = select_engines(pop_claim, mode="deep")
print(f"Pop claim: '{pop_claim}'")
print(f"  Balanced engines: {engines_pop_bal}")
print(f"  Deep engines: {engines_pop_deep}")
assert "google_scholar" not in engines_pop_bal, "Scholar should be suppressed for pop culture!"
assert "google_scholar" not in engines_pop_deep, "Scholar should be suppressed for pop culture!"

planner = PlannerAgent()
plan_pop = planner.plan(pop_claim)
print(f"  Plan surfaces: {plan_pop.target_surfaces}")
print(f"  Plan claim_type: {plan_pop.claim_type}")
assert "google_scholar" not in plan_pop.target_surfaces, "Scholar leaked into plan surfaces for anime/game query!"

sci_claim = "High-dose creatine supplementation directly causes hair loss in healthy athletes."
engines_sci = select_engines(sci_claim, mode="deep")
print(f"Sci claim: '{sci_claim}' -> Engines: {engines_sci}")
assert "google_scholar" in engines_sci, "Scholar should be included for scientific claim!"
print("[OK] Pop-culture scholar suppression verified!")

print("\n=== 3. TESTING SERPAPI NATIVE AI OVERVIEW EXTRACTION ===")
mock_serp = {
    "ai_overview": {
        "text_blocks": [
            {"type": "paragraph", "snippet": "Gudako is the female protagonist of Fate/Grand Order."},
            {"type": "paragraph", "snippet": "She is also known as Ritsuka Fujimaru."}
        ]
    }
}
overview_txt = extract_serpapi_ai_overview(mock_serp)
print("Extracted overview:", overview_txt)
assert overview_txt == "Gudako is the female protagonist of Fate/Grand Order. She is also known as Ritsuka Fujimaru."
print("[OK] SerpApi AI Overview extraction verified!")

print("\n=== 4. TESTING PARALLEL RETRIEVAL & FULL VERIFICATION ===")
start = time.time()
res = verify_claim(pop_claim, credit_mode="balanced")
elapsed = time.time() - start

print(f"Audit completed in: {elapsed:.2f}s")
print(f"Verdict: {res['verdict']} ({res['confidence']*100:.1f}%)")
telem = res.get("telemetry", {})
print(f"Telemetry: Total={telem.get('total_latency')}s, Planner={telem.get('planner_latency')}s, Retrieval={telem.get('retrieval_latency')}s")
print(f"Engines queried: {res.get('engines_queried')}")
assert "google_scholar" not in res.get("engines_queried", []), "Scholar was queried for anime!"
print("Sources:")
for s in res.get("sources", []):
    print(f"  [{s['index']}] {s['domain']} | Tier {s['authority_tier']} | {s['title'][:50]}")
    assert "books.google" not in s["domain"], "Academic books.google thesis leaked into anime query!"

print("\nALL 4 P0/P1/P2 UPGRADES VALIDATED SUCCESSFULLY!")
