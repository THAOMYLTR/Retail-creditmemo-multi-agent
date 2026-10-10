"""Verify the conversation flow is gone and nothing regressed."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.config import Config
from src.agents.supervisor import DEFAULT_ROUTE, Supervisor

failures = []
sup = Supervisor(Config())

# --- 1. graph shape ----------------------------------------------------------
g = sup.workflow_graph.get_graph()
nodes = {n for n in g.nodes if n not in ("__start__", "__end__")}
edges = [(e.source, e.target, getattr(e, "data", None)) for e in g.edges]
print("1. Graph")
print(f"   nodes: {sorted(nodes)}")
if "conversation" in nodes:
    failures.append("conversation node still in graph")
out_edges = [(s, t, d) for s, t, d in edges if s == "discover_documents"]
print(f"   discover_documents ->: {out_edges}")
if not any(d == "out_of_scope" for _, _, d in out_edges):
    failures.append("no out_of_scope edge from discover_documents")

# --- 2. out-of-scope request costs nothing -----------------------------------
print()
print("2. 'Xin chào' với input_paths=[]")
from src.config import shared_rate_limiter

before = shared_rate_limiter().acquired_count
res = sup.process("Xin chào", [], [])
spent = shared_rate_limiter().acquired_count - before
print(f"   agent_name      : {res['agent_name']}")
print(f"   LLM calls spent : {spent}")
running = [s for s in res["steps"] if s.startswith("Running")]
print(f"   steps           : {res['steps']}")
if res["agent_name"] != "OUT_OF_SCOPE":
    failures.append(f"expected OUT_OF_SCOPE, got {res['agent_name']}")
if spent != 0:
    failures.append(f"expected 0 LLM calls, got {spent}")
if running:
    failures.append(f"no agent should run: {running}")
if "thẩm định tín dụng khách hàng cá nhân" not in res["response"]:
    failures.append("out-of-scope response missing its explanation")

# --- 3. regression: intent without documents still reaches the gap check -----
print()
print("3. 'phân tích thu nhập' không kèm tài liệu (không được bị nuốt)")
before2 = shared_rate_limiter().acquired_count
res2 = sup.process("Hãy phân tích thu nhập và khả năng trả nợ của khách hàng này", [], [])
print(f"   agent_name      : {res2['agent_name']}")
print(f"   LLM calls spent : {shared_rate_limiter().acquired_count - before2}")
print(f"   missing         : {[m['type'] for m in res2.get('gap_analysis', {}).get('missing_evidence', [])]}")
if res2["agent_name"] != "EVIDENCE_GAP_CHECK":
    failures.append(f"expected EVIDENCE_GAP_CHECK, got {res2['agent_name']}")
if "income_documents" not in str(res2.get("gap_analysis", {})):
    failures.append("gap analysis lost its specific missing-document list")

# --- 4. every remaining route still maps to a branch -------------------------
print()
print("4. Route -> workflow_mode -> extraction passes")
branch_nodes = nodes
for route in Supervisor.ROUTE_AGENTS:
    mode = Supervisor._workflow_mode_for_route(route)
    passes = sorted(Supervisor._passes_needed_for_route(route))
    ok = mode in branch_nodes
    print(f"   {route:28} -> {mode:28} {passes or '(none)'}  {'OK' if ok else 'NO BRANCH'}")
    if not ok:
        failures.append(f"{route} -> {mode} has no graph node")

# unknown route must fall back to something real
unknown_mode = Supervisor._workflow_mode_for_route("NOPE")
print(f"   {'(unknown route)':28} -> {unknown_mode}")
if unknown_mode not in branch_nodes:
    failures.append(f"unknown route falls back to {unknown_mode}, which has no node")
print(f"   DEFAULT_ROUTE = {DEFAULT_ROUTE}")

# --- 5. fallbacks never return a dead route ----------------------------------
print()
print("5. Fallback không bao giờ trả route đã bị xoá")
valid = set(Supervisor.ROUTE_AGENTS)
for q in ["Xin chào", "abcxyz", "", "Bạn là ai?"]:
    for has_file in (True, False):
        fb = sup._fallback_route(q, has_file)
        ov = sup._override_route("GARBAGE_ROUTE", q, has_file, set())
        for name, val in (("fallback", fb), ("override", ov)):
            if val not in valid:
                failures.append(f"{name}({q!r}, has_file={has_file}) -> {val}")
print("   kiểm tra xong")

print()
if failures:
    print("FAILURES:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("All checks passed.")
