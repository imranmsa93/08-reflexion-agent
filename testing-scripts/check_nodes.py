import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nodes import responder_node, execute_tools_node, revisor_node

state = {"question": "What does LangGraph persistence do?"}


def run(label, node):
    update = node(state)
    print(f"\n=== {label} returned these keys: {list(update.keys())}")
    state.update(update)        # mimics LangGraph: new values replace old ones


run("responder", responder_node)
first_answer = state["answer"]
print("ANSWER 1:", first_answer)
print("QUERIES:", state["search_queries"])

run("execute_tools", execute_tools_node)
print("SEARCH RESULTS (first 300 chars):", state["search_results"][:300])

run("revisor", revisor_node)
print("ANSWER 2:", state["answer"])
print("REFERENCES:", state["references"])
print("ROUNDS:", state["rounds"])

print("\nAnswer was replaced:", state["answer"] != first_answer)