import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from graph import graph

print(graph.get_graph().draw_mermaid())

state = {"question": "What does LangGraph persistence do?"}

for step in graph.stream(state, stream_mode="updates"):
    for node_name, update in step.items():
        print(f"-> {node_name} wrote: {list(update.keys())}")
        state.update(update)      # merge, as LangGraph does

print("\nFINAL ANSWER:", state["answer"])
print("REFERENCES:", state["references"])
print("ROUNDS:", state["rounds"])