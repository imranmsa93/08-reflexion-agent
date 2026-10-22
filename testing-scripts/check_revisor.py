import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage
from chains import revisor_chain

SAMPLE_INPUT = """Question:
What does LangGraph persistence do?

Previous answer:
LangGraph persistence refers to saving the model's state, such as parameters or training data.


Search results (sample test data):
[1] Source: https://docs.langchain.com/oss/python/langgraph/persistence
LangGraph persistence uses checkpointers to save a snapshot of the graph state at
every step, grouped into a thread. This enables memory across runs, human-in-the-loop
pauses, and recovery from failures.
"""

result = revisor_chain.invoke({"messages": [HumanMessage(content=SAMPLE_INPUT)]})

print("ANSWER:", result.answer)
print("MISSING:", result.reflection.missing)
print("SUPERFLUOUS:", result.reflection.superfluous)
print("QUERIES:", result.search_queries)
print("REFERENCES:", result.references)