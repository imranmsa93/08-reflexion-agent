import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage
from chains import responder_chain

result = responder_chain.invoke(
    {"messages": [HumanMessage(content="What does LangGraph persistence do?")]}
)

print("\n## ANSWER:", result.answer)
print("\n## MISSING:", result.reflection.missing)
print("\n## SUPERFLUOUS:", result.reflection.superfluous)
print("\n## QUERIES (raw):", result.search_queries)
print("\n## QUERIES (first 3):", result.search_queries[:3])