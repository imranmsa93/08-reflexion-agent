import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage

from chains import responder_chain
from state import ReflexionState

result = responder_chain.invoke(
    {"messages": [HumanMessage(content="What does LangGraph persistence do?")]}
)

print("TYPE OF result:", type(result).__name__)
print("FIELDS IN THE FORM :", list(result.model_dump().keys()))
print("KEYS IN THE STATE  :", list(ReflexionState.__annotations__.keys()))

in_form_not_state = set(result.model_dump().keys()) - set(ReflexionState.__annotations__.keys())
print("IN THE FORM BUT NOT A STATE KEY:", in_form_not_state)