from langchain_core.messages import HumanMessage

from llm import llm
from schemas import AnswerQuestion, Reflection

# Part A: no model needed. Does the schema itself work?
sample = AnswerQuestion(
    answer="Persistence saves graph state between runs.",
    reflection=Reflection(missing="How to enable it.", superfluous="None."),
    search_queries=["LangGraph persistence checkpointer"],
)
print("PART A:", sample.model_dump())

# Part B: can qwen3:1.7b fill the schema?
structured_llm = llm.with_structured_output(AnswerQuestion)
result = structured_llm.invoke(
    [HumanMessage(content="What does LangGraph persistence do?")]
)
print("PART B type:", type(result).__name__)
print("PART B value:", result)