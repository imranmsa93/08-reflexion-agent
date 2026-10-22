from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

RESPONDER_SYSTEM = (
    "You are an expert researcher. Fill in the structured fields as follows.\n"
    "- answer: 2 to 4 plain sentences answering the question. "
    "ONLY the answer. Never put the critique or search queries here.\n"
    "- reflection.missing: ONE sentence naming the most important missing fact.\n"
    "- reflection.superfluous: ONE sentence about what is unnecessary, or 'nothing'.\n"
    "- search_queries: 1 to 3 queries written as plain words separated by spaces, "
    "for example: LangGraph checkpointer persistence. No hyphens.\n"
    "If you are not sure about a fact, say no in the answer instead of guessing."
)

responder_prompt = ChatPromptTemplate.from_messages([
    ("system", RESPONDER_SYSTEM),
    MessagesPlaceholder(variable_name="messages"),
])

REVISOR_SYSTEM = (
    "You are an expert researcher revising an answer. You will receive the question, "
    "you previous answer, and web search results. Fill in the fields:\n"
    "- answer: 2 to 4 plain sentences. Rebuild the answer from the search results. "
    "Use ONLY facts the search results support, and delete any sentence from the "
    "previous answer that they do not support. Explain the core concept first. "
    "ONLY the answer here.\n"
    "- reflection.missing: ONE sentence on what is still missing from YOUR NEW answer.\n"
    "- reflection.superfluous: ONE sentence on any claim in YOUR NEW answer that the "
    "search results do not support, or 'nothing'.\n"
    "- search_queries: 1 to 3 queries as plain words separated by spaces, "
    "for anything still unclear.\n"
    "- references: only URLs that appear in the search results. Never invent a URL.\n"
    "Rules: mention a vendor-specific or third-party example only if the question "
    "asks for it. Prefer official documentation. Do not mention any source or example "
    "that is not in the search results."
)

revisor_prompt = ChatPromptTemplate.from_messages([
    ("system", REVISOR_SYSTEM),
    MessagesPlaceholder(variable_name="messages"),
])



