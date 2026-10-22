from langchain_core.messages import HumanMessage

from chains import responder_chain, revisor_chain
from state import ReflexionState
from tools import perform_search


def responder_node(state: ReflexionState):
    result = responder_chain.invoke(
        {"messages": [HumanMessage(content=state["question"])]}
    )
    return {
        "answer": result.answer,
        "missing": result.reflection.missing,
        "superfluous": result.reflection.superfluous,
        "search_queries": result.search_queries[:3],
        "rounds": 0,
    }


def execute_tools_node(state: ReflexionState):
    return {"search_results": perform_search(state["search_queries"])}


def revisor_node(state: ReflexionState):
    text = (
        f"Question:\n{state['question']}\n\n"
        f"Previous answer:\n{state['answer']}\n\n"
        f"Search results:\n{state['search_results']}"
    )
    result = revisor_chain.invoke({"messages": [HumanMessage(content=text)]})

    # Keep only URLs that really appear in the search results
    valid_refs = [u for u in result.references if u in state["search_results"]]

    return {
        "answer": result.answer,
        "missing": result.reflection.missing,
        "superfluous": result.reflection.superfluous,
        "search_queries": result.search_queries[:3],
        "references": valid_refs,
        "rounds": state.get("rounds", 0) + 1,
    }