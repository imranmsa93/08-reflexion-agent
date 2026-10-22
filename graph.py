from langgraph.graph import StateGraph, START, END

from config import EXECUTE_TOOLS, MAX_ROUNDS, RESPONDER, REVISOR
from nodes import execute_tools_node, responder_node, revisor_node
from state import ReflexionState


def should_continue(state: ReflexionState):
    if state["rounds"] >= MAX_ROUNDS:
        return END
    return EXECUTE_TOOLS


builder = StateGraph(ReflexionState)

builder.add_node(RESPONDER, responder_node)
builder.add_node(EXECUTE_TOOLS, execute_tools_node)
builder.add_node(REVISOR, revisor_node)

builder.add_edge(START, RESPONDER)
builder.add_edge(RESPONDER, EXECUTE_TOOLS)
builder.add_edge(EXECUTE_TOOLS, REVISOR)
builder.add_conditional_edges(REVISOR, should_continue, [EXECUTE_TOOLS, END])

graph = builder.compile()