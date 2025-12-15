from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.orbitqa.nodes import (
    clarify,
    compute_and_plot,
    deny,
    draft_final_response,
    proceed,
    read_request,
    retrieve,
    retriever,
    tool_selector,
    understand,
)
from app.orbitqa.state import OrbitQAState


def build_graph():
    agent_builder = StateGraph(state_schema=OrbitQAState)

    # Add nodes
    agent_builder.add_node("read_request", read_request)
    agent_builder.add_node("understand", understand)
    agent_builder.add_node("deny", deny)
    agent_builder.add_node("proceed", proceed)
    agent_builder.add_node("clarify", clarify)
    agent_builder.add_node("retrieve", retrieve)
    agent_builder.add_node("retriever", retriever)
    agent_builder.add_node("tool_selector", tool_selector)
    agent_builder.add_node("compute_and_plot", compute_and_plot)
    agent_builder.add_node("draft_final_response", draft_final_response)

    # Add edges
    agent_builder.add_edge(START, "read_request")
    agent_builder.add_edge("read_request", "understand")
    agent_builder.add_conditional_edges(
        "understand",
        lambda state: state["request_action"],
        {"toDeny": "deny", "toClarify": "clarify", "toProceed": "proceed"},
    )
    agent_builder.add_edge("deny", END)
    agent_builder.add_edge("proceed", "retrieve")
    # clarify -> understand loop handled internally via Command
    agent_builder.add_edge("retrieve", "retriever")
    agent_builder.add_edge("retriever", "tool_selector")
    agent_builder.add_edge("tool_selector", "compute_and_plot")
    agent_builder.add_edge("compute_and_plot", "draft_final_response")
    agent_builder.add_edge("draft_final_response", END)

    checkpointer = MemorySaver()

    agent = agent_builder.compile(checkpointer=checkpointer)

    return agent
