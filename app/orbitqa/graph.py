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
    """
    Construct and compile the OrbitQA workflow graph.

    This function builds the complete LangGraph state machine for the OrbitQA
    aerospace question-answering system. It defines all workflow nodes, edges,
    and control flow logic, then compiles the graph with a memory checkpointer
    for state persistence across interactions.

    The workflow implements a multi-stage pipeline:
    1. read_request: Initialize state with user query
    2. understand: Parse and analyze the request
    3. [deny/clarify/proceed]: Branch based on understanding
    4. retrieve: Generate knowledge retrieval queries
    5. retriever: Execute vector database search
    6. tool_selector: Plan computational tool sequence
    7. compute_and_plot: Execute tools and generate plots
    8. draft_final_response: Format final user response

    Returns
    -------
    langgraph.graph.CompiledStateGraph
        A compiled state graph ready for execution with:
        - All workflow nodes registered
        - Conditional and direct edges configured
        - MemorySaver checkpointer attached for state persistence
        - Support for interrupts and multi-turn clarification

    Notes
    -----
    The graph supports three control flow paths from the understand node:
    - toDeny: Reject unsafe/out-of-scope requests
    - toClarify: Request additional information (creates clarify->understand loop)
    - toProceed: Continue with normal workflow execution

    The clarify node uses LangGraph's Command mechanism to loop back to
    understand, enabling multi-turn clarification dialogues.

    The MemorySaver checkpointer enables:
    - State persistence across function calls
    - Support for interrupts and resumption
    - Conversation history tracking
    - Thread-based session management

    Examples
    --------
    >>> from app.orbitqa.graph import build_graph
    >>> graph = build_graph()
    >>> config = {"configurable": {"thread_id": "user-123", ...}}
    >>> result = graph.invoke({"user_query": "Calculate orbital period"}, config)

    See Also
    --------
    OrbitQAState : The state schema used by the graph.
    app.orbitqa.orbitqa_app.main : Main entry point that invokes the graph.
    """
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
