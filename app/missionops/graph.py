from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.missionops.nodes import (
    clarify,
    deny,
    draft_final_response,
    mission_summary,
    orbit_propagator_visibility,
    power_comm_thermal_duty,
    proceed,
    re_eval,
    read_request,
    retrieve,
    retriever,
    sun_eclipse,
    tool_executor,
    understand,
    validator,
)
from app.missionops.state import MissionOpsState


def build_graph():
    agent_builder = StateGraph(state_schema=MissionOpsState)

    # Add nodes
    agent_builder.add_node("read_request", read_request)
    agent_builder.add_node("understand", understand)
    agent_builder.add_node("deny", deny)
    agent_builder.add_node("proceed", proceed)
    agent_builder.add_node("clarify", clarify)
    agent_builder.add_node("retrieve", retrieve)
    agent_builder.add_node("retriever", retriever)
    agent_builder.add_node("re_evaulate", re_eval)
    agent_builder.add_node("orbit_propagator_visibility", orbit_propagator_visibility)
    agent_builder.add_node("sun_eclipse", sun_eclipse)
    agent_builder.add_node("power_comm_thermal_duty", power_comm_thermal_duty)
    agent_builder.add_node("mission_summary", mission_summary)
    agent_builder.add_node("tool_executor", tool_executor)
    agent_builder.add_node("validator", validator)
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
    agent_builder.add_edge("retriever", "re_evaulate")
    agent_builder.add_edge("re_evaulate", "orbit_propagator_visibility")
    agent_builder.add_edge("orbit_propagator_visibility", "tool_executor")
    agent_builder.add_edge("sun_eclipse", "tool_executor")
    agent_builder.add_edge("power_comm_thermal_duty", "tool_executor")
    agent_builder.add_edge("mission_summary", "tool_executor")

    agent_builder.add_edge("tool_executor", "validator")

    agent_builder.add_conditional_edges(
        "validator", lambda state: validator_routing(state)
    )

    agent_builder.add_edge("draft_final_response", END)

    checkpointer = MemorySaver()

    agent = agent_builder.compile(checkpointer=checkpointer)

    return agent


def validator_routing(state: MissionOpsState):
    if state["request_action"] == "toDeny":
        return END
    elif state["tool_selection_state_done"] == "orbit_propagator_visibility":
        return "sun_eclipse"
    elif state["tool_selection_state_done"] == "sun_eclipse":
        return "power_comm_thermal_duty"
    elif state["tool_selection_state_done"] == "power_comm_thermal_duty":
        return "mission_summary"
    elif state["tool_selection_state_done"] == "mission_summary":
        return "draft_final_response"
