from app.missionops.state import MissionOpsState


def read_request(state: MissionOpsState):
    """
    Initialize the workflow state with the user's request.

    This node serves as the entry point for the MissionOps workflow,
    reading the user's query and initializing all state fields to their
    default values. The user query is stripped of leading/trailing whitespace.

    Parameters
    ----------
    state : MissionOpsState
        The current state of the MissionOps workflow, containing:
        - user_query : str
            The original user request to be processed

    Returns
    -------
    dict
        A dictionary containing initialized state fields:
        - user_query : str
            Cleaned user query with whitespace stripped
        - understood_request : list
            Empty list for extracted tasks
        - user_passed_params : dict
            Empty dict for extracted parameters
        - request_action : str
            Set to "toProceed" to continue workflow
        - to_ask : str
            Empty string for clarification questions
        - user_clarifications : list
            Empty list for user clarifications
        - data_query : list
            Empty list for data retrieval queries
        - retrieved_docs : list
            Empty list for retrieved documents
        - start_execution_from_idx : int
            Set to 0 for tool execution index
        - tool_sequence : list
            Empty list for planned tool invocations
        - tool_outputs : dict
            Empty dict for tool execution results
        - tool_selection_state_done : str
            Empty string for tracking tool planning stages
        - final_response : dict
            Empty dict for final response
        - warnings : list
            Empty list for workflow warnings

    Notes
    -----
    This function resets all workflow state to default values, making it
    suitable for starting a new mission analysis request.
    """
    return {
        "user_query": state["user_query"].strip(),
        "understood_request": [],
        "user_passed_params": {},
        "request_action": "toProceed",
        "to_ask": "",
        "user_clarifications": [],
        "data_query": [],
        "retrieved_docs": [],
        "start_execution_from_idx": 0,
        "tool_sequence": [],
        "tool_outputs": {},
        "tool_selection_state_done": "",
        "final_response": {},
        "warnings": [],
    }
