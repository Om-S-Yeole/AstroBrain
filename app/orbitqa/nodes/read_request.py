from app.orbitqa.state import OrbitQAState


def read_request(state: OrbitQAState):
    """
    Initialize the OrbitQA workflow state with a new user query.

    This function serves as the entry point for processing a new user request in the
    OrbitQA workflow. It reads the user query from the state, strips whitespace, and
    initializes all other state fields with default/empty values, preparing the workflow
    for processing.

    This is the first node in the OrbitQA graph and sets up the initial state that will
    be passed through the workflow pipeline (understand → proceed/clarify/deny → retrieve →
    compute → plot → draft_final_response).

    Parameters
    ----------
    state : OrbitQAState
        The current state object of the OrbitQA workflow containing the user's query
        in the `user_query` field. The query string will be trimmed of leading/trailing
        whitespace.

    Returns
    -------
    dict
        A dictionary containing initialized state fields for the workflow:

        - user_query : str
            The user's query with leading/trailing whitespace removed.
        - understood_request : list
            Empty list, will be populated by the understand module with extracted tasks.
        - user_passed_params : dict
            Empty dictionary, will store user-provided parameter values.
        - request_action : str
            Set to 'toProceed' by default, indicating the workflow should proceed to
            the next stage.
        - to_ask : str
            Empty string, will contain clarification questions if needed.
        - user_clarifications : list
            Empty list, will store user responses to clarification questions.
        - data_query : list
            Empty list, will contain semantic search queries for documentation retrieval.
        - retrieved_docs : list
            Empty list, will store retrieved reference documentation.
        - tool_sequence : list
            Empty list, will contain the ordered sequence of tools to execute.
        - tool_outputs : dict
            Empty dictionary, will map tool IDs to their execution results.
        - plots : list
            Empty list, will contain generated visualization outputs.
        - final_response : dict
            Empty dictionary, will contain the structured final response to the user.
        - warnings : list
            Empty list, will accumulate any warnings or issues during workflow execution.

    Notes
    -----
    This function performs minimal processing - it only trims the query string and
    initializes empty containers. The actual query understanding and task extraction
    happens in subsequent workflow nodes.

    The function returns a state mutation dictionary that LangGraph will merge into
    the existing state, allowing stateful progression through the workflow.

    Examples
    --------
    >>> from app.orbitqa.state import OrbitQAState
    >>> # Create initial state with user query
    >>> state = OrbitQAState(user_query="  Calculate orbital period for ISS  ")
    >>> result = read_request(state)
    >>> result['user_query']
    'Calculate orbital period for ISS'
    >>> result['request_action']
    'toProceed'
    >>> result['understood_request']
    []
    >>> result['warnings']
    []

    >>> # State after processing
    >>> state = OrbitQAState(user_query="Plot Hohmann transfer\\n")
    >>> result = read_request(state)
    >>> result['user_query']
    'Plot Hohmann transfer'
    >>> len(result.keys())
    14

    See Also
    --------
    OrbitQAState : The state model for the OrbitQA workflow.
    understand : Next node in the workflow that interprets the user query.
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
        "tool_sequence": [],
        "tool_outputs": {},
        "plots": [],
        "final_response": {},
        "warnings": [],
    }
