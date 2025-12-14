from app.orbitqa.state import OrbitQAState


def read_request(state: OrbitQAState, user_query: str):
    """
    Initialize the OrbitQA state with a new user query.

    This function serves as the entry point for processing a new user request in the
    OrbitQA workflow. It initializes all state fields with default values and stores
    the user's query after stripping whitespace.

    Parameters
    ----------
    state : OrbitQAState
        The current state object of the OrbitQA workflow. May be empty or contain
        previous conversation context.
    user_query : str
        The raw user query string to be processed. Whitespace will be stripped.

    Returns
    -------
    dict
        A dictionary containing initialized state fields:
        - user_query : str
            The trimmed user query.
        - understood_request : list
            Empty list to store extracted tasks.
        - user_passed_params : dict
            Empty dict to store user-provided parameters.
        - request_action : str
            Default action set to 'toProceed'.
        - to_ask : str
            Empty string for clarification questions.
        - user_clarifications : list
            Empty list to store user's clarification responses.
        - data_query : list
            Empty list for data retrieval queries.
        - retrieved_docs : list
            Empty list for retrieved documentation.
        - tool_sequence : list
            Empty list for planned tool executions.
        - tool_outputs : dict
            Empty dict for tool execution results.
        - plots : list
            Empty list for generated plots.
        - final_response : dict
            Empty dict for the final response to user.
        - warnings : list
            Empty list for any warnings or issues.

    Examples
    --------
    >>> from app.orbitqa.state import OrbitQAState
    >>> state = OrbitQAState()
    >>> result = read_request(state, "  Calculate orbital period for ISS  ")
    >>> result['user_query']
    'Calculate orbital period for ISS'
    >>> result['request_action']
    'toProceed'
    """
    return {
        "user_query": user_query.strip(),
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
