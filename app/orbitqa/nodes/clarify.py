from langgraph.types import Command, interrupt

from app.orbitqa.state import OrbitQAState


def clarify(state: OrbitQAState):
    """
    Request clarification from the user when the query is ambiguous or incomplete.

    This function interrupts the workflow execution to collect additional information
    from the user. It is invoked when the understand module determines that the
    request requires clarification (request_action == 'toClarify'). After receiving
    the user's response, it routes back to the understand node for re-evaluation
    with the additional context.

    The function uses LangGraph's interrupt mechanism to pause execution and wait
    for user input. The clarification is appended to the user_clarifications list
    in the state, allowing the system to maintain a history of all clarifying
    exchanges.

    Parameters
    ----------
    state : OrbitQAState
        The current workflow state containing:
        - to_ask : str or None
            The specific clarification question to ask, or None to use default prompt.
        - user_clarifications : list of str
            Existing list of clarifications from previous interactions.

    Returns
    -------
    Command
        A LangGraph Command object that:
        - Updates the state with the new clarification appended to user_clarifications.
        - Routes workflow back to the 'understand' node for re-processing.

    Notes
    -----
    This function implements an interactive multi-turn dialogue pattern, allowing
    the system to iteratively refine its understanding of complex or ambiguous
    aerospace queries.

    Common scenarios requiring clarification include:
    - Missing orbital parameters (e.g., "Calculate transfer orbit" without specifying radii)
    - Ambiguous celestial body references (e.g., "orbit" without specifying planet)
    - Unclear time specifications (e.g., "propagate orbit" without duration)
    - Multiple interpretation possibilities (e.g., "delta-v" could refer to various maneuvers)

    The workflow may pass through this node multiple times if successive
    clarifications are still insufficient.

    Examples
    --------
    >>> from app.orbitqa.state import OrbitQAState
    >>> state = OrbitQAState(
    ...     user_query="Calculate the transfer",
    ...     to_ask="Which transfer type: Hohmann, bi-elliptic, or general?",
    ...     user_clarifications=[],
    ...     request_action="toClarify"
    ... )
    >>> # User would be prompted: "Which transfer type: Hohmann, bi-elliptic, or general?"
    >>> # Assuming user responds "Hohmann"
    >>> command = clarify(state)
    >>> # command.update["user_clarifications"] == ["Hohmann"]
    >>> # command.goto == "understand"

    With default prompt:
    >>> state = OrbitQAState(
    ...     user_query="Compute something",
    ...     to_ask=None,
    ...     user_clarifications=["Previous clarification"],
    ...     request_action="toClarify"
    ... )
    >>> # User would be prompted: "Your request is unclear. Please provide more clarification:"
    >>> # Assuming user responds "I want orbital period"
    >>> command = clarify(state)
    >>> # command.update["user_clarifications"] == ["Previous clarification", "I want orbital period"]

    See Also
    --------
    understand : Module that determines when clarification is needed.
    proceed : Alternative path when request is sufficiently clear.
    interrupt : LangGraph function for pausing workflow execution.
    """
    user_input = interrupt(
        state["to_ask"]
        if state["to_ask"]
        else "Your request is unclear. Please provide more clarification:"
    )

    return Command(
        update={
            "user_clarifications": state["user_clarifications"]
            + [str(user_input).strip()]
        },
        goto="understand",
    )
