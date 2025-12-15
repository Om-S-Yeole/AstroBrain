from app.orbitqa.state import OrbitQAState


def deny(state: OrbitQAState):
    """
    Generate a denial response for unsafe or out-of-scope requests.

    This function is called when a user's request has been determined to be unsafe,
    unethical, illegal, or outside the supported aerospace and space-systems domain.
    It creates a structured denial response and logs the denial in the warnings list.

    Parameters
    ----------
    state : OrbitQAState
        The current workflow state containing the request context and any existing warnings.

    Returns
    -------
    dict
        A dictionary containing state mutations:
        - final_response : dict
            Contains 'status', 'reason', and 'message' keys explaining the denial.
            Status is always 'denied'.
        - warnings : list
            Updated warnings list with a denial notification appended.

    Notes
    -----
    This function is typically invoked when the understand module sets
    request_action to 'toDeny'. Common reasons for denial include:

    - Unsafe content (self-harm, violence, illegal activities)
    - Unethical requests
    - Non-consensual actions
    - Requests completely unrelated to aerospace, physics, or engineering domains

    The denial message is intentionally generic to avoid engaging with
    potentially harmful content.

    Examples
    --------
    >>> from app.orbitqa.state import OrbitQAState
    >>> state = OrbitQAState(warnings=[])
    >>> result = deny(state)
    >>> result['final_response']['status']
    'denied'
    >>> result['final_response']['reason']
    'The request cannot be processed.'
    >>> len(result['warnings'])
    1

    See Also
    --------
    understand : The module that determines when to invoke deny.
    """
    return {
        "final_response": {
            "status": "denied",
            "reason": "The request cannot be processed.",
            "message": (
                "This request falls outside the supported aerospace and space-systems "
                "domain or violates safety and ethical constraints."
            ),
            "plots": [],
            "warnings": state["warnings"]
            + ["Request denied due to safety, ethical, or domain constraints."],
        },
        "warnings": state["warnings"]
        + ["Request denied due to safety, ethical, or domain constraints."],
    }
