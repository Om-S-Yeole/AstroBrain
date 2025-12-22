from app.missionops.state import MissionOpsState


def deny(state: MissionOpsState):
    """
    Deny the user's request and terminate the workflow.

    This node generates a denial response when a request falls outside the
    supported aerospace and space-systems domain or violates safety and
    ethical constraints. It returns a structured response with denial
    information and updates the warnings list.

    Parameters
    ----------
    state : MissionOpsState
        The current state of the MissionOps workflow, containing:
        - warnings : list of str
            Existing list of warnings to append to.

    Returns
    -------
    dict
        A dictionary containing:
        - final_response : dict
            Structured response with:
            - status : str
                Set to "denied"
            - reason : str
                Brief reason for denial
            - message : str
                Detailed denial message
            - warnings : list of str
                Complete list of warnings including the denial warning
        - warnings : list of str
            Updated warnings list with denial message appended
    """
    return {
        "final_response": {
            "status": "denied",
            "reason": "The request cannot be processed.",
            "message": (
                "This request falls outside the supported aerospace and space-systems "
                "domain or violates safety and ethical constraints."
            ),
            "warnings": state["warnings"]
            + ["Request denied due to safety, ethical, or domain constraints."],
        },
        "warnings": state["warnings"]
        + ["Request denied due to safety, ethical, or domain constraints."],
    }
