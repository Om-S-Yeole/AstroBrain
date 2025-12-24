from langgraph.types import Command, interrupt

from app.missionops.state import MissionOpsState


def clarify(state: MissionOpsState):
    """
    Interrupt workflow to request clarification from the user.

    This node pauses the workflow execution and prompts the user for additional
    information when their request is unclear or requires more details. The user's
    response is added to the clarifications list and the workflow proceeds to the
    understand node.

    Parameters
    ----------
    state : MissionOpsState
        The current state of the MissionOps workflow, containing:
        - to_ask : str, optional
            Specific clarification question to ask. If not provided, uses a default message.
        - user_clarifications : list of str
            Existing list of user clarifications to append to.

    Returns
    -------
    Command
        A LangGraph Command object that:
        - Updates user_clarifications with the new input
        - Redirects workflow to the "understand" node
    """
    user_input = interrupt(
        state["to_ask"]
        if state["to_ask"]
        else "Your request is unclear. Please provide more clarification"
    )

    return Command(
        update={"user_clarifications": state["user_clarifications"] + [str(user_input).strip()]},
        goto="understand",
    )
