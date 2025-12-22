from app.missionops.state import MissionOpsState


def proceed(state: MissionOpsState):
    """
    Allow workflow to proceed without state modifications.

    This node acts as a pass-through in the workflow graph, permitting
    execution to continue to the next node without making any changes
    to the current state.

    Parameters
    ----------
    state : MissionOpsState
        The current state of the MissionOps workflow (unused).

    Returns
    -------
    dict
        An empty dictionary indicating no state updates.
    """
    return {}
