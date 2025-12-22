from typing import Literal

from langchain.chat_models import BaseChatModel
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from app.missionops.state import MissionOpsState


class ValidatorResFormat(BaseModel):
    """.
    Structured model for validation results.

    Defines the output format for mission feasibility validation,
    including the action decision, non-feasibility message, and warnings.

    Attributes
    ----------
    request_action : Literal["toDeny", "toProceed"]
        Control action: "toDeny" if mission is not feasible,
        "toProceed" if mission analysis is valid.
    drafted_message_for_non_feasibility : str
        Message to present to user if mission is not feasible.
        Defaults to empty string.
    warnings : List[str]
        List of warnings to notify the user about analysis concerns
        or limitations. Defaults to empty list.
    """

    request_action: Literal["toDeny", "toProceed"] = Field(
        ..., description="Action to be taken on the analysis done till now"
    )
    drafted_message_for_non_feasibility: str = Field(
        default="",
        description="If mission is not feasible, then what message must be given to the user",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Any warnings that must be notified to the user",
    )


def validator(state: MissionOpsState, config: RunnableConfig):
    """.
    Validate mission feasibility based on tool execution outputs.

    This node uses an LLM to review all mission analysis results,
    including tool outputs and retrieved documentation, to determine
    whether the mission is feasible. If not feasible, it generates
    an error response with explanation. If feasible, it collects
    warnings and allows the workflow to proceed.

    Parameters
    ----------
    state : MissionOpsState
        The current state of the MissionOps workflow, containing:
        - user_query : str
            Original user request
        - understood_request : str
            Parsed and interpreted mission tasks
        - user_passed_params : dict
            Extracted parameters from user input
        - retrieved_docs : list or dict
            Retrieved reference material
        - tool_sequence : list
            Complete sequence of planned tools
        - tool_outputs : dict
            Results from all tool executions
        - warnings : list of str
            Accumulated warnings from previous stages
    config : RunnableConfig
        LangGraph configuration containing:
        - configurable["validator_model"] : BaseChatModel
            The language model used for validation

    Returns
    -------
    dict
        If mission is feasible (request_action == "toProceed"):
            - warnings : list of str
                Updated warnings list with validation warnings appended

        If mission is not feasible (request_action == "toDeny"):
            - request_action : str
                Set to "toDeny"
            - warnings : list of str
                Updated warnings list
            - final_response : dict
                Error response containing:
                - status : str
                    Set to "error"
                - reason : str
                    Set to "Mission is not feasible"
                - message : str
                    Detailed non-feasibility explanation
                - warnings : list of str
                    Complete warnings list

    Notes
    -----
    This function serves as a quality gate before final response generation.
    It ensures that mission analysis results meet feasibility criteria and
    generates appropriate error responses if constraints are violated.
    """
    model: BaseChatModel = config["configurable"]["validator_model"]
    structured_model = model.with_structured_output(ValidatorResFormat)

    # --- SYSTEM PROMPT ---
    SYSTEM_PROMPT = SystemMessage(
        """
You are an aerospace MISSION VALIDATION AND SAFETY REVIEW module
inside a deterministic AI mission-analysis workflow.

Your role is NOT to execute tools.
Your role is NOT to perform calculations.
Your role is NOT to plan further analysis.
Your role is to ASSESS FEASIBILITY and DECIDE
whether the mission analysis may PROCEED or MUST BE TERMINATED.

You operate AFTER multiple analysis stages have already been completed.
All physics calculations and simulations are assumed to be done externally.

------------------------
WHAT YOU MUST DO
------------------------

1. Review the entire mission analysis context, including:
   - the original user request
   - the understood mission tasks
   - extracted mission parameters
   - retrieved reference material
   - the planned and executed analysis tools
   - warnings accumulated so far

2. Decide EXACTLY ONE action:
   - "toProceed":
        If the mission is feasible OR conditionally feasible.
   - "toDeny":
        If the mission is NOT feasible due to fundamental constraints,
        safety violations, or critical analysis failures.

3. If request_action == "toDeny":
   - List ALL critical errors that make the mission infeasible.
   - Draft a clear, professional explanation for the user
     describing WHY the mission cannot proceed.
   - The explanation must be factual, concise, and non-judgmental.

4. If request_action == "toProceed":
   - List any WARNINGS that the user should be aware of.
   - Warnings must NOT contradict feasibility.

------------------------
DECISION GUIDELINES
------------------------

You MUST choose "toDeny" if ANY of the following are true:
- Required analysis tools failed to execute successfully.
- Tool dependencies were missing or unresolved.
- Power, thermal, or orbital constraints make the mission impossible.
- The mission violates physical or operational constraints.
- Critical outputs required for decision-making are absent.

You MAY choose "toProceed" if:
- The mission is feasible as specified, OR
- The mission is feasible with acceptable risks and warnings.

------------------------
STRICT RULES
------------------------

- Do NOT invent new data.
- Do NOT perform calculations.
- Do NOT suggest fixes or alternatives.
- Do NOT ask clarifying questions.
- Do NOT modify mission goals.
- Do NOT repeat tool outputs verbatim.
- Do NOT include reasoning or internal analysis text.

You must output ONLY structured data that conforms EXACTLY
to the required schema.

------------------------
IMPORTANT
------------------------

This decision is FINAL for the current execution path.
If you choose "toDeny", the mission analysis MUST terminate.
"""
    )

    # User message
    USER_MESSAGE = HumanMessage(
        f"""User query:
{state["user_query"]}

Understood tasks:
{state["understood_request"]}

Extracted parameters:
{state["user_passed_params"]}

Retrieved reference material:
{state["retrieved_docs"]}

Previously planned tools:
{state["tool_sequence"]}

Tool execution outputs:
{state["tool_outputs"]}
"""
    )

    # ----- Add messages together ----
    model_messages = [SYSTEM_PROMPT, USER_MESSAGE]

    # ----- Get response from model ----
    response: dict = structured_model.invoke(model_messages).model_dump()

    if response["request_action"] == "toProceed":
        return {"warnings": state["warnings"] + response["warnings"]}
    else:
        return {
            "request_action": response["request_action"],
            "warnings": state["warnings"] + response["warnings"],
            "final_response": {
                "status": "error",
                "reason": "Mission is not feasible",
                "message": response["drafted_message_for_non_feasibility"],
                "warnings": state["warnings"] + response["warnings"],
            },
        }
