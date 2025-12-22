from typing import Literal

from langchain.chat_models import BaseChatModel
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from app.missionops.state import MissionOpsState


class LLMFinalResponse(BaseModel):
    """.
    Structured model for the final response output.

    Attributes
    ----------
    status : Literal["success", "denied", "error"]
        The status of the final response indicating outcome.
    reason : str
        Reason for the status. If 'success', this should be "success".
        If 'denied' or 'error', provides explanation for that outcome.
    message : str
        Final formatted message (response) presented to the user.
    """

    status: Literal["success", "denied", "error"] = Field(
        ..., description="The status of the final response"
    )
    reason: str = Field(
        ...,
        description="If response status is 'denied' or 'error', then this field denotes the reason for that. Otherwise 'success'",
    )
    message: str = Field(..., description="Final formatted message (response) given to the user")


def draft_final_response(state: MissionOpsState, config: RunnableConfig):
    """.
    Generate the final user-facing response for a mission analysis request.

    This node operates as the final step in the MissionOps workflow, producing
    a structured response based on completed analysis results, tool outputs,
    and validation decisions. It uses an LLM to synthesize all workflow
    information into a clear, professional response.

    Parameters
    ----------
    state : MissionOpsState
        The current state of the MissionOps workflow, containing:
        - user_query : str
            Original user request
        - understood_request : str
            Parsed and interpreted tasks
        - user_clarifications : list of str
            Any clarifications provided by the user
        - tool_sequence : list
            Sequence of tools executed
        - tool_outputs : list or dict
            Results from tool executions
        - retrieved_docs : list or dict
            Retrieved reference material
        - warnings : list of str
            Accumulated system warnings
    config : RunnableConfig
        LangGraph configuration containing:
        - configurable["final_response_model"] : BaseChatModel
            The language model used to generate the final response

    Returns
    -------
    dict
        A dictionary containing:
        - final_response : dict
            Structured response with:
            - status : str
                One of "success", "denied", or "error"
            - reason : str
                Reason for the status
            - message : str
                Final formatted message for the user
            - warnings : list of str
                Complete list of warnings from the workflow

    Notes
    -----
    This function does not perform analysis or execute tools. It only
    synthesizes existing results into a final response. The LLM is
    instructed to base the response strictly on the provided state
    without inventing new data or conclusions.
    """
    model: BaseChatModel = config["configurable"]["final_response_model"]
    structured_model = model.with_structured_output(LLMFinalResponse)

    # Write a best system prompt for me
    SYSTEM_PROMPT = SystemMessage(
        """
You are an aerospace MISSION REPORTING module inside a deterministic AI mission-analysis workflow.

Your role is NOT to perform analysis.
Your role is NOT to execute tools.
Your role is NOT to make feasibility decisions.
Your role is to PRODUCE THE FINAL USER-FACING RESPONSE
based STRICTLY on completed analysis results.

You operate AFTER:
- all mission analysis has been completed
- feasibility has been validated
- all tool executions have finished
- warnings and errors have been finalized

------------------------
WHAT YOU MUST DO
------------------------

1. Generate a final response for the user that:
   - Clearly answers the user's original request.
   - Accurately reflects the results of the mission analysis.
   - Is consistent with tool outputs, warnings, and validation decisions.

2. Set the response status:
   - "success":
        If the mission analysis completed successfully.
   - "denied":
        If the mission was rejected due to infeasibility or safety constraints.
   - "error":
        If the analysis failed due to internal errors or missing critical data.

3. Set the reason field:
   - If status is "success", set reason to "success".
   - If status is "denied" or "error", provide a concise reason.

4. Write the message field as a professional, structured explanation:
   - Use clear language suitable for an engineering audience.
   - Summarize key findings and conclusions.
   - Mention important assumptions explicitly if they exist.
   - Include relevant warnings, if any, without exaggeration.
   - Message must be in detail.

------------------------
STRICT RULES
------------------------

- Do NOT invent any new data, numbers, or conclusions.
- Do NOT perform calculations.
- Do NOT contradict the validator decision.
- Do NOT suggest fixes, alternatives, or next steps unless they were already derived.
- Do NOT mention internal tool names, tool IDs, or system internals.
- Do NOT include chain-of-thought or reasoning text.
- Do NOT include raw tool outputs verbatim.

You must base your response ONLY on:
- the executed tool outputs
- retrieved reference material
- accumulated warnings
- the final mission validation outcome

------------------------
IMPORTANT
------------------------

This is the FINAL step of the workflow.
Your output will be sent directly to the user.
Accuracy and honesty are more important than optimism or completeness.
"""
    )

    USER_MESSAGE = HumanMessage(
        f"""User query:
{state["user_query"]}

The understood request (Tasks identified):
{state["understood_request"]}

Clarifications made by user:
{state["user_clarifications"]}

Sequence in which tools are run:
{state["tool_sequence"]}

Computed tool outputs:
{state["tool_outputs"]}

Retrieved reference material:
{state["retrieved_docs"]}

System warnings:
{state["warnings"]}
"""
    )

    model_message = [SYSTEM_PROMPT, USER_MESSAGE]

    response: dict = structured_model.invoke(model_message).model_dump()

    mutations = response

    mutations["warnings"] = state["warnings"]

    return {"final_response": mutations}
