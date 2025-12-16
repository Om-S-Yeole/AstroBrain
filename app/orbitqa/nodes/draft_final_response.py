from typing import Literal

from langchain.chat_models import BaseChatModel
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from app.orbitqa.state import OrbitQAState


class LLMFinalResponse(BaseModel):
    """
    Pydantic model for structured final response output from the response formatter.

    This model defines the schema for the LLM's final response generation,
    ensuring that responses have consistent structure with status, reason, and
    message fields. It enables the system to provide clear, standardized feedback
    to users across different execution scenarios (success, denial, error).

    Attributes
    ----------
    status : Literal["success", "denied", "error"]
        The overall status of the workflow execution:
        - "success": Request was processed successfully with valid results
        - "denied": Request was rejected due to safety/ethical/scope concerns
        - "error": Request failed due to technical or execution errors
    reason : str
        Explanation for the status:
        - For "success" status: Must be "SUCCESS"
        - For "denied" status: Brief explanation of why request was denied
        - For "error" status: High-level description of what failed
    message : str
        The complete, formatted, user-facing response message. This is the
        primary output that will be displayed to the user. Should be:
        - Clear and technically accurate
        - Written in professional aerospace engineering tone
        - Based strictly on provided system state
        - Free of internal system details or stack traces

    Examples
    --------
    Success response:
    >>> response = LLMFinalResponse(
    ...     status="success",
    ...     reason="SUCCESS",
    ...     message="The Hohmann transfer from 7000 km to 42164 km requires a total delta-v of 3.9 km/s. The transfer time is approximately 5.28 hours."
    ... )

    Denied response:
    >>> response = LLMFinalResponse(
    ...     status="denied",
    ...     reason="Request involves unethical content",
    ...     message="I cannot assist with this request as it involves content that falls outside the scope of aerospace mission analysis."
    ... )

    Error response:
    >>> response = LLMFinalResponse(
    ...     status="error",
    ...     reason="Invalid orbital parameter: radius must be positive",
    ...     message="I encountered an error while processing your request. The orbital radius provided was invalid. Please ensure all parameters are physically meaningful."
    ... )

    Notes
    -----
    This model is used with LangChain's `with_structured_output` to constrain
    the LLM to produce valid, parseable responses.

    The status field is determined by the workflow execution, not by the LLM.
    The LLM's role is only to format the message appropriately for each status.

    See Also
    --------
    draft_final_response : Function that generates instances of this model.
    """

    status: Literal["success", "denied", "error"] = Field(
        ..., description="The status of the final response"
    )
    reason: str = Field(
        ...,
        description="If response status is 'denied' or 'error', then this field denotes the reason for that. Otherwise 'SUCCESS'",
    )
    message: str = Field(
        ..., description="Final formatted message (response) given to the user"
    )


def draft_final_response(state: OrbitQAState, config: RunnableConfig):
    """
    Generate the final formatted response for the user based on workflow execution state.

    This function is the final step in the OrbitQA workflow. It uses an LLM to
    synthesize all workflow results (understood request, tool outputs, plots,
    warnings, etc.) into a clear, professional, user-facing message. The LLM
    acts only as a formatter, not as a reasoner or calculator.

    The function provides the LLM with complete workflow state and instructs it
    to produce a response that:
    - Accurately reflects the computed results
    - Acknowledges any warnings or errors
    - References generated plots if present
    - Maintains professional aerospace engineering tone
    - Contains no internal system details

    Parameters
    ----------
    state : OrbitQAState
        The complete workflow state containing:
        - user_query : str
            The original user question or request.
        - understood_request : list
            Extracted tasks from the understand module.
        - user_clarifications : list of str
            Any clarifications provided by the user.
        - tool_sequence : list of ToolDataPydantic
            The sequence of tools that were executed.
        - tool_outputs : Dict[str, ToolOutput]
            Results from all executed tools.
        - plots : list of PlotOutput
            Generated visualizations in JSON format.
        - retrieved_docs : list of RetrievedDoc
            Retrieved reference material from RAG system.
        - warnings : list of str
            Any warnings or errors encountered during execution.

    config : RunnableConfig
        LangChain configuration object containing:
        - configurable.final_response_model : BaseChatModel
            The chat model to use for response generation (e.g., GPT-4, Claude).

    Returns
    -------
    dict
        A dictionary containing the response fields plus retained state:
        - status : str
            One of "success", "denied", or "error".
        - reason : str
            Explanation for the status ("SUCCESS" for successful requests).
        - message : str
            The complete, formatted user-facing response.
        - plots : list of PlotOutput
            Plots from the state (passed through unchanged).
        - warnings : list of str
            Warnings from the state (passed through unchanged).

    Notes
    -----
    The system prompt instructs the LLM to:
    - Format responses based strictly on provided data
    - NOT perform new calculations or reasoning
    - NOT invent numbers or parameters
    - NOT contradict tool outputs
    - Maintain consistent status and reason fields
    - Use professional aerospace engineering terminology

    Status Determination:
    The status field is determined by the workflow execution, not by the LLM:
    - "success": All tools executed successfully, results available
    - "denied": Request was rejected by the deny module
    - "error": Tool execution failed or dependencies missing

    The LLM's role is purely formatting - it takes the determined status and
    crafts an appropriate message.

    Response Characteristics:
    - Success messages: Explain results clearly with numbers and units
    - Denied messages: Polite rejection without workarounds
    - Error messages: High-level explanation without stack traces

    Examples
    --------
    Success case:
    >>> from app.orbitqa.state import OrbitQAState
    >>> from langchain_core.runnables import RunnableConfig
    >>> state = OrbitQAState(
    ...     user_query="Calculate Hohmann transfer delta-v",
    ...     understood_request=[{"task": "compute_hohmann_transfer"}],
    ...     user_clarifications=[],
    ...     tool_sequence=[{"tool_id": "uuid-1", "tool_name": "hohmann_transfer", ...}],
    ...     tool_outputs={"uuid-1": {"success": True, "output": {"delta_v": 3.9}}},
    ...     plots=[],
    ...     retrieved_docs=[],
    ...     warnings=[]
    ... )
    >>> config = RunnableConfig(configurable={"final_response_model": chat_model})
    >>> result = draft_final_response(state, config)
    >>> result["status"]
    'success'
    >>> "3.9" in result["message"]
    True

    Error case:
    >>> state.warnings = ["Tool dependency not found. Tool id: uuid-2"]
    >>> state.tool_outputs["uuid-1"]["success"] = False
    >>> result = draft_final_response(state, config)
    >>> result["status"]
    'error'
    >>> "failed" in result["message"].lower()
    True

    With plots:
    >>> state.plots = [{"tool_id": "uuid-3", "plot_id": "plot-1", "plot": "..."}]
    >>> result = draft_final_response(state, config)
    >>> result["plots"] == state.plots
    True

    See Also
    --------
    LLMFinalResponse : Pydantic model defining the response schema.
    compute_and_plot : Module that generates tool_outputs and plots.
    deny : Module that generates denied responses.
    """
    model: BaseChatModel = config["configurable"]["final_response_model"]
    structured_model = model.with_structured_output(LLMFinalResponse)

    SYSTEM_PROMPT = SystemMessage(
        """You are the FINAL RESPONSE FORMATTER inside an aerospace mission-analysis system.

Your role is NOT to perform reasoning.
Your role is NOT to perform calculations.
Your role is NOT to call tools.

Your ONLY responsibility is to produce a clear, accurate, and user-facing
final response based strictly on the provided system state.

---------------------
WHAT YOU MUST DO
---------------------

1. Generate a clear and technically correct final message for the user.
2. Use ONLY the information provided in the input.
3. Respect the system-determined response status and reason.
4. Explain results in a professional aerospace-engineering tone.
5. If plots are present, briefly reference them in the message.
6. If warnings are present, acknowledge them clearly and transparently.

---------------------
WHAT YOU MUST NOT DO
---------------------

- Do NOT perform any new calculations.
- Do NOT invent numbers, parameters, or results.
- Do NOT contradict tool outputs.
- Do NOT change the response status.
- Do NOT change the response reason.
- Do NOT include policy explanations.
- Do NOT include internal system details.

---------------------
STATUS HANDLING RULES
---------------------

- If status == "success":
  - reason MUST be "SUCCESS".
  - Explain the results clearly and confidently.

- If status == "denied":
  - Do NOT suggest workarounds.
  - Do NOT ask follow-up questions.
  - Keep the message polite and professional.

- If status == "error":
  - Explain what failed at a high level.
  - Do NOT expose stack traces or internal errors.

---------------------
OUTPUT RULES
---------------------

- Output MUST strictly conform to the required schema.
- Output MUST contain ONLY structured data.
- Do NOT include any extra text outside the schema.
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

Generated plots:
{state["plots"]}

Retrieved reference material:
{state["retrieved_docs"]}

System warnings:
{state["warnings"]}
"""
    )

    model_message = [SYSTEM_PROMPT, USER_MESSAGE]

    response: dict = structured_model.invoke(model_message).model_dump()

    mutations = response

    mutations["plots"] = state["plots"]
    mutations["warnings"] = state["warnings"]

    return {"final_response": mutations}
