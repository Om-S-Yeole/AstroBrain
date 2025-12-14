from typing import Any, Dict, List, Literal

from langchain.chat_models import BaseChatModel
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from app.orbitqa.state import OrbitQAState, TaskDescription


class UnderstandResFormat(BaseModel):
    """
    Structured output format for the understanding module.

    This Pydantic model defines the schema for the structured output returned by
    the understand function. It encapsulates the extracted tasks, parameters,
    and control flow decisions for processing a user's aerospace-related query.

    Attributes
    ----------
    understood_request : list of TaskDescription
        List of specific, actionable tasks extracted from the user's query.
        Each task represents a concrete aerospace operation to be performed.
    user_passed_params : dict
        Dictionary containing parameter names and their explicit values provided
        by the user (e.g., altitudes, orbital elements, dates).
    request_action : {'toDeny', 'toClarify', 'toProceed'}
        Control action indicating how to handle the user's request:
        - 'toDeny': Request is unsafe, unethical, or unrelated to aerospace.
        - 'toClarify': Request needs additional information from the user.
        - 'toProceed': Request is clear and ready for execution.
    to_ask : str
        The specific clarification question to ask the user. Only populated when
        request_action is 'toClarify', otherwise empty string.

    Examples
    --------
    >>> result = UnderstandResFormat(
    ...     understood_request=[TaskDescription(task="compute_hohmann_transfer")],
    ...     user_passed_params={"initial_altitude": 400, "final_altitude": 35786},
    ...     request_action="toProceed",
    ...     to_ask=""
    ... )
    >>> result.request_action
    'toProceed'
    """

    understood_request: List[TaskDescription] = Field(
        ..., description="Specific tasks that can be extracted from user's query"
    )
    user_passed_params: Dict[str, Any] = Field(
        default={},
        description="If any specific parameter with values are given in user query, then those will be inserted here as dict",
    )
    request_action: Literal["toDeny", "toClarify", "toProceed"] = Field(
        ..., description="Action to be taken on the user request"
    )
    to_ask: str = Field(
        default="",
        description="If request_action is 'toClarify', then what exact clarification LLM wants to ask the user",
    )


def understand(state: OrbitQAState, config: RunnableConfig):
    """
    Understand and analyze the user's aerospace-related query.

    This function serves as the understanding module in the OrbitQA workflow. It uses
    a language model to extract actionable tasks, parameters, and determine the
    appropriate control flow action. It operates incrementally, building on previous
    understanding to avoid redundant extraction.

    The function does NOT execute tasks or answer questions - it only analyzes the
    request structure and decides whether to proceed, clarify, or deny.

    Parameters
    ----------
    state : OrbitQAState
        The current workflow state containing:
        - user_query : str
            The user's current question or request.
        - understood_request : list
            Previously extracted tasks to avoid duplication.
        - user_passed_params : dict
            Previously extracted parameters to build upon.
        - user_clarifications : list
            History of user clarification responses.
    config : RunnableConfig
        Runtime configuration containing:
        - configurable['understand_model'] : BaseChatModel
            The language model to use for understanding.

    Returns
    -------
    dict
        A dictionary containing state mutations:
        - understood_request : list
            Updated list combining previous and newly extracted tasks.
        - user_passed_params : dict
            Updated parameters dictionary with new values merged/overriding old ones.
        - request_action : {'toDeny', 'toClarify', 'toProceed'}
            The determined control action.
        - to_ask : str
            Clarification question if request_action is 'toClarify', empty otherwise.

    Raises
    ------
    KeyError
        If 'understand_model' is not present in config['configurable'].

    Notes
    -----
    The function implements three safety control paths:

    - **toDeny**: Used for unsafe, unethical, illegal, or non-aerospace requests
    - **toClarify**: Used when aerospace-related but missing critical information
    - **toProceed**: Used when request is sufficiently clear to continue

    The model is instructed to be incremental and NOT repeat previously extracted
    information. This is crucial for multi-turn clarification dialogues.

    Examples
    --------
    >>> from app.orbitqa.state import OrbitQAState
    >>> from langchain_core.runnables import RunnableConfig
    >>> from langchain_openai import ChatOpenAI
    >>>
    >>> state = OrbitQAState(
    ...     user_query="Calculate delta-v for Hohmann transfer",
    ...     understood_request=[],
    ...     user_passed_params={},
    ...     user_clarifications=[]
    ... )
    >>> config = RunnableConfig(
    ...     configurable={"understand_model": ChatOpenAI(model="gpt-4")}
    ... )
    >>> mutations = understand(state, config)
    >>> mutations['request_action']
    'toClarify'
    >>> mutations['to_ask']
    'What are the initial and final orbit altitudes?'

    See Also
    --------
    UnderstandResFormat : The structured output schema.
    OrbitQAState : The workflow state definition.
    """
    model: BaseChatModel = config["configurable"]["understand_model"]
    structured_model = model.with_structured_output(UnderstandResFormat)

    # ----- System Prompt ----
    SYSTEM_PROMPT = SystemMessage(
        """You are an aerospace mission-analysis understanding module inside a deterministic AI workflow.

    Your role is NOT to answer the user.
    Your role is to UNDERSTAND the user's request and decide the next control action.

    You operate inside a multi-step planning system with memory.
    Previous understanding and parameters may already exist.
    You MUST build on them incrementally.

    ------------------------
    WHAT YOU MUST DO
    ------------------------

    1. Extract ONLY NEW tasks from the user request.
    - A task is a concrete, actionable aerospace operation (e.g., "compute Hohmann transfer delta-v", "plot orbit", "retrieve textbook explanation").
    - DO NOT repeat tasks already present in previous understanding.

    2. Extract ONLY NEW user-passed parameters with explicit values.
    - Examples: altitudes, orbital elements, time durations, vectors, dates.
    - DO NOT restate parameters already present.
    - If a new parameter overrides an old one, include the new value.

    3. Decide EXACTLY ONE request action:
    - "toDeny":
        If the request is unsafe, unethical, illegal, non-consensual, self-harm related,
        or completely unrelated to aerospace / engineering / space systems.
    - "toClarify":
        If the request is aerospace-related but ambiguous or missing critical information and you want further clarifications from user to successfully understand it.
    - "toProceed":
        If the request is sufficiently clear to continue and you think we can give a resonable response to user using the current information user passed.

    4. If and ONLY if request_action == "toClarify":
    - Ask ONE precise clarification question.
    - The question must directly unblock execution.
    - Do NOT ask multiple questions.

    ------------------------
    STRICT RULES
    ------------------------

    - Do NOT perform calculations.
    - Do NOT explain concepts.
    - Do NOT generate plots.
    - Do NOT suggest tools.
    - Do NOT repeat prior tasks or parameters.
    - Do NOT invent missing values.
    - Do NOT include reasoning text.

    You must output ONLY structured data that conforms exactly to the required schema.

    ------------------------
    SAFETY
    ------------------------

    Immediately choose "toDeny" if the request involves:
    - self-harm or suicide
    - violence or weapons
    - illegal activity
    - non-consensual wrongdoing
    - requests unrelated to aerospace, physics, or engineering

    When in doubt between "toClarify" and "toProceed", choose "toClarify"."""
    )

    # ----- Human Message ----
    USER_MESSAGE = HumanMessage(
        f"""User query:
        {state["user_query"]}

        Previously extracted tasks (do NOT repeat):
        {state["understood_request"]}

        Previously extracted user parameters (do NOT repeat):
        {state["user_passed_params"]}

        User clarifications so far:
        {state["user_clarifications"]}
        """
    )

    # ----- Add messages together ----
    model_messages = [SYSTEM_PROMPT, USER_MESSAGE]

    # ----- Get response from model ----
    response: dict = structured_model.invoke(model_messages).model_dump()

    mutations = {}

    mutations["understood_request"] = (
        state["understood_request"] + response["understood_request"]
    )  # We only want model to extract new tasks (if user makes clarifications, then this is useful to not extract whole context but rather only new things)

    mutations["user_passed_params"] = {
        **state["user_passed_params"],
        **response["user_passed_params"],
    }  # We only want model to extract new user passed params (if user makes clarifications, then this is useful to not extract whole params again, but to only extract new ones). If new params are extracted and their key matches with old params, then we override.

    mutations["request_action"] = response["request_action"]

    mutations["to_ask"] = response["to_ask"]

    return mutations
