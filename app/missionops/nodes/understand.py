from typing import Any, Literal

from langchain.chat_models import BaseChatModel
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from app.missionops.state import MissionOpsState, TaskDescription


class UnderstandResFormat(BaseModel):
    """.
    Structured model for understanding module results.

    Defines the output format for parsed mission understanding,
    including extracted tasks, parameters, and control actions.

    Attributes
    ----------
    understood_request : List[TaskDescription]
        New specific tasks extracted from the user's query.
    user_passed_params : Dict[str, Any]
        New parameters with explicit values from user query.
        Defaults to empty dict.
    request_action : Literal["toDeny", "toClarify", "toProceed"]
        Control action to be taken: deny (unsafe/unrelated),
        clarify (ambiguous), or proceed (clear).
    to_ask : str
        Clarification question to ask user when request_action
        is "toClarify". Defaults to empty string.
    """

    understood_request: list[TaskDescription] = Field(
        ..., description="Specific tasks that can be extracted from user's query"
    )
    user_passed_params: dict[str, Any] = Field(
        default_factory=dict,
        description="If any specific parameter with values are given in user query, then those will be inserted here as dict",
    )
    request_action: Literal["toDeny", "toClarify", "toProceed"] = Field(
        ..., description="Action to be taken on the user request"
    )
    to_ask: str = Field(
        default="",
        description="If request_action is 'toClarify', then what exact clarification LLM wants to ask the user",
    )


def understand(state: MissionOpsState, config: RunnableConfig):
    """.
    Understand the user's request and determine the next control action.

    This node uses an LLM to interpret the user's aerospace mission request,
    incrementally extracting new tasks and parameters while building on
    previous understanding. It decides whether to proceed, request clarification,
    or deny the request based on safety and clarity criteria.

    Parameters
    ----------
    state : MissionOpsState
        The current state of the MissionOps workflow, containing:
        - user_query : str
            Original user request
        - understood_request : list
            Previously extracted tasks (not to be repeated)
        - user_passed_params : dict
            Previously extracted parameters (not to be repeated)
        - user_clarifications : list of str
            Any clarifications provided by the user so far
    config : RunnableConfig
        LangGraph configuration containing:
        - configurable["understand_model"] : BaseChatModel
            The language model used for understanding

    Returns
    -------
    dict
        A dictionary containing:
        - understood_request : list
            Updated list combining previous and newly extracted tasks
        - user_passed_params : dict
            Updated dictionary merging previous and new parameters
            (new parameters override conflicting previous ones)
        - request_action : str
            One of "toDeny", "toClarify", or "toProceed"
        - to_ask : str
            Clarification question when request_action is "toClarify",
            empty string otherwise

    Notes
    -----
    The function operates incrementally, only extracting new information
    to avoid duplication. It automatically denies requests that are unsafe,
    unethical, illegal, or unrelated to aerospace/engineering domains.
    When ambiguous, it requests clarification with a single precise question.
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
    - A task is a concrete, actionable aerospace operation.
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
    - You are REQUIRED to ask the question if you set request_action as "toClarify".

    5. You are also provided with list of user clarifications made so far by the user. Make use of those user clarifications.

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
