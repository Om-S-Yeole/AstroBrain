from typing import Any, Dict, List

from langchain.chat_models import BaseChatModel
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from app.missionops.state import MissionOpsState, TaskDescription


class ReEvaluateResFormat(BaseModel):
    """.
    Structured model for re-evaluation results.

    Defines the output format for refined mission understanding,
    including updated tasks and parameters after knowledge retrieval.

    Attributes
    ----------
    understood_request : List[TaskDescription]
        New or refined specific tasks extracted from the user's query
        after domain knowledge retrieval.
    user_passed_params : Dict[str, Any]
        New or updated parameters with values from user query or
        refined based on retrieved knowledge. Defaults to empty dict.
    """

    understood_request: List[TaskDescription] = Field(
        ..., description="Specific tasks that can be extracted from user's query"
    )
    user_passed_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="If any specific parameter with values are given in user query, then those will be inserted here as dict",
    )


def re_eval(state: MissionOpsState, config: RunnableConfig):
    """.
    Re-evaluate and refine mission understanding after knowledge retrieval.

    This node operates after initial understanding and knowledge retrieval
    phases. It uses an LLM to review the mission request in light of
    retrieved domain knowledge and extracts new or refined tasks and
    parameters without repeating existing ones.

    Parameters
    ----------
    state : MissionOpsState
        The current state of the MissionOps workflow, containing:
        - user_query : str
            Original user request
        - understood_request : list
            Previously extracted tasks
        - user_passed_params : dict
            Previously extracted parameters
        - user_clarifications : list of str
            Any clarifications provided by the user
        - retrieved_docs : list or dict
            Retrieved domain knowledge from vector database
    config : RunnableConfig
        LangGraph configuration containing:
        - configurable["re_evaluate_model"] : BaseChatModel
            The language model used for re-evaluation

    Returns
    -------
    dict
        A dictionary containing:
        - understood_request : list
            Updated list combining previous and newly extracted tasks
        - user_passed_params : dict
            Updated dictionary merging previous and new parameters
            (new parameters override conflicting previous ones)

    Notes
    -----
    This function refines the mission understanding by incorporating
    retrieved domain knowledge. It only adds new tasks or parameters,
    avoiding duplication unless refinement is needed. Conflicting
    parameter values are resolved by keeping the refined version.
    """
    model: BaseChatModel = config["configurable"]["re_evaluate_model"]
    structured_model = model.with_structured_output(ReEvaluateResFormat)

    # ----- System Prompt ----
    SYSTEM_PROMPT = SystemMessage(
        """
You are an aerospace mission-analysis re-evaluation module inside a deterministic AI workflow.

Your role is NOT to answer the user.
Your role is NOT to plan tools or execution steps.
Your role is to RE-EVALUATE and REFINE the mission understanding
after additional domain knowledge has been retrieved.

You operate AFTER an initial understanding phase and AFTER knowledge retrieval.
Previous tasks and parameters already exist in state.

------------------------
WHAT YOU MUST DO
------------------------

1. Review the mission request using:
   - the original user query
   - previously extracted tasks
   - previously extracted user parameters
   - user clarifications
   - retrieved domain knowledge from vector database

2. Extract ONLY NEW or REFINED tasks if necessary.
   - A task is a concrete, actionable aerospace operation.
   - DO NOT repeat tasks already present unless refinement is required.
   - If a task needs refinement, add a more specific version of it.

3. Extract ONLY NEW or UPDATED user parameters.
   - Parameters must come from:
       a) explicit user statements, OR
       b) logical refinement implied by retrieved knowledge.
   - DO NOT invent numerical values.
   - DO NOT assume defaults.
   - If a parameter logically constrains another (e.g. orbit type), include the refined constraint.

4. If a newly refined parameter conflicts with a previous one:
   - Include ONLY the refined value.
   - Do NOT explain the conflict.

------------------------
STRICT RULES
------------------------

- Do NOT perform calculations.
- Do NOT explain concepts.
- Do NOT generate plots.
- Do NOT suggest tools.
- Do NOT change control flow.
- Do NOT ask clarifying questions.
- Do NOT invent missing values.
- Do NOT include reasoning text.

You must output ONLY structured data that conforms exactly to the required schema.

If no new tasks or parameters are required,
return empty lists/dictionaries accordingly.
"""
    )

    # ----- Human Message ----
    USER_MESSAGE = HumanMessage(
        f"""User query:
        {state["user_query"]}

        Previously extracted tasks:
        {state["understood_request"]}

        Previously extracted user parameters:
        {state["user_passed_params"]}

        User clarifications so far:
        {state["user_clarifications"]}

        Retrieved documents from vector database:
        {state["retrieved_docs"]}
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

    return mutations
