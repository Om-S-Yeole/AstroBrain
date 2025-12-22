from typing import List

from langchain.chat_models import BaseChatModel
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from app.missionops.state import MissionOpsState


class RetrieveResFormat(BaseModel):
    """.
    Structured model for retrieval query results.

    Defines the output format for search queries generated to retrieve
    relevant aerospace knowledge from a vector database.

    Attributes
    ----------
    data_query : List[str]
        List of search queries optimized for aerospace documentation,
        generated based on the current user request.
    """

    data_query: List[str] = Field(
        ...,
        description="List of queries to be asked to vector database based on the current user request",
    )


def retrieve(state: MissionOpsState, config: RunnableConfig):
    """.
    Generate search queries for retrieving aerospace knowledge.

    This node uses an LLM to generate effective search queries optimized
    for retrieving relevant information from a vector database containing
    aerospace textbooks, research papers, and official space-agency
    documentation. Queries are focused and concept-specific.

    Parameters
    ----------
    state : MissionOpsState
        The current state of the MissionOps workflow, containing:
        - user_query : str
            Original user request
        - understood_request : list
            Extracted tasks from the user query
        - user_passed_params : dict
            Extracted parameters from user input
        - user_clarifications : list of str
            Any clarifications provided by the user
        - data_query : list of str
            Previously generated retrieval queries
    config : RunnableConfig
        LangGraph configuration containing:
        - configurable["retrieve_model"] : BaseChatModel
            The language model used for query generation

    Returns
    -------
    dict
        A dictionary containing:
        - data_query : list of str
            Updated list combining previous and newly generated queries

    Notes
    -----
    Generated queries are short, precise, and optimized for technical
    aerospace documentation. Each query focuses on one concept only.
    If no external knowledge retrieval is necessary, an empty list is
    appended to the existing queries.
    """
    model: BaseChatModel = config["configurable"]["retrieve_model"]
    structured_model = model.with_structured_output(RetrieveResFormat)

    # Write system prompt for me
    SYSTEM_PROMPT = SystemMessage(
        """You are a retrieval-query generation module inside an aerospace mission planning system.

Your role is NOT to answer the user.
Your role is NOT to explain concepts.

Your only job is to generate effective search queries
to retrieve relevant aerospace knowledge from a vector database.

----------------
WHAT YOU MUST DO
----------------

1. Generate a list of short, precise retrieval queries.
2. Queries must be optimized for technical aerospace textbooks, research papers, and official space-agency documentation.
3. Each query should focus on ONE concept only.
4. Use terminology as it would appear in textbooks.

----------------
WHAT YOU MUST CONSIDER
----------------

- The original user query
- The current understanding of tasks
- Any parameters already extracted
- Any clarifications provided by the user

----------------
STRICT RULES
----------------

- Do NOT perform calculations.
- Do NOT answer the question.
- Do NOT repeat the user query verbatim.
- Do NOT generate more than necessary queries.
- Do NOT include explanations or commentary.
- Do NOT invent facts.

If no external knowledge retrieval is necessary, return an EMPTY list.

Output MUST strictly follow the required schema.
"""
    )

    # Write human message for me
    USER_MESSAGE = HumanMessage(
        f"""User query:
{state["user_query"]}

Extracted tasks:
{state["understood_request"]}

Extracted parameters:
{state["user_passed_params"]}

User clarifications:
{state["user_clarifications"]}

Previously generated retrieval queries:
{state["data_query"]}
"""
    )

    model_message = [SYSTEM_PROMPT, USER_MESSAGE]

    response = structured_model.invoke(model_message).model_dump()

    return {"data_query": state["data_query"] + response["data_query"]}
