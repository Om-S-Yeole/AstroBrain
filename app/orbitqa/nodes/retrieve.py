from typing import List

from langchain.chat_models import BaseChatModel
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from app.orbitqa.state import OrbitQAState


class RetrieveResFormat(BaseModel):
    """
    Pydantic model for structured output from the retrieval-query generation module.

    This model defines the schema for the LLM's response when generating vector
    database search queries. It ensures that the model outputs a list of retrieval
    queries in a structured, parseable format.

    Attributes
    ----------
    data_query : List[str]
        List of search queries optimized for retrieving relevant aerospace knowledge
        from the vector database. Each query should:
        - Focus on a single concept or topic
        - Use technical terminology as it appears in textbooks
        - Be short and precise (typically 3-10 words)
        - Target specific aerospace concepts, formulas, or procedures

    Examples
    --------
    >>> result = RetrieveResFormat(data_query=[
    ...     "Hohmann transfer delta-v formula",
    ...     "orbital inclination change equations",
    ...     "Lambert problem solution methods"
    ... ])
    >>> result.data_query
    ['Hohmann transfer delta-v formula', 'orbital inclination change equations', 'Lambert problem solution methods']

    Notes
    -----
    This model is used with LangChain's `with_structured_output` method to
    constrain the LLM to produce valid JSON matching this schema.
    """

    data_query: List[str] = Field(
        ...,
        description="List of queries to be asked to vector database based on the current user request",
    )


def retrieve(state: OrbitQAState, config: RunnableConfig):
    """
    Generate vector database search queries to retrieve relevant aerospace knowledge.

    This function uses an LLM to analyze the user's request and generate optimized
    search queries for retrieving relevant information from the RAG system's vector
    database. The queries are designed to match technical aerospace terminology
    as it appears in textbooks, research papers, and official documentation.

    The function does NOT answer questions or retrieve documents directly. It only
    generates the search queries that will be used by downstream retrieval components.

    Parameters
    ----------
    state : OrbitQAState
        The current workflow state containing:
        - user_query : str
            The original user question or request.
        - understood_request : list
            Extracted tasks from the understand module.
        - user_passed_params : dict
            Parameters extracted from the user query.
        - user_clarifications : list of str
            Any clarifications provided by the user.
        - data_query : list of str
            Previously generated retrieval queries (if any).

    config : RunnableConfig
        LangChain configuration object containing:
        - configurable.retrieve_model : BaseChatModel
            The chat model to use for query generation (e.g., GPT-4, Claude).

    Returns
    -------
    dict
        A dictionary with a single key:
        - data_query : list of str
            Updated list containing both previous and newly generated retrieval queries.

    Notes
    -----
    The system prompt instructs the LLM to:
    - Generate short, precise queries (typically 3-10 words)
    - Focus each query on a single aerospace concept
    - Use technical terminology from authoritative sources
    - Avoid answering questions or performing calculations
    - Return an empty list if no external knowledge is needed

    The generated queries are appended to existing queries in the state, allowing
    multiple retrieval iterations if needed.

    Common query patterns include:
    - "[Concept] definition" (e.g., "specific orbital energy definition")
    - "[Process] equation" (e.g., "Hohmann transfer delta-v equation")
    - "[Effect] calculation" (e.g., "J2 perturbation RAAN precession")
    - "[Algorithm] method" (e.g., "Lambert problem universal variables")

    Examples
    --------
    >>> from app.orbitqa.state import OrbitQAState
    >>> from langchain_core.runnables import RunnableConfig
    >>> state = OrbitQAState(
    ...     user_query="Calculate the delta-v for a Hohmann transfer",
    ...     understood_request=[{"task": "compute_hohmann_transfer"}],
    ...     user_passed_params={"r1": 7000, "r2": 42164},
    ...     user_clarifications=[],
    ...     data_query=[]
    ... )
    >>> config = RunnableConfig(configurable={"retrieve_model": chat_model})
    >>> result = retrieve(state, config)
    >>> result["data_query"]
    ['Hohmann transfer delta-v formula', 'two-impulse orbital transfer equations']

    With existing queries:
    >>> state["data_query"] = ["orbital mechanics basics"]
    >>> result = retrieve(state, config)
    >>> result["data_query"]
    ['orbital mechanics basics', 'Hohmann transfer delta-v formula', 'two-impulse orbital transfer equations']

    See Also
    --------
    RetrieveResFormat : Pydantic model defining the output schema.
    understand : Module that extracts tasks before retrieval.
    app.rag.retriever.Retriever : Component that executes the generated queries.
    """
    model: BaseChatModel = config["configurable"]["retrieve_model"]
    structured_model = model.with_structured_output(RetrieveResFormat)

    # Write system prompt for me
    SYSTEM_PROMPT = SystemMessage(
        """You are a retrieval-query generation module inside an aerospace question-answering system.

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
4. Use terminology as it would appear in textbooks (e.g., "Hohmann transfer delta-v", "RAAN precession J2 effect").

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
