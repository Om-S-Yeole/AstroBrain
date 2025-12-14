from typing import Dict, List

from langchain.chat_models import BaseChatModel
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from app.orbitqa.state import OrbitQAState, ToolInput


class ToolDataPydantic(BaseModel):
    """
    Pydantic model representing a single tool invocation in the execution plan.

    This model defines the schema for a planned tool call, including the tool's
    unique identifier, name, and input parameter specifications. It ensures that
    the LLM generates valid, structured tool invocation plans with explicit
    data dependencies.

    Attributes
    ----------
    tool_id : str
        A unique identifier for this tool invocation, generated using `uuid_generator`.
        Must be unique across all tool calls in the workflow to enable dependency
        tracking and result referencing.
    tool_name : str
        The exact name of the tool as registered in the tool registry. Must match
        a key in the available tools dictionary.
    inputs : Dict[str, ToolInput]
        A dictionary mapping parameter names to their input specifications. Each
        key is a parameter name expected by the tool, and each value is a ToolInput
        object specifying how to obtain the parameter value:
        - LiteralInput: Direct value (e.g., numerical constant, string)
        - DictRefInput: Reference to a dictionary output from a previous tool
        - IndexRefInput: Reference to a list/tuple element from a previous tool

    Examples
    --------
    >>> from app.orbitqa.state import ToolInput
    >>> tool_call = ToolDataPydantic(
    ...     tool_id="550e8400-e29b-41d4-a716-446655440000",
    ...     tool_name="compute_hohmann_transfer",
    ...     inputs={
    ...         "r1": {"type": "literal", "value": 7000},
    ...         "r2": {"type": "literal", "value": 42164},
    ...         "body": {"type": "literal", "value": "Earth"}
    ...     }
    ... )

    With dependencies:
    >>> tool_call_2 = ToolDataPydantic(
    ...     tool_id="660f9511-f3ac-52e5-b827-557766551111",
    ...     tool_name="plot_transfer_orbit",
    ...     inputs={
    ...         "orbit_data": {
    ...             "type": "dict_ref",
    ...             "tool_id": "550e8400-e29b-41d4-a716-446655440000",
    ...             "key": "transfer_orbit"
    ...         }
    ...     }
    ... )

    Notes
    -----
    This model is used with LangChain's `with_structured_output` to constrain
    the LLM's tool planning output to a valid, parseable format.

    The tool_id must be generated using `uuid_generator` to ensure uniqueness
    across the entire workflow execution.
    """

    tool_id: str = Field(..., description="A unique tool id")
    tool_name: str = Field(..., description="Name of the tool")

    # str = param name & ToolInput = how to get value for that param
    inputs: Dict[str, ToolInput] = Field(
        ...,
        description="From where the value of the inputs of the tool must be found out. Key of the dictionary is the input name (as it is) and that specific key's corresponding value is from where the input value can get",
    )


class ToolSeqList(BaseModel):
    """
    Pydantic model for structured output containing a sequence of tool invocations.

    This model defines the top-level schema for the LLM's tool planning response.
    It ensures that the model outputs a valid list of tool invocations in the
    correct execution order, respecting data dependencies.

    Attributes
    ----------
    tool_seq_list : List[ToolDataPydantic]
        An ordered list of tool invocation specifications. Tools are executed
        sequentially in the order they appear in this list. Each element is a
        ToolDataPydantic object containing:
        - A unique tool_id (generated via uuid_generator)
        - The tool_name from the registry
        - Input specifications with dependencies resolved

    Notes
    -----
    The sequence order is critical:
    - Tools must be ordered such that all dependencies are satisfied
    - A tool referencing another tool's output must appear AFTER that tool
    - The execution engine will invoke tools in the exact order specified

    An empty list indicates that the request cannot be fulfilled with available tools.

    Examples
    --------
    >>> tool_sequence = ToolSeqList(tool_seq_list=[
    ...     ToolDataPydantic(
    ...         tool_id="uuid-1",
    ...         tool_name="compute_hohmann_transfer",
    ...         inputs={"r1": {"type": "literal", "value": 7000},
    ...                 "r2": {"type": "literal", "value": 42164}}
    ...     ),
    ...     ToolDataPydantic(
    ...         tool_id="uuid-2",
    ...         tool_name="plot_transfer",
    ...         inputs={"transfer_data": {"type": "dict_ref",
    ...                                     "tool_id": "uuid-1",
    ...                                     "key": "transfer_orbit"}}
    ...     )
    ... ])

    Empty sequence when request cannot be fulfilled:
    >>> impossible_request = ToolSeqList(tool_seq_list=[])

    See Also
    --------
    ToolDataPydantic : Schema for individual tool invocations.
    """

    tool_seq_list: List[ToolDataPydantic] = Field(
        ...,
        description="List of tool schema. Tool schema is defined as ToolDataPydantic",
    )


def tool_selector(state: OrbitQAState, config: RunnableConfig):
    """
    Generate an ordered sequence of tool invocations to fulfill the user's request.

    This function uses an LLM to analyze the user's request, understood tasks,
    extracted parameters, and retrieved reference material to plan a deterministic
    sequence of tool calls. Each tool invocation is assigned a unique ID and
    explicit input specifications, enabling the execution engine to run tools
    in the correct order with proper dependency resolution.

    The function does NOT execute tools. It only plans the execution sequence.

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
        - retrieved_docs : list of RetrievedDoc
            Relevant reference material from the RAG system.
        - tool_sequence : list of ToolDataPydantic
            Previously planned tool invocations (if any).

    config : RunnableConfig
        LangChain configuration object containing:
        - configurable.tool_registry : dict
            Dictionary mapping tool names (str) to tool functions or descriptions.
            Format: {'tool_name': tool_func, ...}
        - configurable.tool_selector_model : BaseChatModel
            The chat model to use for tool planning (e.g., GPT-4, Claude).
            Must have tools bound to it, including `uuid_generator`.

    Returns
    -------
    dict
        A dictionary with a single key:
        - tool_sequence : list of ToolDataPydantic
            Updated list containing both previous and newly planned tool invocations.

    Notes
    -----
    The system prompt instructs the LLM to:
    - Use ONLY tools from the registry (no invented tools)
    - Generate unique tool_id for each invocation using `uuid_generator`
    - Specify inputs explicitly using ToolInput schema variants:
        * LiteralInput: Direct values (constants, strings)
        * DictRefInput: Dictionary outputs from previous tools
        * IndexRefInput: List/tuple elements from previous tools
    - Order tools to respect data dependencies
    - Minimize the number of tool calls
    - Include plotting tools only if visualization is explicitly requested
    - Return an empty sequence if the request cannot be fulfilled

    Tool ID Generation:
    The LLM MUST call `uuid_generator` for each tool invocation to obtain a
    unique identifier. Manual or reused IDs are strictly prohibited.

    Dependency Rules:
    - Tools must be ordered such that all dependencies appear earlier
    - Cross-references use tool_id to link outputs to inputs
    - Circular dependencies are not allowed

    Common tool sequences:
    - Orbital calculations: keplerian_to_cartesian → propagate → plot
    - Transfer analysis: hohmann_transfer → delta_v_calculation → plot_transfer
    - Multi-body: compute_orbit_A → compute_orbit_B → compare_orbits

    Examples
    --------
    >>> from app.orbitqa.state import OrbitQAState
    >>> from langchain_core.runnables import RunnableConfig
    >>> state = OrbitQAState(
    ...     user_query="Calculate and plot a Hohmann transfer from LEO to GEO",
    ...     understood_request=[{"task": "compute_and_visualize_hohmann_transfer"}],
    ...     user_passed_params={"r1": 7000, "r2": 42164},
    ...     retrieved_docs=[],
    ...     tool_sequence=[]
    ... )
    >>> config = RunnableConfig(configurable={
    ...     "tool_registry": {
    ...         "compute_hohmann_transfer": "Calculate Hohmann transfer parameters",
    ...         "plot_transfer_orbit": "Visualize transfer orbit",
    ...         "uuid_generator": "Generate unique UUID"
    ...     },
    ...     "tool_selector_model": chat_model
    ... })
    >>> result = tool_selector(state, config)
    >>> len(result["tool_sequence"])
    2
    >>> result["tool_sequence"][0]["tool_name"]
    'compute_hohmann_transfer'
    >>> result["tool_sequence"][1]["tool_name"]
    'plot_transfer_orbit'

    With existing tools:
    >>> state["tool_sequence"] = [previous_tool]
    >>> result = tool_selector(state, config)
    >>> len(result["tool_sequence"])
    3  # previous + 2 new

    See Also
    --------
    ToolDataPydantic : Schema for individual tool invocations.
    ToolSeqList : Schema for the complete tool sequence.
    uuid_generator : Function for generating unique tool IDs.
    """
    tool_registry: dict = config["configurable"][
        "tool_registry"
    ]  # dict of {'tool_name': tool_func}
    model: BaseChatModel = config["configurable"][
        "tool_selector_model"
    ]  # NOTE: This model must have tools binded to it
    structured_model = model.with_structured_output(ToolSeqList)

    # You have to write system prompt for me
    SYSTEM_PROMPT = SystemMessage(
        """You are a tool-planning module inside an aerospace mission-analysis system.

Your role is NOT to execute tools.
Your role is NOT to perform calculations.
Your role is to PLAN a deterministic sequence of tool invocations.

You operate inside a workflow engine where every tool invocation
must be uniquely identifiable.

A function named `uuid_generator` is available to you.
You MUST use this function to generate tool IDs.

--------------------
TOOL ID RULES (STRICT)
--------------------

- Every tool invocation MUST have a unique `tool_id`.
- You MUST obtain each `tool_id` by CALLING `uuid_generator`.
- You MUST NOT invent, guess, reuse, or manually write tool IDs.
- Each call to `uuid_generator` produces exactly one tool_id.
- Never reuse a tool_id from previous steps.

--------------------
WHAT YOU MUST DO
--------------------

1. Select ONLY tools that exist in the provided tool registry.
2. Produce an ordered sequence of tool calls such that calling those tools in that particular order will do the necessary computations to satisfy the user request.
3. For each tool call:
   - Generate a fresh `tool_id` using `uuid_generator`.
   - Specify `tool_name` exactly as in the registry.
   - Specify all inputs explicitly using the ToolInput schema.
4. Respect data dependencies:
   - If an input comes from a previous tool output, reference it explicitly using that tool_id.
5. Use the MINIMUM number of tools required.
6. Respect the input type of each tool:
   - Ensure each tool receives inputs of the correct type.
   - Use type-conversion tools ONLY if strictly necessary.
   - Do NOT perform unnecessary or redundant type conversions.
   - You have provided all type conversion tools in tool registry.
7. Include plotting tools ONLY if visualization is explicitly required.

--------------------
WHAT YOU MUST NOT DO
--------------------

- Do NOT invent tools.
- Do NOT invent parameter values.
- Do NOT perform calculations.
- Do NOT explain reasoning.
- Do NOT include text outside the schema.
- Do NOT repeat tool calls already planned earlier UNNECESSARILY.

--------------------
DEPENDENCY RULES
--------------------

- LiteralInput: use only if the value is directly known.
- DictRefInput: use when referencing dictionary outputs.
- IndexRefInput: use when referencing list or tuple outputs.
- Every dependency MUST reference a valid earlier tool_id.

--------------------
FAILURE RULE
--------------------

If the request cannot be fulfilled with available tools,
return an EMPTY tool sequence.

Output MUST strictly conform to the required schema."""
    )

    # You have to write user message for me
    USER_MESSAGE = HumanMessage(
        f"""User query:
{state["user_query"]}

Understood tasks:
{state["understood_request"]}

Extracted parameters:
{state["user_passed_params"]}

Retrieved reference material:
{state["retrieved_docs"]}

Available tools (tool_name → description):
{tool_registry}

Previously planned tools:
{state["tool_sequence"]}
"""
    )

    model_message = [SYSTEM_PROMPT, USER_MESSAGE]

    response = structured_model.invoke(model_message).model_dump()

    return {"tool_sequence": state["tool_sequence"] + response["tool_seq_list"]}
