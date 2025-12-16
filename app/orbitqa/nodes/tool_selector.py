from typing import Any, Dict, List, Literal, Optional, Union

from langchain.chat_models import BaseChatModel
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from app.orbitqa.state import OrbitQAState


class ToolInputSafe(BaseModel):
    """Schema for tool input specifications compatible with various LLM backends.

    This simplified schema represents how to obtain input values for tool parameters,
    designed to avoid Union/Any type issues with some LLM backends (e.g., Ollama).
    It provides three ways to specify input values: literal values, dictionary
    references, or indexed references to previous tool outputs.

    Attributes
    ----------
    kind : Literal["literal", "dict_ref", "index_ref"]
        The type of input specification:
        - "literal": Direct value provided inline (e.g., constants, strings)
        - "dict_ref": Reference to a key in a dictionary output from a previous tool
        - "index_ref": Reference to an element in a list/tuple output from a previous tool
    value : Optional[Union[str, int, float, bool, List[Any], Dict[str, Any]]]
        The actual value when kind is "literal". Must be None for "dict_ref" and
        "index_ref" kinds.
    from_tool_id : Optional[str]
        The unique ID of the source tool when kind is "dict_ref" or "index_ref".
        Must be None for "literal" kind. This establishes data dependencies between
        tool invocations.
    key : Optional[str]
        The dictionary key to extract when kind is "dict_ref". Must be None for
        "literal" and "index_ref" kinds.
    index : Optional[int]
        The list/tuple index to extract when kind is "index_ref". Must be None for
        "literal" and "dict_ref" kinds.

    Examples
    --------
    Literal input:
    >>> input_spec = ToolInputSafe(
    ...     kind="literal",
    ...     value=7000.0
    ... )

    Dictionary reference:
    >>> input_spec = ToolInputSafe(
    ...     kind="dict_ref",
    ...     from_tool_id="550e8400-e29b-41d4-a716-446655440000",
    ...     key="semi_major_axis"
    ... )

    Index reference:
    >>> input_spec = ToolInputSafe(
    ...     kind="index_ref",
    ...     from_tool_id="550e8400-e29b-41d4-a716-446655440000",
    ...     index=0
    ... )

    Notes
    -----
    This schema is used within ToolDataPydantic to specify tool inputs with
    explicit data dependencies, enabling the execution engine to resolve
    parameter values from previous tool outputs.

    See Also
    --------
    ToolDataPydantic : Uses ToolInputSafe to define tool invocation inputs.
    """

    kind: Literal["literal", "dict_ref", "index_ref"] = Field(
        ..., description="Type of input"
    )
    value: Optional[Union[str, int, float, bool, List[Any], Dict[str, Any]]] = Field(
        None, description="Value if kind is literal"
    )
    from_tool_id: Optional[str] = Field(
        None, description="Tool ID if kind is dict_ref or index_ref"
    )
    key: Optional[str] = Field(None, description="Key if kind is dict_ref")
    index: Optional[int] = Field(None, description="Index if kind is index_ref")


class ToolDataPydantic(BaseModel):
    """Schema representing a single tool invocation in the execution plan.

    This model defines the complete specification for a planned tool call, including
    the tool's unique identifier, name, and input parameter specifications. It ensures
    that the LLM generates valid, structured tool invocation plans with explicit
    data dependencies that can be resolved by the execution engine.

    Each tool invocation is uniquely identified and contains all information needed
    to execute the tool with the correct parameters, either from literal values or
    from outputs of previously executed tools.

    Attributes
    ----------
    tool_id : str
        A unique identifier for this specific tool invocation, generated using the
        `uuid_generator` tool. Must be globally unique across all tool calls in the
        entire workflow to enable unambiguous dependency tracking and result referencing.
        Format: UUID4 string (e.g., "550e8400-e29b-41d4-a716-446655440000").
    tool_name : str
        The exact name of the tool as registered in the tool registry. Must match
        a key in the available tools dictionary. Case-sensitive. Examples:
        "keplerian_to_cartesian", "hohmann_transfer", "plot_orbit_3d".
    inputs : Dict[str, ToolInputSafe]
        A dictionary mapping parameter names to their input specifications. Each
        key is a parameter name expected by the tool (must match the tool's function
        signature), and each value is a ToolInputSafe object specifying how to obtain
        the parameter value:

        - kind="literal": Direct value (e.g., numerical constant, string, boolean)
        - kind="dict_ref": Reference to a dictionary output from a previous tool
        - kind="index_ref": Reference to a list/tuple element from a previous tool

        All required parameters for the tool must be present in this dictionary.

    Examples
    --------
    Simple tool call with literal inputs:
    >>> tool_call = ToolDataPydantic(
    ...     tool_id="550e8400-e29b-41d4-a716-446655440000",
    ...     tool_name="hohmann_transfer",
    ...     inputs={
    ...         "r_1_vec": {"kind": "literal", "value": [7000, 0, 0]},
    ...         "v_1_vec": {"kind": "literal", "value": [0, 7.5, 0]},
    ...         "r_2": {"kind": "literal", "value": 42164},
    ...         "attractor": {"kind": "literal", "value": "earth"}
    ...     }
    ... )

    Tool call with dependency on previous tool output:
    >>> tool_call_2 = ToolDataPydantic(
    ...     tool_id="660f9511-f3ac-52e5-b827-557766551111",
    ...     tool_name="plot_orbit_3d",
    ...     inputs={
    ...         "r_vec": {
    ...             "kind": "index_ref",
    ...             "from_tool_id": "550e8400-e29b-41d4-a716-446655440000",
    ...             "index": 0
    ...         },
    ...         "v_vec": {
    ...             "kind": "index_ref",
    ...             "from_tool_id": "550e8400-e29b-41d4-a716-446655440000",
    ...             "index": 1
    ...         },
    ...         "attractor": {"kind": "literal", "value": "earth"},
    ...         "color": {"kind": "literal", "value": "red"}
    ...     }
    ... )

    Dictionary reference from previous tool:
    >>> tool_call_3 = ToolDataPydantic(
    ...     tool_id="770e8511-f4bd-63f6-c938-668877662222",
    ...     tool_name="calculate_period",
    ...     inputs={
    ...         "semi_major_axis": {
    ...             "kind": "dict_ref",
    ...             "from_tool_id": "550e8400-e29b-41d4-a716-446655440000",
    ...             "key": "a"
    ...         }
    ...     }
    ... )

    Notes
    -----
    This model is used with LangChain's `with_structured_output` to constrain
    the LLM's tool planning output to a valid, parseable format.

    The tool_id MUST be generated using the `uuid_generator` tool to ensure
    uniqueness across the entire workflow execution. Manual or reused IDs will
    cause dependency resolution failures.

    The execution engine uses this schema to:
    1. Validate that all required parameters are specified
    2. Resolve dependencies by looking up outputs from previous tool_ids
    3. Execute tools in the correct order
    4. Store results keyed by tool_id for future references

    See Also
    --------
    ToolInputSafe : Schema for individual input specifications.
    ToolSeqList : Container for the complete sequence of tool invocations.
    uuid_generator : Tool for generating unique tool IDs.
    """

    tool_id: str = Field(..., description="A unique tool id")
    tool_name: str = Field(..., description="Name of the tool")

    # str = param name & ToolInput = how to get value for that param
    inputs: Dict[str, ToolInputSafe] = Field(
        ...,
        description="From where the value of the inputs of the tool must be found out. Key of the dictionary is the input name (as it is) and that specific key's corresponding value is from where the input value can get",
    )


class ToolSeqList(BaseModel):
    """Container for an ordered sequence of tool invocations in the execution plan.

    This model defines the top-level schema for the LLM's tool planning response.
    It ensures that the model outputs a valid list of tool invocations in the
    correct execution order, with all data dependencies properly resolved.

    The sequence represents a complete execution plan that transforms the user's
    request into a series of concrete tool calls. The execution engine will invoke
    tools sequentially in the exact order specified in this list.

    Attributes
    ----------
    tool_seq_list : List[ToolDataPydantic]
        An ordered list of tool invocation specifications. Tools are executed
        sequentially in the order they appear in this list. Each element is a
        ToolDataPydantic object containing:

        - A unique tool_id (generated via uuid_generator tool)
        - The tool_name from the registry (exact match required)
        - Input specifications with dependencies explicitly resolved

        The list order is critical for correctness:
        - Tools must be ordered such that all dependencies are satisfied
        - A tool referencing another tool's output must appear AFTER that tool
        - No circular dependencies are allowed
        - The execution engine validates dependencies before execution

    Notes
    -----
    The sequence order is critical:
    - Tools are executed in the exact order specified (index 0 first, then 1, etc.)
    - A tool at index i can only reference outputs from tools at indices 0 to i-1
    - Forward references (referencing a tool that hasn't executed yet) will cause errors
    - The execution engine does NOT reorder tools - the LLM must plan correctly

    An empty list (tool_seq_list=[]) indicates that:
    - The request cannot be fulfilled with available tools, OR
    - The request requires capabilities outside the system's scope, OR
    - The request is ambiguous and requires clarification

    The execution engine handles this list by:
    1. Validating all tool names exist in the registry
    2. Checking that all tool_ids are unique
    3. Verifying that all dependencies reference previous tools
    4. Executing tools sequentially, storing outputs by tool_id
    5. Resolving input references before each tool execution

    Examples
    --------
    Simple sequence with literal inputs:
    >>> tool_sequence = ToolSeqList(tool_seq_list=[
    ...     ToolDataPydantic(
    ...         tool_id="550e8400-e29b-41d4-a716-446655440000",
    ...         tool_name="orbit_period",
    ...         inputs={
    ...             "a": {"kind": "literal", "value": 7000},
    ...             "mu": {"kind": "literal", "value": 398600.4418}
    ...         }
    ...     )
    ... ])

    Multi-tool sequence with dependencies:
    >>> tool_sequence = ToolSeqList(tool_seq_list=[
    ...     ToolDataPydantic(
    ...         tool_id="uuid-1",
    ...         tool_name="keplerian_to_cartesian",
    ...         inputs={"a": {"kind": "literal", "value": 7000},
    ...                 "ecc": {"kind": "literal", "value": 0.01},
    ...                 "inc": {"kind": "literal", "value": 45},
    ...                 "raan": {"kind": "literal", "value": 0},
    ...                 "argp": {"kind": "literal", "value": 0},
    ...                 "nu": {"kind": "literal", "value": 0},
    ...                 "attractor": {"kind": "literal", "value": "earth"}}
    ...     ),
    ...     ToolDataPydantic(
    ...         tool_id="uuid-2",
    ...         tool_name="plot_orbit_3d",
    ...         inputs={
    ...             "r_vec": {"kind": "index_ref",
    ...                       "from_tool_id": "uuid-1",
    ...                       "index": 0},
    ...             "v_vec": {"kind": "index_ref",
    ...                       "from_tool_id": "uuid-1",
    ...                       "index": 1},
    ...             "attractor": {"kind": "literal", "value": "earth"},
    ...             "color": {"kind": "literal", "value": "red"}
    ...         }
    ...     )
    ... ])

    Empty sequence when request cannot be fulfilled:
    >>> impossible_request = ToolSeqList(tool_seq_list=[])

    See Also
    --------
    ToolDataPydantic : Schema for individual tool invocations.
    tool_selector : Function that generates instances of this model.
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
EXAMPLE OUTPUT (REFERENCE ONLY)
--------------------

The following is an EXAMPLE of a valid tool sequence.
It is provided ONLY to demonstrate the required structure and formatting.
The UUIDs in this example are only for example purpose and you must generate unique UUIDs using 'uuid_generator' tool provided to you.

{
  "tool_seq_list": [
    {
      "tool_id": "c1a2f3b4-1111-2222-3333-444455556666",
      "tool_name": "keplerian_to_cartesian",
      "inputs": {
        "a": {
          "kind": "literal",
          "value": 7000
        },
        "ecc": {
          "kind": "literal",
          "value": 0.001
        },
        "inc": {
          "kind": "literal",
          "value": 98.7
        },
        "raan": {
          "kind": "literal",
          "value": 0.0
        },
        "argp": {
          "kind": "literal",
          "value": 0.0
        },
        "nu": {
          "kind": "literal",
          "value": 0.0
        },
        "attractor": {
          "kind": "literal",
          "value": "earth"
        }
      }
    },
    {
      "tool_id": "d7e8f9a0-7777-8888-9999-000011112222",
      "tool_name": "plot_orbit_3d",
      "inputs": {
        "r_vec": {
          "kind": "index_ref",
          "from_tool_id": "c1a2f3b4-1111-2222-3333-444455556666",
          "index": 0
        },
        "v_vec": {
          "kind": "index_ref",
          "from_tool_id": "c1a2f3b4-1111-2222-3333-444455556666",
          "index": 1
        },
        "attractor": {
          "kind": "literal",
          "value": "Earth"
        },
        "color": {
          "kind": "literal",
          "value": "red"
        }
      }
    }
  ]
}

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
