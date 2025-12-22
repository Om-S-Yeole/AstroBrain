from typing import Any, Dict, List, Literal, Optional, Union

from langchain.chat_models import BaseChatModel
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from app.missionops.state import MissionOpsState


class ToolInputSafe(BaseModel):
    """.
    Structured model for safe tool input references.

    Defines how to retrieve input values for tool invocations, supporting
    literal values, dictionary references, and index references.

    Attributes
    ----------
    kind : Literal["literal", "dict_ref", "index_ref"]
        Type of input reference.
    value : Union[str, int, float, bool, List[Any], Dict[str, Any]], optional
        Direct value when kind is "literal".
    from_tool_id : str, optional
        Tool ID to reference when kind is "dict_ref" or "index_ref".
    key : str, optional
        Dictionary key when kind is "dict_ref".
    index : int, optional
        List/tuple index when kind is "index_ref".
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
    """.
    Structured model for tool invocation metadata.

    Defines a complete specification for a single tool invocation,
    including the tool identifier, name, and input mappings.

    Attributes
    ----------
    tool_id : str
        Unique identifier for this tool invocation.
    tool_name : str
        Name of the tool to be invoked.
    inputs : Dict[str, ToolInputSafe]
        Mapping of parameter names to their input sources.
        Key is the parameter name, value is a ToolInputSafe object
        specifying how to retrieve the parameter value.
    """

    tool_id: str = Field(..., description="A unique tool id")
    tool_name: str = Field(..., description="Name of the tool")

    # str = param name & ToolInput = how to get value for that param
    inputs: Dict[str, ToolInputSafe] = Field(
        ...,
        description="From where the value of the inputs of the tool must be found out. Key of the dictionary is the input name (as it is) and that specific key's corresponding value is from where the input value can get",
    )


class ToolSeqList(BaseModel):
    """.
    Structured model for a sequence of tool invocations.

    Wraps a list of tool invocation specifications for orbit
    propagation and visibility analysis.

    Attributes
    ----------
    tool_seq_list : List[ToolDataPydantic]
        Ordered list of tool invocations to be executed.
    """

    tool_seq_list: List[ToolDataPydantic] = Field(
        ...,
        description="List of tool schema. Tool schema is defined as ToolDataPydantic",
    )


def orbit_propagator_visibility(state: MissionOpsState, config: RunnableConfig):
    """.
    Plan tool invocations for orbit propagation and visibility analysis.

    This node uses an LLM to plan the sequence of tools needed for orbit
    propagation and visibility analysis tasks. It determines which tools
    are required based on the user's request and generates a structured
    tool execution plan with proper data dependencies.

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
            Previously planned tool invocations
    config : RunnableConfig
        LangGraph configuration containing:
        - configurable["orbit_propagator_visibility_model"] : BaseChatModel
            The language model used for tool planning (must have tools bound)

    Returns
    -------
    dict
        A dictionary containing:
        - tool_sequence : list
            Updated tool sequence with new orbit/visibility tools appended
        - start_execution_from_idx : int
            Index from which to start executing tools (set to 0)
        - tool_selection_state_done : str
            Set to "orbit_propagator_visibility" to indicate this planning stage is complete

    Notes
    -----
    This function only plans orbit creation, orbit propagation, and ground
    visibility/access window analysis tools. It does not execute any tools.
    Each planned tool receives a unique ID generated via uuid_generator_tool.
    If no orbit or visibility tools are required, an empty sequence is returned.
    """
    model: BaseChatModel = config["configurable"][
        "orbit_propagator_visibility_model"
    ]  # NOTE: This model must have tools binded to it
    structured_model = model.with_structured_output(ToolSeqList)

    # --- SYSTEM PROMPT ----
    SYSTEM_PROMPT = SystemMessage(
        """
You are an aerospace mission-analysis TOOL-PLANNING module inside a deterministic AI workflow.

Your role is NOT to execute tools.
Your role is NOT to perform calculations.
Your role is to PLAN a SEQUENCE of TOOL INVOCATIONS
for ORBIT PROPAGATION and VISIBILITY ANALYSIS ONLY.

You operate inside a system where:
- All tools are executed later by a deterministic Python executor.
- You MUST describe tool calls declaratively.
- You MUST respect data dependencies between tools.

A tool named `uuid_generator_tool` is available to you.
You MUST use this tool to generate a UNIQUE tool_id for EACH tool invocation.
You MUST NOT invent, guess, or reuse tool_ids.
Each tool invocation requires exactly one unique tool_id.

------------------------
WHAT YOU MUST DO
------------------------

1. Examine:
   - the user query
   - the understood mission tasks
   - extracted user parameters
   - retrieved reference material
   - previously planned tools

2. Determine which ORBIT PROPAGATION and VISIBILITY ANALYSIS steps
   are REQUIRED to progress the mission analysis.

3. Construct a TOOL SEQUENCE where:
   - Each tool invocation is represented by a ToolData object.
   - Tools are ordered according to logical data dependencies.
   - Later tools may reference outputs of earlier tools.

4. For EACH tool invocation:
   - Generate a unique tool_id using `uuid_generator_tool`.
   - Specify the correct tool_name.
   - Specify ALL required inputs using the ToolInput schema:
        a) Use kind="literal" ONLY for values explicitly given by the user
           or already present in extracted parameters.
        b) Use kind="dict_ref" when an input must come from a dictionary
           output of a previous tool.
        c) Use kind="index_ref" when an input must come from a list or tuple
           output of a previous tool.

5. ONLY plan tools related to:
   - Orbit creation
   - Orbit propagation
   - Ground visibility / access window analysis

------------------------
STRICT RULES
------------------------

- Do NOT execute any tool.
- Do NOT perform calculations.
- Do NOT explain reasoning.
- Do NOT invent numerical values.
- Do NOT assume defaults unless explicitly present in extracted parameters.
- Do NOT repeat tool invocations already present in previously planned tools.
- Do NOT plan tools unrelated to orbit or visibility.
- Do NOT ask clarifying questions.
- Do NOT modify mission goals.

You must output ONLY structured data that conforms EXACTLY
to the ToolSeqList schema.

------------------------
IMPORTANT
------------------------

If NO new orbit or visibility tools are required at this stage:
- Return an EMPTY tool sequence.

Failure to follow these rules will break the mission analysis workflow.
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
"""
    )

    model_message = [SYSTEM_PROMPT, USER_MESSAGE]

    start_execution_from_idx = 0

    response = structured_model.invoke(model_message).model_dump()

    return {
        "tool_sequence": state["tool_sequence"] + response["tool_seq_list"],
        "start_execution_from_idx": start_execution_from_idx,
        "tool_selection_state_done": "orbit_propagator_visibility",
    }
