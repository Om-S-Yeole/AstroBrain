from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

from pydantic import BaseModel, Field


# ----- For 'understand' to define the tasks ------
class TaskDescription(TypedDict):
    task: str

    # Any user given param which can be attributed to this specific task (same param will be stored in 'user_passed_params' as well)
    params: Dict[str, Any]
    constraints: List[str]


# ---- For docs retrieved from vector database ----
class RetrievedDoc(TypedDict):
    text: str
    source: str
    page_no: Optional[int]


# ----- For tools ------


# For literal outputs
class LiteralInput(BaseModel):
    kind: Literal["literal"] = Field(
        ..., description="Input is a literal value and can be taken as it is"
    )
    value: Any = Field(..., description="Value of the input")


# For dict outputs
class DictRefInput(BaseModel):
    kind: Literal["dict_ref"] = Field(
        ..., description="Input value must be accessed from dictionary"
    )
    from_tool_id: str = Field(
        ...,
        description="Tool id of the tool from whom output the input value is to be taken. Output of this tool is dictionary.",
    )
    key: str = Field(
        ...,
        description="Which key of the tool's output dictionary must be accessed to get the input value.",
    )


# For index based outputs like tuple and list
class IndexRefInput(BaseModel):
    kind: Literal["index_ref"] = Field(
        ...,
        description="Input value must be accessed from tuple or list (Index based sequences).",
    )
    from_tool_id: str = Field(
        ...,
        description="Tool id of the tool from whom output the input value is to be taken. Output of this tool is list, or tuple or any other index based sequence which as inherent order. But not python dictionary.",
    )
    index: int = Field(
        ...,
        description="Which index of the tool's output sequence like structure must be accessed to get the input value.",
    )


ToolInput = Union[LiteralInput, DictRefInput, IndexRefInput]


# --- Every tool schema will be like this ---
class ToolData(TypedDict):
    tool_id: str
    tool_name: str

    # str = param name & ToolInput = how to get value for that param
    inputs: Dict[str, ToolInput]


# --- For output of every tool ---
class ToolOutput(TypedDict):
    tool_id: str
    output: Any  # May be dict, tuple, float, etc.
    success: bool
    error: Optional[str]


# --- For output of plots ---
class PlotOutput(TypedDict):
    tool_id: str
    plot_id: str
    plot: Any


# ---- For final response ---
class FinalResponse(TypedDict):
    status: Literal["success", "denied", "error"]
    reason: str
    message: str
    plots: List[PlotOutput]
    warnings: List[str]


# -------------------------------------------------
# Final state of OrbitQA agent
# -------------------------------------------------


class OrbitQAState(TypedDict):
    # ----  For 'read_request' ----
    user_query: str

    # ---- For 'understand' ----
    understood_request: List[TaskDescription]
    user_passed_params: Dict[str, Any]  # Store user given params globally
    request_action: Literal["toDeny", "toClarify", "toProceed"]
    to_ask: str

    # ---- For 'clarify' ----
    user_clarifications: List[str]

    # ---- For 'retrieve' ----
    data_query: List[str]

    # ---- For 'retriever' ----
    retrieved_docs: List[RetrievedDoc]

    # ---- For 'tool_selector' ----
    tool_sequence: List[ToolData]

    # ---- For 'compute_and_plot' ----
    # The str will be the key of the tool from which output is created (This is for faster lookups)
    tool_outputs: Dict[str, ToolOutput]
    plots: List[PlotOutput]

    # ---- For 'draft_final_response' ----
    final_response: FinalResponse

    # ---- warnings -----
    warnings: List[str]
