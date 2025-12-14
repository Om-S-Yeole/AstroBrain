from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

from pydantic import BaseModel


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
    kind: Literal["literal"]
    value: Any


# For dict outputs
class DictRefInput(BaseModel):
    kind: Literal["dict_ref"]
    from_tool_id: str
    key: str


# For index based outputs like tuple and list
class IndexRefInput(BaseModel):
    kind: Literal["index_ref"]
    from_tool_id: str
    index: int


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
    final_response: Dict[str, Any]

    # ---- warnings -----
    warnings: List[str]
