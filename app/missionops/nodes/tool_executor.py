from typing import Callable, Dict

from langchain_core.runnables import RunnableConfig

from app.missionops.state import MissionOpsState, ToolOutput


def tool_executor(state: MissionOpsState, config: RunnableConfig):
    """.
    Execute the planned sequence of mission analysis tools.

    This node iterates through the tool sequence starting from a specified
    index, resolves input dependencies, executes each tool, and collects
    outputs. Execution stops if a tool fails or if a dependency is missing.

    Parameters
    ----------
    state : MissionOpsState
        The current state of the MissionOps workflow, containing:
        - tool_sequence : list
            Ordered list of tool invocations to execute
        - start_execution_from_idx : int
            Index to start execution from
        - tool_outputs : dict
            Previously executed tool outputs
        - warnings : list of str
            Accumulated warnings
    config : RunnableConfig
        LangGraph configuration containing:
        - configurable["tool_registry"] : Dict[str, Callable]
            Registry mapping tool names to callable functions

    Returns
    -------
    dict
        A dictionary containing:
        - tool_outputs : dict
            Updated dictionary of tool execution results, keyed by tool_id.
            Each entry contains:
            - tool_id : str
                Identifier of the tool
            - success : bool
                Whether execution succeeded
            - error : str or None
                Error message if execution failed
            - output : Any or None
                Tool output if execution succeeded
        - warnings : list of str
            Updated list of warnings including any execution errors

    Notes
    -----
    The executor resolves three types of inputs:
    - "literal": Direct values from the tool specification
    - "dict_ref": Values from dictionary output of previous tools
    - "index_ref": Values from list/tuple output of previous tools

    Execution halts at the first tool failure or missing dependency,
    recording the error in both tool_outputs and warnings.
    """
    tool_registry: Dict[str, Callable] = config["configurable"]["tool_registry"]

    tool_outputs: Dict[str, ToolOutput] = {}
    state_warnings = list(state["warnings"])

    start_execution_from_idx = state["start_execution_from_idx"]

    for tool_ele in state["tool_sequence"][start_execution_from_idx:]:
        # Resolve the inputs
        inputs = {}
        tool_output_subdict = {}

        is_break: bool = False
        why_break: str = ""

        for input_name, tool_input_desc in tool_ele["inputs"].items():
            match tool_input_desc["kind"]:
                case "literal":
                    inputs[input_name] = tool_input_desc["value"]
                case "dict_ref":
                    if tool_input_desc["from_tool_id"] in tool_outputs:
                        if tool_outputs[tool_input_desc["from_tool_id"]]["success"]:
                            inputs[input_name] = tool_outputs[
                                tool_input_desc["from_tool_id"]
                            ]["output"][tool_input_desc["key"]]
                        else:
                            msg = unsuccessful_tool_message(
                                tool_input_desc["from_tool_id"]
                            )

                            state_warnings.append(msg)
                            is_break = True
                            why_break = msg
                            break

                    else:
                        msg = too_dependency_not_found_message(
                            tool_input_desc["from_tool_id"]
                        )

                        state_warnings.append(msg)
                        is_break = True
                        why_break = msg
                        break
                case "index_ref":
                    if tool_input_desc["from_tool_id"] in tool_outputs:
                        if tool_outputs[tool_input_desc["from_tool_id"]]["success"]:
                            inputs[input_name] = tool_outputs[
                                tool_input_desc["from_tool_id"]
                            ]["output"][tool_input_desc["index"]]
                        else:
                            msg = unsuccessful_tool_message(
                                tool_input_desc["from_tool_id"]
                            )

                            state_warnings.append(msg)
                            is_break = True
                            why_break = msg
                            break

                    else:
                        msg = too_dependency_not_found_message(
                            tool_input_desc["from_tool_id"]
                        )

                        state_warnings.append(msg)
                        is_break = True
                        why_break = msg
                        break

        if is_break:
            tool_outputs.update(
                unsuccessful_tool_dict_creator(tool_ele["tool_id"], why_break)
            )
            break

        try:
            output = None
            if tool_ele["tool_name"] in tool_registry:
                output = tool_registry[tool_ele["tool_name"]](**inputs)
            else:
                raise RuntimeError(f"Unknown tool: {tool_ele['tool_name']}")

            tool_output_subdict["tool_id"] = tool_ele["tool_id"]
            tool_output_subdict["success"] = True
            tool_output_subdict["error"] = None
            tool_output_subdict["output"] = output

        except Exception as e:
            tool_outputs.update(
                unsuccessful_tool_dict_creator(tool_ele["tool_id"], f"{e}")
            )
            break

        tool_outputs[tool_ele["tool_id"]] = tool_output_subdict

    return {
        "tool_outputs": {**state["tool_outputs"], **tool_outputs},
        "warnings": state_warnings,
    }


def unsuccessful_tool_dict_creator(tool_id: str, error: str):
    """.
    Create a dictionary entry for an unsuccessful tool execution.

    Parameters
    ----------
    tool_id : str
        Identifier of the failed tool.
    error : str
        Error message describing the failure.

    Returns
    -------
    dict
        Dictionary with tool_id as key and failure details as value,
        including tool_id, output (None), success (False), and error message.
    """
    return {
        f"{tool_id}": {
            "tool_id": tool_id,
            "output": None,
            "success": False,
            "error": error,
        }
    }


def unsuccessful_tool_message(tool_id: str) -> str:
    """.
    Generate an error message for accessing output from a failed tool.

    Parameters
    ----------
    tool_id : str
        Identifier of the unsuccessful tool.

    Returns
    -------
    str
        Error message indicating that output cannot be accessed from
        the unsuccessful tool execution.
    """
    return f"Can not access output value from unsuccessful tool execution. Tool id: {tool_id}"


def too_dependency_not_found_message(tool_id: str) -> str:
    """.
    Generate an error message for a missing tool dependency.

    Parameters
    ----------
    tool_id : str
        Identifier of the missing dependency tool.

    Returns
    -------
    str
        Error message indicating that the required tool dependency
        was not found in the execution outputs.
    """
    return f"Tool dependency not found. Tool id: {tool_id}"
