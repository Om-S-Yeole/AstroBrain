import json
from typing import Callable, Dict, List

import mpld3
from langchain_core.runnables import RunnableConfig
from matplotlib.figure import Figure
from poliastro.plotting import OrbitPlotter

from app.core import uuid_generator
from app.orbitqa.state import OrbitQAState, PlotOutput, ToolOutput


def compute_and_plot(state: OrbitQAState, config: RunnableConfig):
    """
    Execute the planned tool sequence and generate computational results and visualizations.

    This function is the execution engine for the OrbitQA workflow. It takes the
    planned tool sequence from the tool_selector module, resolves all input
    dependencies, executes each tool in order, and collects outputs. For plotting
    tools, it converts matplotlib/poliastro figures to JSON format for later
    rendering.

    The function implements robust error handling:
    - Validates that all dependency references are available
    - Checks that referenced tools executed successfully
    - Stops execution at the first error and records warnings
    - Handles both computational and plotting tools

    Parameters
    ----------
    state : OrbitQAState
        The current workflow state containing:
        - tool_sequence : list of ToolDataPydantic
            Ordered list of tool invocations to execute.
        - warnings : list of str
            Existing warnings from previous workflow stages.

    config : RunnableConfig
        LangChain configuration object containing:
        - configurable.tool_registry : Dict[str, Callable]
            Dictionary mapping tool names to executable Python functions.
            Format: {'tool_name': tool_func, ...}

    Returns
    -------
    dict
        A dictionary with three keys:
        - tool_outputs : Dict[str, ToolOutput]
            Mapping from tool_id to execution results. Each ToolOutput contains:
            * tool_id : str - The unique identifier
            * success : bool - Whether execution succeeded
            * output : Any - The tool's return value (or None if failed)
            * error : str or None - Error message if execution failed
        - plots : List[PlotOutput]
            List of generated plots in JSON format. Each PlotOutput contains:
            * tool_id : str - The tool that generated the plot
            * plot_id : str - Unique identifier for the plot
            * plot : str - JSON representation of the figure
        - warnings : list of str
            Updated list of warnings including any execution errors.

    Notes
    -----
    Input Resolution:
    The function resolves three types of inputs for each tool:
    1. LiteralInput (kind='literal'): Direct values passed as-is
    2. DictRefInput (kind='dict_ref'): Extract value from a previous tool's
       dictionary output using from_tool_id and key
    3. IndexRefInput (kind='index_ref'): Extract value from a previous tool's
       list/tuple output using from_tool_id and index

    Error Handling:
    Execution stops immediately if:
    - A dependency tool_id is not found in tool_outputs
    - A referenced tool failed (success=False)
    - A tool raises an exception during execution
    - A tool_name is not found in the registry

    When execution stops, all subsequent tools are skipped and a warning is added.

    Plot Conversion:
    - poliastro.OrbitPlotter: Converted using backend.figure.to_json()
    - matplotlib.Figure: Converted using mpld3.fig_to_dict() then json.dumps()
    - Both formats are stored as JSON strings for later rendering

    Tool Execution Order:
    Tools are executed in the exact order they appear in tool_sequence.
    The tool_selector module ensures dependencies are satisfied.

    Examples
    --------
    >>> from app.orbitqa.state import OrbitQAState
    >>> from langchain_core.runnables import RunnableConfig
    >>> state = OrbitQAState(
    ...     tool_sequence=[
    ...         {
    ...             "tool_id": "uuid-1",
    ...             "tool_name": "compute_hohmann_transfer",
    ...             "inputs": {
    ...                 "r1": {"kind": "literal", "value": 7000},
    ...                 "r2": {"kind": "literal", "value": 42164},
    ...                 "body": {"kind": "literal", "value": "Earth"}
    ...             }
    ...         }
    ...     ],
    ...     warnings=[]
    ... )
    >>> config = RunnableConfig(configurable={
    ...     "tool_registry": {"compute_hohmann_transfer": hohmann_func}
    ... })
    >>> result = compute_and_plot(state, config)
    >>> result["tool_outputs"]["uuid-1"]["success"]
    True
    >>> result["tool_outputs"]["uuid-1"]["output"]
    {'delta_v': 3.9, 'transfer_time': 19000.0}

    With dependencies:
    >>> state.tool_sequence = [
    ...     {"tool_id": "uuid-1", "tool_name": "calc_orbit", "inputs": {...}},
    ...     {
    ...         "tool_id": "uuid-2",
    ...         "tool_name": "plot_orbit",
    ...         "inputs": {
    ...             "orbit": {
    ...                 "kind": "dict_ref",
    ...                 "from_tool_id": "uuid-1",
    ...                 "key": "orbit_obj"
    ...             }
    ...         }
    ...     }
    ... ]
    >>> result = compute_and_plot(state, config)
    >>> len(result["plots"])
    1
    >>> result["plots"][0]["tool_id"]
    'uuid-2'

    With error:
    >>> state.tool_sequence[0]["inputs"]["r1"]["value"] = -7000  # Invalid
    >>> result = compute_and_plot(state, config)
    >>> result["tool_outputs"]["uuid-1"]["success"]
    False
    >>> "ValueError" in result["tool_outputs"]["uuid-1"]["error"]
    True

    See Also
    --------
    tool_selector : Module that generates the tool_sequence.
    unsuccessful_tool_dict_creator : Helper for creating error records.
    """
    tool_registry: Dict[str, Callable] = config["configurable"]["tool_registry"]

    tool_outputs: Dict[str, ToolOutput] = {}
    plot_outputs: List[PlotOutput] = []
    state_warnings = list(state["warnings"])

    for tool_ele in state["tool_sequence"]:
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

            if isinstance(output, OrbitPlotter):
                output = output.backend.figure.to_json()  # Store JSON output
                tool_output_subdict["output"] = output

                # Update to plot list as well
                plot_outputs.append(
                    {
                        "tool_id": tool_ele["tool_id"],
                        "plot_id": uuid_generator(),
                        "plot": output,
                    }
                )
            elif isinstance(output, Figure):
                output = json.dumps(mpld3.fig_to_dict(output))
                tool_output_subdict["output"] = output

                # Update to plot list as well
                plot_outputs.append(
                    {
                        "tool_id": tool_ele["tool_id"],
                        "plot_id": uuid_generator(),
                        "plot": output,
                    }
                )
            else:
                tool_output_subdict["output"] = output

        except Exception as e:
            tool_outputs.update(
                unsuccessful_tool_dict_creator(tool_ele["tool_id"], f"{e}")
            )
            break

        tool_outputs[tool_ele["tool_id"]] = tool_output_subdict

    return {
        "tool_outputs": tool_outputs,
        "plots": plot_outputs,
        "warnings": state_warnings,
    }


def unsuccessful_tool_dict_creator(tool_id: str, error: str):
    """
    Create a standardized dictionary representing a failed tool execution.

    This helper function constructs a ToolOutput-compatible dictionary for tools
    that failed to execute. It ensures consistent error reporting across the
    workflow execution engine.

    Parameters
    ----------
    tool_id : str
        The unique identifier of the tool that failed.
    error : str
        A description of the error that caused the failure. Can be an exception
        message, dependency error, or custom error description.

    Returns
    -------
    dict
        A dictionary with a single key (the tool_id) mapping to a ToolOutput
        dictionary containing:
        - tool_id : str
            The unique identifier (same as input).
        - output : None
            No output is available from failed executions.
        - success : bool
            Always False for unsuccessful executions.
        - error : str
            The error description.

    Notes
    -----
    This function is used internally by compute_and_plot to record failures
    due to:
    - Missing dependencies
    - Failed dependency tools
    - Runtime exceptions during tool execution
    - Unknown tool names

    The returned dictionary format matches successful tool outputs, allowing
    uniform handling of execution results.

    Examples
    --------
    >>> unsuccessful_tool_dict_creator("uuid-123", "Tool not found in registry")
    {'uuid-123': {'tool_id': 'uuid-123', 'output': None, 'success': False, 'error': 'Tool not found in registry'}}

    >>> error_msg = "ValueError: radius must be positive"
    >>> result = unsuccessful_tool_dict_creator("uuid-456", error_msg)
    >>> result["uuid-456"]["success"]
    False
    >>> result["uuid-456"]["error"]
    'ValueError: radius must be positive'

    See Also
    --------
    compute_and_plot : Main function that uses this helper.
    unsuccessful_tool_message : Generate error message for failed dependencies.
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
    """
    Generate an error message for failed tool dependency access.

    This helper function creates a standardized warning message when a tool
    attempts to reference the output of another tool that failed to execute
    successfully.

    Parameters
    ----------
    tool_id : str
        The unique identifier of the tool that failed, whose output cannot
        be accessed.

    Returns
    -------
    str
        A descriptive error message indicating that the tool's output is
        unavailable due to unsuccessful execution.

    Notes
    -----
    This message is added to the workflow warnings list when:
    - A tool with kind='dict_ref' references a failed tool
    - A tool with kind='index_ref' references a failed tool

    The message helps users understand that the workflow stopped because
    a dependency did not complete successfully.

    Examples
    --------
    >>> unsuccessful_tool_message("550e8400-e29b-41d4-a716-446655440000")
    'Can not access output value from unsuccessful tool execution. Tool id: 550e8400-e29b-41d4-a716-446655440000'

    >>> msg = unsuccessful_tool_message("uuid-123")
    >>> "unsuccessful tool execution" in msg
    True

    See Also
    --------
    compute_and_plot : Function that uses this message generator.
    too_dependency_not_found_message : Similar message for missing dependencies.
    """
    return f"Can not access output value from unsuccessful tool execution. Tool id: {tool_id}"


def too_dependency_not_found_message(tool_id: str) -> str:
    """
    Generate an error message for missing tool dependency.

    This helper function creates a standardized warning message when a tool
    attempts to reference another tool that has not been executed or does not
    exist in the tool_outputs dictionary.

    Parameters
    ----------
    tool_id : str
        The unique identifier of the missing dependency tool that was referenced
        but not found.

    Returns
    -------
    str
        A descriptive error message indicating that the tool dependency was
        not found.

    Notes
    -----
    This message is added to the workflow warnings list when:
    - A tool with kind='dict_ref' references a tool_id not in tool_outputs
    - A tool with kind='index_ref' references a tool_id not in tool_outputs

    This typically indicates:
    - An error in the tool_selector's dependency ordering
    - A tool_id that was never executed
    - A tool that was skipped due to earlier failures

    The message helps diagnose workflow planning or execution issues.

    Examples
    --------
    >>> too_dependency_not_found_message("660f9511-f3ac-52e5-b827-557766551111")
    'Tool dependency not found. Tool id: 660f9511-f3ac-52e5-b827-557766551111'

    >>> msg = too_dependency_not_found_message("missing-uuid")
    >>> "dependency not found" in msg
    True

    See Also
    --------
    compute_and_plot : Function that uses this message generator.
    unsuccessful_tool_message : Similar message for failed tool access.
    tool_selector : Module responsible for correct dependency ordering.
    """
    return f"Tool dependency not found. Tool id: {tool_id}"
