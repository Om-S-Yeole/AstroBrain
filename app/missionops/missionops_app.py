import asyncio
import os
from typing import Optional, TypedDict

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.types import Command

from app.missionops.graph import build_graph
from app.missionops.state import FinalResponse
from app.missionops.tool_factory import (
    ALL_TOOL_REGISTRY,
    BASE_TOOL_REGISTRY,
    COMMUNICATION_TOOL_REGISTRY,
    DUTY_CYCLE_TOOL_REGISTRY,
    ECLIPSE_TOOL_REGISTRY,
    MISSION_SUMMARY_TOOL_REGISTRY,
    POWER_TOOL_REGISTRY,
    PROPAGATOR_TOOL_REGISTRY,
    SUN_GEOMETRY_TOOL_REGISTRY,
    THERMAL_TOOL_REGISTRY,
    VISIBILITY_TOOL_REGISTRY,
)
from app.rag import VectorStore


class MissionOpsRes(TypedDict):
    isInterrupted: bool
    clarification_limit_exceeded: bool
    interrupt_message: str
    final_response: FinalResponse | None


# --------- Globally shared resources ------------

load_dotenv("./.env")

vectorstore = VectorStore(
    api_key_pinecone=os.getenv("pinecone_api_key"),
    index_name=os.getenv("pinecone_index_name"),
    namespace=os.getenv("pinecone_namespace_name"),
    embedding_model=os.getenv("embedding_model"),
    api_key_embedder=None,
    dimensions=int(os.getenv("embedding_dimensions")),
    ollama_model=None,
    hugging_face_model=os.getenv("hugging_face_model"),
    hf_device=os.getenv("hf_device"),
    cloud=os.getenv("cloud"),
    region=os.getenv("region"),
    index_deletion_protection=(True if os.getenv("index_deletion_protection") == "True" else False),
)

model_config_per_node = {
    "understand": {
        "model": os.getenv("understand_model_name_missionops"),
        "temperature": float(os.getenv("understand_model_temp_missionops")),
    },
    "retrieve": {
        "model": os.getenv("retrieve_model_name_missionops"),
        "temperature": float(os.getenv("retrieve_model_temp_missionops")),
    },
    "re_evaluate": {
        "model": os.getenv("re_evaluate_model_name_missionops"),
        "temperature": float(os.getenv("re_evaluate_model_temp_missionops")),
    },
    "orbit_propagator_visibility": {
        "model": os.getenv("orbit_propagator_visibility_model_name_missionops"),
        "temperature": float(os.getenv("orbit_propagator_visibility_model_temp_missionops")),
    },
    "sun_eclipse": {
        "model": os.getenv("sun_eclipse_model_name_missionops"),
        "temperature": float(os.getenv("sun_eclipse_model_temp_missionops")),
    },
    "power_comm_thermal_duty": {
        "model": os.getenv("power_comm_thermal_duty_model_name_missionops"),
        "temperature": float(os.getenv("power_comm_thermal_duty_model_temp_missionops")),
    },
    "mission_summary": {
        "model": os.getenv("mission_summary_model_name_missionops"),
        "temperature": float(os.getenv("mission_summary_model_temp_missionops")),
    },
    "validator": {
        "model": os.getenv("validator_model_name_missionops"),
        "temperature": float(os.getenv("validator_model_temp_missionops")),
    },
    "draft_final_response": {
        "model": os.getenv("draft_final_response_model_name_missionops"),
        "temperature": float(os.getenv("draft_final_response_model_temp_missionops")),
    },
}

models_per_nodes = {key: ChatOllama(**value) for key, value in model_config_per_node.items()}

graph = build_graph()

config = {
    "configurable": {
        "understand_model": models_per_nodes["understand"],
        "retrieve_model": models_per_nodes["retrieve"],
        "re_evaluate_model": models_per_nodes["re_evaluate"],
        "orbit_propagator_visibility_model": models_per_nodes[
            "orbit_propagator_visibility"
        ].bind_tools(BASE_TOOL_REGISTRY + PROPAGATOR_TOOL_REGISTRY + VISIBILITY_TOOL_REGISTRY),
        "sun_eclipse_model": models_per_nodes["sun_eclipse"].bind_tools(
            BASE_TOOL_REGISTRY + SUN_GEOMETRY_TOOL_REGISTRY + ECLIPSE_TOOL_REGISTRY
        ),
        "power_comm_thermal_duty_model": models_per_nodes["power_comm_thermal_duty"].bind_tools(
            BASE_TOOL_REGISTRY
            + POWER_TOOL_REGISTRY
            + COMMUNICATION_TOOL_REGISTRY
            + THERMAL_TOOL_REGISTRY
            + DUTY_CYCLE_TOOL_REGISTRY
        ),
        "mission_summary_model": models_per_nodes["mission_summary"].bind_tools(
            BASE_TOOL_REGISTRY + MISSION_SUMMARY_TOOL_REGISTRY
        ),
        "validator_model": models_per_nodes["validator"],
        "final_response_model": models_per_nodes["draft_final_response"],
        "vectorstore": vectorstore,
        "top_k": 7,
        "tool_registry": ALL_TOOL_REGISTRY,
    }
}

# ---------------------------------------------------------

# -------- For CLI testing purposes ----------------


async def get_user_clarification_cli(thread_id: str, res: MissionOpsRes):
    clarification = input(f"{res['interrupt_message']} ")

    return clarification


# --------------------------------------------------


async def main(thread_id: str, user_req: str) -> MissionOpsRes:
    # We will use UUID. We expect frontend to pass this id unique per user
    request_config = {
        "configurable": {
            **config["configurable"],
            "thread_id": thread_id,
        }
    }

    result = await asyncio.to_thread(graph.invoke, {"user_query": user_req}, request_config)

    num_clarifications = 0
    CLARIFICATION_LIMIT = 5

    while "__interrupt__" in result:
        if num_clarifications >= CLARIFICATION_LIMIT:
            return {
                "isInterrupted": True,
                "clarification_limit_exceeded": True,
                "interrupt_message": "You have exhausted the clarification limits",
                "final_response": None,
            }

        # The interrupt message for user
        msg = result["__interrupt__"]
        res = {
            "isInterrupted": True,
            "clarification_limit_exceeded": False,
            "interrupt_message": msg,
            "final_response": None,
        }

        # We will write this function later 'get_user_clarification' for reciving inputs from web user
        try:
            user_clarification = await asyncio.wait_for(
                get_user_clarification_cli(thread_id=thread_id, res=res), timeout=600
            )
        except asyncio.TimeoutError:
            return {
                "isInterrupted": False,
                "clarification_limit_exceeded": False,
                "interrupt_message": "",
                "final_response": {
                    "status": "error",
                    "reason": "SESSION_TIMEOUT",
                    "message": "Session expired due to inactivity.",
                    "warnings": [],
                },
            }

        num_clarifications += 1

        # Resume the graph
        result = await asyncio.to_thread(
            graph.invoke, Command(resume=user_clarification), request_config
        )

    res = {
        "isInterrupted": False,
        "clarification_limit_exceeded": False,
        "interrupt_message": "",
        "final_response": result["final_response"],
    }

    return res
