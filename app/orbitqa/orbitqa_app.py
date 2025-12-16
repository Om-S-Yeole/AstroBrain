import asyncio
import os
from typing import Optional, TypedDict

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.types import Command

from app.orbitqa.graph import build_graph
from app.orbitqa.state import FinalResponse
from app.orbitqa.tool_factory import TOOL_REGISTRY
from app.rag import VectorStore


class OrbitQARes(TypedDict):
    isInterrupted: bool
    clarification_limit_exceeded: bool
    interrupt_message: str
    final_response: Optional[FinalResponse]


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
    index_deletion_protection=(
        True if os.getenv("index_deletion_protection") == "True" else False
    ),
)

# -------- Ideal -----------------
# model_config_per_node = {
#     "understand": {
#         "model": "llama3.1:8b-instruct-q4_K_M",
#         "temperature": 0.0,
#     },
#     "retrieve": {"model": "mistral:7b", "temperature": 0.4},
#     "tool_selector": {
#         "model": "llama3.1:8b-instruct-q4_K_M",
#         "temperature": 0.0,
#     },
#     "draft_final_response": {
#         "model": "mistral:7b",
#         "temperature": 0.4,
#     },
# }

# -------- For testing ---------------

model_config_per_node = {
    "understand": {
        "model": "llama3.1:8b-instruct-q4_K_M",
        "temperature": 0.0,
    },
    "retrieve": {"model": "llama3.1:8b-instruct-q4_K_M", "temperature": 0.4},
    "tool_selector": {
        "model": "llama3.1:8b-instruct-q4_K_M",
        "temperature": 0.0,
    },
    "draft_final_response": {
        "model": "llama3.1:8b-instruct-q4_K_M",
        "temperature": 0.4,
    },
}

# ------- For testing --------------

models_per_nodes = {
    key: ChatOllama(**value) for key, value in model_config_per_node.items()
}

graph = build_graph()

config = {
    "configurable": {
        "understand_model": models_per_nodes["understand"],
        "retrieve_model": models_per_nodes["retrieve"],
        "vectorstore": vectorstore,
        "top_k": 7,
        "tool_registry": TOOL_REGISTRY,
        "tool_selector_model": models_per_nodes["tool_selector"].bind_tools(
            TOOL_REGISTRY
        ),
        "final_response_model": models_per_nodes["draft_final_response"],
    }
}

# ---------------------------------------------------------


# -------- For testing CLI purposes ----------------
async def get_user_clarification_cli(thread_id: str, res: OrbitQARes):
    clarification = input(f"{res['interrupt_message']} ")

    return clarification


async def main(thread_id: str, user_req: str) -> OrbitQARes:
    # We will use UUID. We expect frontend to pass this id unique per user
    request_config = {
        "configurable": {
            **config["configurable"],
            "thread_id": thread_id,
        }
    }

    result = await asyncio.to_thread(
        graph.invoke, {"user_query": user_req}, request_config
    )

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
                    "plots": [],
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
