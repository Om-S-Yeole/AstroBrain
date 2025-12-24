"""
WebSocket Server for AstroBrain
Handles real-time communication with users and supports agent interrupts for clarifications.
"""

import asyncio
import logging
import os
from typing import Literal

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langchain.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from app.logging_config import setup_logging
from app.missionops.missionops_app import main as missionops_main
from app.orbitqa.orbitqa_app import main as orbitqa_main

# Load environment variables
load_dotenv("./.env")
setup_logging()
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="AstroBrain WebSocket Server")


class AstroBrainModelSelectionResponse(BaseModel):
    model_to_select: Literal["orbitqa", "missionops"]


class ConnectionManager:
    """Manages WebSocket connections and clarification queues."""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self.clarification_queues: dict[str, asyncio.Queue] = {}
        self.processing_tasks: dict[str, asyncio.Task] = {}

    async def connect(self, websocket: WebSocket, thread_id: str):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[thread_id] = websocket
        self.clarification_queues[thread_id] = asyncio.Queue()
        logger.info(f"Client connected: {thread_id}")

    def disconnect(self, thread_id: str):
        """Remove a WebSocket connection and clean up resources."""
        if thread_id in self.active_connections:
            del self.active_connections[thread_id]
        if thread_id in self.clarification_queues:
            del self.clarification_queues[thread_id]
        if thread_id in self.processing_tasks:
            task = self.processing_tasks[thread_id]
            if not task.done():
                task.cancel()
            del self.processing_tasks[thread_id]
        logger.info(f"Client disconnected: {thread_id}")

    async def send_message(self, thread_id: str, message: dict):
        """Send a JSON message to a specific client."""
        if thread_id in self.active_connections:
            websocket = self.active_connections[thread_id]
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to {thread_id}: {e}")

    async def wait_for_clarification(self, thread_id: str, timeout: int = 600) -> str | None:
        """Wait for user clarification with timeout."""
        if thread_id not in self.clarification_queues:
            return None

        queue = self.clarification_queues[thread_id]
        try:
            clarification = await asyncio.wait_for(queue.get(), timeout=timeout)
            return clarification
        except TimeoutError:
            logger.warning(f"Clarification timeout for thread {thread_id}")
            return None

    async def provide_clarification(self, thread_id: str, clarification: str):
        """Provide clarification response from user."""
        if thread_id in self.clarification_queues:
            await self.clarification_queues[thread_id].put(clarification)


# Global connection manager
manager = ConnectionManager()


def select_agent(user_query: str) -> str:
    """
    Routes the user query to the appropriate agent (orbitqa or missionops).
    Returns: 'orbitqa' or 'missionops'
    """
    try:
        astrobrain_model_settings = {
            "model": os.getenv("model_name_astrobrain", "llama3.2:latest"),
            "temperature": float(os.getenv("model_temperature_astrobrain", "0.0")),
        }

        llm = ChatOllama(**astrobrain_model_settings)
        structured_llm = llm.with_structured_output(AstroBrainModelSelectionResponse)

        SYSTEM_PROMPT = SystemMessage(
            """You are an aerospace AI agent router inside AstroBrain system.

Your ONLY role is to analyze the user's query and select the appropriate AI agent.

You have access to TWO specialized agents:

1. **OrbitQA Agent**
   Purpose: Handles straightforward orbital mechanics questions, computations, and visualizations.
   Use when the request involves:+
   - Direct orbital calculations (position, velocity, state vectors)
   - Plotting orbits or orbital parameters
   - Answering conceptual questions about orbital mechanics
   - Computing specific orbital elements or transformations
   - Simple propagation or visibility queries
   - Quick orbital analysis without multi-stage planning

2. **MissionOps Agent**
   Purpose: Performs comprehensive mission feasibility analysis and multi-stage planning.
   Use when the request involves:
   - Mission feasibility studies or assessments
   - Multi-stage analysis (orbit + power + thermal + communication)
   - Resource constraint evaluation (power budget, thermal limits, duty cycles)
   - Eclipse analysis combined with power/thermal impacts
   - Communication window planning and analysis
   - Complex mission planning requiring multiple tool executions
   - Mission validation and feasibility verdicts
   - Requests mentioning "mission", "feasibility", "viability", or "assessment"

------------------------
DECISION RULES
------------------------

- If the query is about SIMPLE orbital calculations or plotting → select "orbitqa"
- If the query requires MISSION-LEVEL ANALYSIS or FEASIBILITY ASSESSMENT → select "missionops"
- If the query mentions power, thermal, communication, or duty cycles → select "missionops"
- If the query asks for mission viability or assessment → select "missionops"
- If the query is a conceptual question about orbits → select "orbitqa"
- If unsure but query involves complex multi-domain analysis → select "missionops"
- If unsure but query is straightforward computation → select "orbitqa"

You must output ONLY the structured model selection without explanation.
"""
        )

        USER_MESSAGE = HumanMessage(
            f"""User Query: {user_query}

Based on the query above, select the appropriate agent to handle this request.
"""
        )

        llm_messages = [SYSTEM_PROMPT, USER_MESSAGE]
        llm_response = structured_llm.invoke(llm_messages).model_dump()

        return llm_response["model_to_select"]
    except Exception as e:
        logger.error(f"Error selecting agent: {e}")
        return "orbitqa"  # Default fallback


async def get_user_clarification_ws(thread_id: str, res: dict) -> str:
    """
    WebSocket version of clarification handler.
    Sends clarification request to user and waits for response.
    Returns the raw user answer (formatting will be done by the agent).
    """
    # Extract the interrupt message (handle both string and Interrupt object)
    interrupt_msg = res["interrupt_message"]

    # Send clarification request to user
    await manager.send_message(
        thread_id,
        {
            "type": "clarification_request",
            "message": interrupt_msg,
            "timestamp": asyncio.get_event_loop().time(),
        },
    )

    # Wait for user clarification
    user_answer = await manager.wait_for_clarification(thread_id, timeout=600)
    if user_answer is None:
        raise TimeoutError("User did not provide clarification in time")

    # Return raw answer - the agent will format it as: "{question}: User's answer to the question: {answer}"
    return user_answer


async def process_query(thread_id: str, user_query: str):
    """
    Process user query through the appropriate agent with interrupt handling.
    Agent is automatically selected based on query content.
    """
    try:
        # Send processing status
        await manager.send_message(
            thread_id,
            {
                "type": "status",
                "message": "Processing your query...",
                "status": "processing",
            },
        )

        # Automatically select the appropriate agent
        agent_type = select_agent(user_query)

        # Send agent selection info
        await manager.send_message(
            thread_id,
            {
                "type": "status",
                "message": f"Query routed to {agent_type.upper()} agent",
                "agent": agent_type,
            },
        )

        # Create modified agent functions that use WebSocket clarification handler
        if agent_type == "orbitqa":
            # Patch the clarification function for orbitqa
            from app.orbitqa import orbitqa_app

            original_clarification_fn = orbitqa_app.get_user_clarification_cli
            orbitqa_app.get_user_clarification_cli = get_user_clarification_ws

            try:
                result = await orbitqa_main(thread_id, user_query)
            finally:
                # Restore original function
                orbitqa_app.get_user_clarification_cli = original_clarification_fn

        else:  # missionops
            # Patch the clarification function for missionops
            from app.missionops import missionops_app

            original_clarification_fn = missionops_app.get_user_clarification_cli
            missionops_app.get_user_clarification_cli = get_user_clarification_ws

            try:
                result = await missionops_main(thread_id, user_query)
            finally:
                # Restore original function
                missionops_app.get_user_clarification_cli = original_clarification_fn

        # Send final result
        if result.get("clarification_limit_exceeded", False):
            await manager.send_message(
                thread_id,
                {
                    "type": "error",
                    "message": "Clarification limit exceeded",
                },
            )
        else:
            # Convert result to JSON-serializable format
            response_data = {
                "isInterrupted": result.get("isInterrupted", False),
                "clarification_limit_exceeded": result.get("clarification_limit_exceeded", False),
                "interrupt_message": result.get("interrupt_message", ""),
                "final_response": result.get("final_response"),
            }

            await manager.send_message(
                thread_id,
                {
                    "type": "response",
                    "data": response_data,
                    "timestamp": asyncio.get_event_loop().time(),
                },
            )

    except TimeoutError:
        await manager.send_message(
            thread_id,
            {
                "type": "error",
                "message": "Session timeout - no response received in time",
                "code": "TIMEOUT",
            },
        )
    except Exception as e:
        logger.error(f"Error processing query for {thread_id}: {e}", exc_info=True)
        await manager.send_message(
            thread_id,
            {
                "type": "error",
                "message": f"Error processing query: {str(e)}",
                "code": "PROCESSING_ERROR",
            },
        )


@app.websocket("/ws/{thread_id}")
async def websocket_endpoint(websocket: WebSocket, thread_id: str):
    """
    Main WebSocket endpoint for client connections.
    Handles query processing and clarification flow.
    """
    await manager.connect(websocket, thread_id)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message_type = data.get("type")

            if message_type == "query":
                # User sent a new query
                user_query = data.get("message", "").strip()

                if not user_query:
                    await manager.send_message(
                        thread_id, {"type": "error", "message": "Empty query received"}
                    )
                    continue

                # Process query in background task (agent auto-selected)
                task = asyncio.create_task(process_query(thread_id, user_query))
                manager.processing_tasks[thread_id] = task

            elif message_type == "clarification":
                # User sent clarification response
                clarification = data.get("message", "").strip()
                if clarification:
                    await manager.provide_clarification(thread_id, clarification)
                else:
                    await manager.send_message(
                        thread_id, {"type": "error", "message": "Empty clarification received"}
                    )

            elif message_type == "ping":
                # Keep-alive ping
                await manager.send_message(thread_id, {"type": "pong"})

            else:
                await manager.send_message(
                    thread_id, {"type": "error", "message": f"Unknown message type: {message_type}"}
                )

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {thread_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {thread_id}: {e}", exc_info=True)
    finally:
        manager.disconnect(thread_id)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "AstroBrain WebSocket Server",
        "version": "1.0.0",
        "endpoints": {
            "websocket": "/ws/{thread_id}",
        },
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "active_connections": len(manager.active_connections),
        "processing_tasks": len(manager.processing_tasks),
    }


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the WebSocket server."""
    logger.info(f"Starting AstroBrain WebSocket Server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    # Get host and port from environment or use defaults
    HOST = os.getenv("WS_HOST", "0.0.0.0")
    PORT = int(os.getenv("WS_PORT", "8000"))

    start_server(HOST, PORT)
