import asyncio
import logging
import os
import uuid
from typing import Literal

from dotenv import load_dotenv
from langchain.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from app.logging_config import setup_logging
from app.missionops.missionops_app import main as missionops_main
from app.orbitqa.orbitqa_app import main as orbitqa_main


class AstroBrainModelSelectionResponse(BaseModel):
    model_to_select: Literal["orbitqa", "missionops"]


def main():
    load_dotenv("./.env")
    setup_logging()

    user_query = input("How may I help you: ")

    astrobrain_model_settings = {
        "model": os.getenv("model_name_astrobrain"),
        "temperature": float(os.getenv("model_temperature_astrobrain")),
    }

    llm = ChatOllama(**astrobrain_model_settings)
    structured_llm = llm.with_structured_output(AstroBrainModelSelectionResponse)

    # Write a system prompt for me
    SYSTEM_PROMPT = SystemMessage(
        """You are an aerospace AI agent router inside AstroBrain system.

Your ONLY role is to analyze the user's query and select the appropriate AI agent.

You have access to TWO specialized agents:

1. **OrbitQA Agent**
   Purpose: Handles straightforward orbital mechanics questions, computations, and visualizations.
   Use when the request involves:
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

    # Write a human message for me
    USER_MESSAGE = HumanMessage(
        f"""User Query: {user_query}

Based on the query above, select the appropriate agent to handle this request.
"""
    )

    llm_messages = [SYSTEM_PROMPT, USER_MESSAGE]

    llm_response = structured_llm.invoke(llm_messages).model_dump()

    result = None

    if llm_response["model_to_select"] == "orbitqa":
        result = asyncio.run(orbitqa_main(uuid.uuid4(), user_query))
    else:
        result = asyncio.run(missionops_main(uuid.uuid4(), user_query))

    print(result)
