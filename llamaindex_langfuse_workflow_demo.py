"""
LlamaIndex Workflows + Langfuse 3.x Integration Demo
Using Google Gemini 3 Pro Preview with Async Streaming

Based on: https://langfuse.com/integrations/frameworks/llamaindex-workflows
"""

import asyncio
import json
import os
from typing import Annotated

from dotenv import load_dotenv
from google.genai import types

# Load environment variables
load_dotenv()

# --- Step 1: Set Up Environment Variables ---
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE", "service_account.json")
abs_path = os.path.abspath(SERVICE_ACCOUNT_FILE)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = abs_path

with open(abs_path, "r") as f:
    service_account_info = json.load(f)
    project_id = service_account_info["project_id"]

# --- Step 2: Initialize Langfuse Client (v3.x) ---
from langfuse import get_client

langfuse = get_client()

if langfuse.auth_check():
    print("✅ Langfuse 3.x client is authenticated!")
else:
    print("❌ Authentication failed.")
    exit(1)

# --- Step 3: Initialize LlamaIndex Instrumentation ---
# This automatically captures LlamaIndex operations and exports to Langfuse
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

LlamaIndexInstrumentor().instrument()
print("✅ LlamaIndex instrumentation initialized")

# --- Step 4: Create Workflow (per official docs) ---
from llama_index.core.llms import ChatMessage
from llama_index.llms.google_genai import GoogleGenAI
from workflows import Workflow, step
from workflows.events import StartEvent, StopEvent
from workflows.resource import Resource


def get_llm(**kwargs):
    return GoogleGenAI(
        model="gemini-3-pro-preview",
        vertexai_config={"project": project_id, "location": "global"},
        generation_config=types.GenerateContentConfig(
            system_instruction="You are a helpful assistant.",
            temperature=1.0,
            thinking_config=types.ThinkingConfig(
                thinking_level="LOW",
                include_thoughts=True,
            ),
        ),
    )


class MyWorkflow(Workflow):
    @step
    async def step1(
        self,
        ev: StartEvent,
        llm: Annotated[GoogleGenAI, Resource(get_llm)],
    ) -> StopEvent:
        """Process input with async streaming."""
        user_input = ev.get("input")
        print(f"📩 Input: {user_input}")
        print("\n🤖 Streaming response:")
        print("-" * 40)

        msg = ChatMessage(role="user", content=user_input)

        # Async streaming
        response_gen = await llm.astream_chat([msg])
        full_response = ""
        async for chunk in response_gen:
            if chunk.delta:
                print(chunk.delta, end="", flush=True)
                full_response += chunk.delta

        print("\n" + "-" * 40)
        return StopEvent(result=full_response)


async def main():
    print("\n" + "=" * 60)
    print("LlamaIndex Workflows + Gemini 3 Pro + Langfuse 3.x")
    print("=" * 60 + "\n")

    w = MyWorkflow()

    # Simply run the workflow - instrumentation handles the tracing automatically
    response = await w.run(input="What is LLM observability in 2 sentences?")

    print(f"\n📊 Response length: {len(response)} chars")

    # Flush to ensure traces are sent
    langfuse.flush()
    print("✅ Traces sent to Langfuse!")


if __name__ == "__main__":
    asyncio.run(main())
