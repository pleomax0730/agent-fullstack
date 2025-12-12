"""
Simple LLM streaming demo with LlamaIndex GoogleGenAI and tool use.
"""

import json
import os

from dotenv import load_dotenv
from google.genai import types
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool
from llama_index.llms.google_genai import GoogleGenAI
from rich import print

load_dotenv()

CREDENTIALS_FILE = "service_account.json"
LLM_MODEL = "gemini-3-pro-preview"


def load_credentials() -> dict:
    """Load VertexAI credentials from file."""
    if not os.path.exists(CREDENTIALS_FILE):
        raise FileNotFoundError(f"Credentials file '{CREDENTIALS_FILE}' not found.")
    with open(CREDENTIALS_FILE, "r") as f:
        creds = json.load(f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_FILE
    return creds


# --- Tool definitions ---


def check_flight(flight: str) -> dict:
    """Gets the current status of a flight.

    Args:
        flight: The flight number to check (e.g. "AA100")
    """
    print(f"[Tool] check_flight({flight})")
    return {"flight": flight, "status": "delayed", "departure_time": "12 PM"}


def book_taxi(time: str) -> dict:
    """Book a taxi for pickup.

    Args:
        time: Time to book the taxi (e.g. "10 AM")
    """
    print(f"[Tool] book_taxi({time})")
    return {"booking_status": "success", "pickup_time": time}


TOOLS = [
    FunctionTool.from_defaults(fn=check_flight),
    FunctionTool.from_defaults(fn=book_taxi),
]
TOOLS_BY_NAME = {t.metadata.name: t for t in TOOLS}


def main():
    creds = load_credentials()

    llm = GoogleGenAI(
        model=LLM_MODEL,
        vertexai_config={"project": creds["project_id"], "location": "global"},
        generation_config=types.GenerateContentConfig(
            system_instruction="你是一位熱心的助手，總是用繁體中文回覆用戶。",
            temperature=1.0,
            thinking_config=types.ThinkingConfig(
                include_thoughts=True, thinking_level=types.ThinkingLevel.LOW
            ),
        ),
    )

    messages = [
        ChatMessage(
            role="user",
            content="Hello, how can you help me?",
        )
    ]

    max_steps = 10
    for step in range(max_steps):
        print(f"\n--- Step {step + 1} ---")

        # Stream with tools
        stream = llm.stream_chat_with_tools(TOOLS, chat_history=messages)

        # Collect streamed response
        full_response = None
        for chunk in stream:
            print(chunk)
            breakpoint()
            full_response = chunk
            # Print text chunks as they arrive
            if chunk.message.content:
                print(chunk.message.content, end="", flush=True)

        if not full_response:
            break

        # Check for tool calls
        tool_calls = llm.get_tool_calls_from_response(
            full_response, error_on_no_tool_call=False
        )

        if not tool_calls:
            # Final response
            print(f"\n\n[Final] {full_response.message.content}")
            break

        # Add assistant message to history
        messages.append(full_response.message)

        # Execute tools and add responses
        for tc in tool_calls:
            tool = TOOLS_BY_NAME.get(tc.tool_name)
            if tool:
                result = tool.call(**tc.tool_kwargs)
                print(f"  -> {result}")
                messages.append(
                    ChatMessage(
                        role="tool",
                        content=str(result),
                        additional_kwargs={"tool_call_id": tc.tool_id},
                    )
                )


if __name__ == "__main__":
    main()
