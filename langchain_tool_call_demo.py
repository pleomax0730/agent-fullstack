"""
LangChain ChatGoogleGenerativeAI streaming demo with tool use and thinking mode.
Tools defined with Pydantic models.

As of langchain-google-genai 4.0.0, ChatGoogleGenerativeAI supports Vertex AI
via vertexai=True parameter, superseding ChatVertexAI.

IMPORTANT: Gemini 3 requires thought signatures to be passed back with tool responses.
- Don't reconstruct AIMessage manually - use the original object
- Signatures are in additional_kwargs["__gemini_function_call_thought_signatures__"]
- Text block signatures are in content_blocks[-1]["extras"]["signature"]
"""

import base64
import json
import os
from typing import Dict, Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage
from rich import print

load_dotenv()

SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3-pro-preview")


def load_credentials() -> Dict[str, Any]:
    """Load Google Cloud service account credentials from file defined in .env."""
    if SERVICE_ACCOUNT_FILE is None:
        raise EnvironmentError("Missing environment variable: SERVICE_ACCOUNT_FILE")
    
    # Convert to absolute path to ensure reliability in different execution contexts
    abs_path = os.path.abspath(SERVICE_ACCOUNT_FILE)
    
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Service account file not found: {abs_path}")
    
    # Load the service account JSON key
    with open(abs_path, "r") as f:
        service_account_info = json.load(f)
    
    # Set ADC environment variable so Google Cloud SDKs will use this credential
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = abs_path
    
    return service_account_info


# --- Tool definitions with Pydantic models ---


class FlightCheckInput(BaseModel):
    """Input for checking flight status."""

    flight: str = Field(description="The flight number to check (e.g. 'AA100')")


class TaxiBookingInput(BaseModel):
    """Input for booking a taxi."""

    time: str = Field(description="Time to book the taxi (e.g. '10 AM')")
    location: Literal["airport", "hotel", "home"] = Field(
        default="airport", description="Pickup location"
    )


@tool(args_schema=FlightCheckInput)
def check_flight(flight: str) -> dict:
    """Gets the current status of a flight."""
    print(f"[Tool] check_flight({flight})")
    return {"flight": flight, "status": "delayed", "departure_time": "12 PM"}


@tool(args_schema=TaxiBookingInput)
def book_taxi(time: str, location: str = "airport") -> dict:
    """Book a taxi for pickup at specified time and location."""
    print(f"[Tool] book_taxi({time}, {location})")
    return {"booking_status": "success", "pickup_time": time, "location": location}


TOOLS = [check_flight, book_taxi]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def print_signatures(ai_msg) -> None:
    """Print thought signatures from AIMessage for debugging."""
    # Tool call signatures
    thought_sigs = ai_msg.additional_kwargs.get(
        "__gemini_function_call_thought_signatures__", {}
    )
    if thought_sigs:
        print("\n[bold cyan][Tool Call Thought Signatures][/bold cyan]")
        for tool_call_id, sig_b64 in thought_sigs.items():
            decoded = base64.b64decode(sig_b64)
            print(f"  {tool_call_id}: {len(decoded)} bytes")

    # Text block signatures (in content extras)
    if isinstance(ai_msg.content, list):
        for i, block in enumerate(ai_msg.content):
            if isinstance(block, dict) and block.get("extras", {}).get("signature"):
                sig = block["extras"]["signature"]
                print(f"\n[bold cyan][Text Block {i} Signature][/bold cyan]")
                print(f"  Length: {len(sig)} chars (base64)")


def test_streaming_with_tools():
    """Test streaming with tool calls - signatures preserved via chunk accumulation."""
    print("\n[bold green]═══ Streaming Mode with Tools ═══[/bold green]\n")

    creds = load_credentials()
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        vertexai=True,
        project=creds["project_id"],
        location="global",
        include_thoughts=True,
        temperature=1.0,
    )
    llm_with_tools = llm.bind_tools(TOOLS)

    messages = [
        HumanMessage(
            content="Check flight AA100 and book a taxi 2 hours before if delayed."
        )
    ]

    max_steps = 10
    for step in range(max_steps):
        print(f"\n[bold]--- Step {step + 1} ---[/bold]")

        # Stream and accumulate response
        full_response = None
        for chunk in llm_with_tools.stream(messages):
            full_response = chunk if full_response is None else full_response + chunk

            # Print thinking/text as it streams
            if isinstance(chunk.content, list):
                for block in chunk.content:
                    if isinstance(block, dict):
                        if block.get("type") == "thinking":
                            print(
                                f"[dim][Thinking] {block.get('thinking', '')}[/dim]",
                                end="",
                                flush=True,
                            )
                        elif block.get("type") == "text":
                            print(block.get("text", ""), end="", flush=True)
            elif chunk.content:
                print(chunk.content, end="", flush=True)

        if not full_response:
            break

        tool_calls = full_response.tool_calls

        if not tool_calls:
            print()
            break

        print()

        # Show signatures are preserved
        print_signatures(full_response)

        # CRITICAL: Append original AIMessage to preserve signatures
        messages.append(full_response)

        # Execute tools
        for tc in tool_calls:
            tool_fn = TOOLS_BY_NAME.get(tc["name"])
            if tool_fn:
                result = tool_fn.invoke(tc["args"])
                print(f"  [yellow]Result:[/yellow] {result}")
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    print(f"\n[bold green]✓ Completed in {step + 1} steps[/bold green]")


def test_invoke_with_tools():
    """Test non-streaming invoke with tool calls."""
    print("\n[bold green]═══ Invoke Mode with Tools ═══[/bold green]\n")

    creds = load_credentials()
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        vertexai=True,
        project=creds["project_id"],
        location="global",
        include_thoughts=True,
        temperature=1.0,
    )
    llm_with_tools = llm.bind_tools(TOOLS)

    messages = [
        HumanMessage(
            content="Check flight AA100 and book a taxi 2 hours before if delayed."
        )
    ]

    max_steps = 10
    for step in range(max_steps):
        print(f"\n[bold]--- Step {step + 1} ---[/bold]")

        # Non-streaming invoke
        ai_msg = llm_with_tools.invoke(messages)

        # Print content
        if isinstance(ai_msg.content, list):
            for block in ai_msg.content:
                if isinstance(block, dict):
                    if block.get("type") == "thinking":
                        print(f"[dim][Thinking] {block.get('thinking', '')}[/dim]")
                    elif block.get("type") == "text":
                        print(block.get("text", ""))
        elif ai_msg.content:
            print(ai_msg.content)

        tool_calls = ai_msg.tool_calls

        if not tool_calls:
            break

        # Show signatures
        print_signatures(ai_msg)

        # CRITICAL: Append original AIMessage to preserve signatures
        messages.append(ai_msg)

        # Execute tools
        for tc in tool_calls:
            tool_fn = TOOLS_BY_NAME.get(tc["name"])
            if tool_fn:
                result = tool_fn.invoke(tc["args"])
                print(f"  [yellow]Result:[/yellow] {result}")
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    print(f"\n[bold green]✓ Completed in {step + 1} steps[/bold green]")


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "stream"

    if mode == "stream":
        test_streaming_with_tools()
    elif mode == "invoke":
        test_invoke_with_tools()
    else:
        print("Usage: python langchain_stream_tool_demo.py [stream|invoke]")
