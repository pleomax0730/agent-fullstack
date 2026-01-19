"""
LiteLLM replication of langchain_tool_call_demo.py using Gemini 3 on Vertex AI.
Demonstrates streaming with tool use and thinking mode, preserving thought signatures.
"""

import base64
import json
import os
import sys
from typing import Dict, Any, List

from dotenv import load_dotenv
from rich import print
from rich.console import Console

import litellm
from litellm import completion

# litellm._turn_on_debug()

# Load environment variables
load_dotenv()

# Configuration
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE", "service_account.json")
LLM_MODEL = os.getenv("LLM_MODEL", "vertex_ai/gemini-3-pro-preview")

# --- Langfuse Logging Integration (v2.x) ---
# Using langfuse<3.0.0 for stable LiteLLM integration
if os.getenv("LANGFUSE_PUBLIC_KEY"):
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]
    print("[bold green]✓ Langfuse logging enabled (v2 mode)[/bold green]")
else:
    print("[dim]ℹ Langfuse not configured (set LANGFUSE_* env vars to enable)[/dim]")

# Initialize rich console
console = Console()


def load_credentials() -> Dict[str, Any]:
    """Load Google Cloud service account credentials from file defined in .env."""
    if SERVICE_ACCOUNT_FILE is None:
        raise EnvironmentError("Missing environment variable: SERVICE_ACCOUNT_FILE")

    # Convert to absolute path
    abs_path = os.path.abspath(SERVICE_ACCOUNT_FILE)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Service account file not found: {abs_path}")

    # Load the service account JSON key
    with open(abs_path, "r") as f:
        service_account_info = json.load(f)

    # Set ADC environment variable so Google Cloud SDKs/LiteLLM usage works
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = abs_path
    return service_account_info


def message_to_dict(msg) -> dict:
    """Convert a LiteLLM message object to a serializable dict."""
    if isinstance(msg, dict):
        return msg

    # Handle LiteLLM/OpenAI message objects
    result = {}

    # Common fields
    for field in ["role", "content", "name", "function_call", "reasoning_content"]:
        if hasattr(msg, field):
            val = getattr(msg, field)
            if val is not None:
                result[field] = val

    # Tool calls with provider_specific_fields
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        result["tool_calls"] = []
        for tc in msg.tool_calls:
            # Handle both dict-based and object-based tool calls
            if isinstance(tc, dict):
                tc_dict = {
                    "id": tc.get("id"),
                    "type": tc.get("type", "function"),
                    "function": tc.get("function", {}),
                }
                if "provider_specific_fields" in tc:
                    tc_dict["provider_specific_fields"] = tc["provider_specific_fields"]
            else:
                # Object-based (LiteLLM ChatCompletionMessageToolCall)
                func_obj = tc.function
                tc_dict = {
                    "id": tc.id,
                    "type": getattr(tc, "type", "function"),
                    "function": {
                        "name": (
                            func_obj.name
                            if hasattr(func_obj, "name")
                            else func_obj.get("name")
                        ),
                        "arguments": (
                            func_obj.arguments
                            if hasattr(func_obj, "arguments")
                            else func_obj.get("arguments")
                        ),
                    },
                }
                if (
                    hasattr(tc, "provider_specific_fields")
                    and tc.provider_specific_fields
                ):
                    tc_dict["provider_specific_fields"] = tc.provider_specific_fields

            result["tool_calls"].append(tc_dict)

    return result


def save_messages_to_file(messages: List, filename: str):
    """Save all messages with metadata to a JSON file."""
    serializable = []
    for i, msg in enumerate(messages):
        entry = {"index": i, "message": message_to_dict(msg)}
        serializable.append(entry)

    filepath = os.path.join(os.path.dirname(__file__), filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(f"[bold blue]💾 Messages saved to:[/bold blue] {filepath}")


# --- Tool Definitions using Pydantic ---

from pydantic import BaseModel, Field
from typing import Literal


# Pydantic models for tool parameters
class CheckFlightInput(BaseModel):
    """Input schema for checking flight status."""

    flight: str = Field(description="The flight number to check (e.g. 'AA100')")


class BookTaxiInput(BaseModel):
    """Input schema for booking a taxi."""

    time: str = Field(description="Time to book the taxi (e.g. '10 AM')")
    location: Literal["airport", "hotel", "home"] = Field(
        default="airport", description="Pickup location"
    )


# Tool execution functions
def check_flight(flight: str) -> dict:
    """Gets the current status of a flight."""
    console.print(f"[dim][Tool Exec] check_flight({flight})[/dim]")
    return {"flight": flight, "status": "delayed", "departure_time": "12 PM"}


def book_taxi(time: str, location: str = "airport") -> dict:
    """Book a taxi for pickup at specified time and location."""
    console.print(f"[dim][Tool Exec] book_taxi({time}, {location})[/dim]")
    return {"booking_status": "success", "pickup_time": time, "location": location}


# Map for execution
TOOLS_MAP = {"check_flight": check_flight, "book_taxi": book_taxi}


# Helper function to convert Pydantic model to tool schema
def pydantic_to_tool(model: type[BaseModel], name: str, description: str) -> dict:
    """Convert a Pydantic model to OpenAI-compatible tool schema."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": model.model_json_schema(),
        },
    }


# Generate TOOLS_SCHEMA from Pydantic models
TOOLS_SCHEMA = [
    pydantic_to_tool(
        CheckFlightInput, "check_flight", "Gets the current status of a flight."
    ),
    pydantic_to_tool(
        BookTaxiInput,
        "book_taxi",
        "Book a taxi for pickup at specified time and location.",
    ),
]


def print_signatures(message) -> None:
    """Print thought signatures from message for debugging."""
    # In LiteLLM/OpenAI format, provider specific fields are attached to tool calls
    if hasattr(message, "tool_calls") and message.tool_calls:
        print("\n[bold cyan][Tool Call Thought Signatures][/bold cyan]")
        for tc in message.tool_calls:
            # Check for provider_specific_fields in the tool call object
            # Note: Access pattern depends on LiteLLM's internal structure or the object returned
            # Typically it's in provider_specific_fields or similar

            # Try to access as attribute or dict
            provider_fields = getattr(tc, "provider_specific_fields", None)
            if not provider_fields and isinstance(tc, dict):
                provider_fields = tc.get("provider_specific_fields")

            if provider_fields and "thought_signature" in provider_fields:
                sig = provider_fields["thought_signature"]
                print(
                    f"  {tc.id if hasattr(tc, 'id') else tc.get('id')}: {len(sig)} chars (signature present)"
                )
            else:
                print(
                    f"  {tc.id if hasattr(tc, 'id') else tc.get('id')}: [red]No signature found[/red]"
                )


def test_streaming_with_tools():
    """Test streaming with tool calls - signatures preserved via stream_chunk_builder."""
    print("\n[bold green]═══ Streaming Mode with Tools (LiteLLM) ═══[/bold green]\n")

    creds = load_credentials()

    # Generate unique trace_id for this session (for Langfuse grouping)
    import uuid

    trace_id = f"stream-demo-{uuid.uuid4().hex[:8]}"
    print(f"[dim]Langfuse trace_id: {trace_id}[/dim]")

    messages = [
        {
            "role": "user",
            "content": "Check flight AA100 and book a taxi 2 hours before if delayed.",
        }
    ]

    max_steps = 10

    for step in range(max_steps):
        print(f"\n[bold]--- Step {step + 1} ---[/bold]")

        # Stream response
        print("[dim]Streaming response...[/dim]")
        response_chunks = []

        # We use litellm.completion with stream=True
        # Note: reasoning_effort="medium" or similar might be required to trigger thinking if not default
        try:
            stream = completion(
                model=LLM_MODEL,
                messages=messages,
                tools=TOOLS_SCHEMA,
                stream=True,
                timeout=30,  # Safety timeout
                vertex_location="global",
                vertex_project=creds["project_id"],
                reasoning_effort="low",
                metadata={
                    "trace_id": trace_id,
                    "trace_name": "Streaming Tool Call Demo",
                    "generation_name": f"step-{step + 1}",
                },
            )

            # Process chunks
            for chunk in stream:
                response_chunks.append(chunk)
                delta = chunk.choices[0].delta

                # Check for content (text)
                if delta.content:
                    print(delta.content, end="", flush=True)

                # Check for reasoning_content (Thinking)
                # Note: LiteLLM maps this field if supported by provider
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    print(
                        f"[dim][Thinking] {delta.reasoning_content}[/dim]",
                        end="",
                        flush=True,
                    )

        except Exception as e:
            print(f"\n[bold red]Error during completion:[/bold red] {e}")
            break

        print()  # Newline after stream

        # Reconstruct full message using stream_chunk_builder
        full_response_msg = litellm.stream_chunk_builder(
            response_chunks, messages=messages
        )

        # The returned object is a complete ChatCompletionMessage
        assistant_msg = full_response_msg.choices[0].message

        # Check if there are tool calls
        if not assistant_msg.tool_calls:
            print("\n[bold green]Final Answer Reached[/bold green]")
            break

        print_signatures(assistant_msg)

        # CRITICAL: Append original assistant message to history to preserve signatures
        messages.append(assistant_msg)

        # Execute tools
        for tc in assistant_msg.tool_calls:
            func_name = tc.function.name
            args_str = tc.function.arguments

            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                print(
                    f"[red]Failed to parse arguments for {func_name}: {args_str}[/red]"
                )
                continue

            tool_fn = TOOLS_MAP.get(func_name)
            if tool_fn:
                result = tool_fn(**args)
                print(f"  [yellow]Result:[/yellow] {result}")

                # Append tool result to messages
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    }
                )
            else:
                print(f"[red]Tool {func_name} not found[/red]")

    print(f"\n[bold green]✓ Completed in {step + 1} steps[/bold green]")
    save_messages_to_file(messages, "messages_streaming.json")


def test_invoke_with_tools():
    """Test non-streaming invoke with tool calls."""
    print("\n[bold green]═══ Invoke Mode with Tools (LiteLLM) ═══[/bold green]\n")

    creds = load_credentials()

    messages = [
        {
            "role": "user",
            "content": "Check flight AA100 and book a taxi 2 hours before if delayed.",
        }
    ]

    max_steps = 10

    for step in range(max_steps):
        print(f"\n[bold]--- Step {step + 1} ---[/bold]")

        try:
            response = completion(
                model=LLM_MODEL,
                messages=messages,
                tools=TOOLS_SCHEMA,
                stream=False,
                vertex_location="global",
                vertex_project=creds["project_id"],
                reasoning_effort="low",
            )

            assistant_msg = response.choices[0].message

            # Print content/thinking
            if (
                hasattr(assistant_msg, "reasoning_content")
                and assistant_msg.reasoning_content
            ):
                print(f"[dim][Thinking] {assistant_msg.reasoning_content}[/dim]")

            if assistant_msg.content:
                print(assistant_msg.content)

            if not assistant_msg.tool_calls:
                print("\n[bold green]Final Answer Reached[/bold green]")
                break

            print_signatures(assistant_msg)

            # CRITICAL: Append original assistant message
            messages.append(assistant_msg)

            # Execute tools
            for tc in assistant_msg.tool_calls:
                func_name = tc.function.name
                args_str = tc.function.arguments

                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    print(
                        f"[red]Failed to parse arguments for {func_name}: {args_str}[/red]"
                    )
                    continue

                tool_fn = TOOLS_MAP.get(func_name)
                if tool_fn:
                    result = tool_fn(**args)
                    print(f"  [yellow]Result:[/yellow] {result}")

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(result),
                        }
                    )
                else:
                    print(f"[red]Tool {func_name} not found[/red]")

        except Exception as e:
            print(f"\n[bold red]Error during completion:[/bold red] {e}")
            break

    print(f"\n[bold green]✓ Completed in {step + 1} steps[/bold green]")
    save_messages_to_file(messages, "messages_invoke.json")


def test_model_switching():
    """Test switching from Gemini 3 to different model mid-conversation."""
    print("\n[bold green]═══ Model Switching Test ═══[/bold green]\n")

    creds = load_credentials()

    # Step 1: Start with Gemini 3 (Checking Flight)
    print("\n[bold]--- Step 1: Gemini 3 (Thinking) ---[/bold]")
    messages = [
        {
            "role": "user",
            "content": "Check flight AA100. Then, switch models to book a taxi.",
        }
    ]

    try:
        response = completion(
            model=LLM_MODEL,  # gemini-3-pro
            messages=messages,
            tools=TOOLS_SCHEMA,
            vertex_location="global",
            vertex_project=creds["project_id"],
            reasoning_effort="low",
            stream=False,
        )

        assistant_msg = response.choices[0].message
        print_signatures(assistant_msg)
        messages.append(assistant_msg)

        # Execute tool (Check Flight)
        if assistant_msg.tool_calls:
            for tc in assistant_msg.tool_calls:
                # Mock result
                result = {"flight": "AA100", "status": "delayed"}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    }
                )
                print(f"  [yellow]Result:[/yellow] {result}")

    except Exception as e:
        print(f"[red]Step 1 Failed:[/red] {e}")
        return

    # Step 2: Switch to OpenAI and continue the conversation
    print("\n[bold]--- Step 2: Switching to OpenAI (gpt-4o-mini) ---[/bold]")

    # Check if OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "[red]OPENAI_API_KEY not found in .env, skipping OpenAI switch test.[/red]"
        )
        save_messages_to_file(messages, "messages_switch.json")
        return

    NEW_MODEL = "gpt-4o-mini"
    max_steps = 5

    print("[green]Provider switched! Continuing with OpenAI...[/green]")

    for step in range(max_steps):
        print(f"\n[bold]--- OpenAI Step {step + 1} ---[/bold]")

        try:
            # Pass the messages list which contains Gemini 3's thought_signature
            response = completion(
                model=NEW_MODEL,
                messages=messages,
                tools=TOOLS_SCHEMA,
                stream=False,
            )

            assistant_msg = response.choices[0].message

            # Print content
            if assistant_msg.content:
                print(assistant_msg.content)

            # Check if there are tool calls
            if not assistant_msg.tool_calls:
                print("\n[bold green]Final Answer Reached (OpenAI)[/bold green]")
                break

            # Append assistant message
            messages.append(assistant_msg)

            # Execute tools
            for tc in assistant_msg.tool_calls:
                func_name = tc.function.name
                args_str = tc.function.arguments

                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    print(
                        f"[red]Failed to parse arguments for {func_name}: {args_str}[/red]"
                    )
                    continue

                tool_fn = TOOLS_MAP.get(func_name)
                if tool_fn:
                    result = tool_fn(**args)
                    print(f"  [yellow]Result:[/yellow] {result}")

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(result),
                        }
                    )
                else:
                    print(f"[red]Tool {func_name} not found[/red]")

        except Exception as e:
            print(f"\n[bold red]Error during OpenAI completion:[/bold red] {e}")
            break

    # Step 3: Switch BACK to Gemini 3
    print("\n[bold]--- Step 3: Switching BACK to Gemini 3 ---[/bold]")
    print(
        "[yellow]Testing if conversation with OpenAI's message (no thought_signature) works...[/yellow]"
    )

    # Add a follow-up question
    messages.append(
        {
            "role": "user",
            "content": "Great! What time should I leave home to catch my taxi?",
        }
    )

    try:
        response = completion(
            model=LLM_MODEL,  # Back to Gemini 3
            messages=messages,
            tools=TOOLS_SCHEMA,
            stream=False,
            vertex_location="global",
            vertex_project=creds["project_id"],
            reasoning_effort="low",
        )

        assistant_msg = response.choices[0].message

        # Print thinking if available
        if (
            hasattr(assistant_msg, "reasoning_content")
            and assistant_msg.reasoning_content
        ):
            print(f"[dim][Thinking] {assistant_msg.reasoning_content}[/dim]")

        # Print content
        if assistant_msg.content:
            print(f"\n[bold]Gemini 3 Response:[/bold] {assistant_msg.content}")

        print("\n[bold green]✓ Successfully switched back to Gemini 3![/bold green]")
        messages.append(assistant_msg)

    except Exception as e:
        print(f"\n[bold red]Error switching back to Gemini 3:[/bold red] {e}")
        print(
            "[yellow]This may indicate Gemini 3 requires thought signatures from previous turns.[/yellow]"
        )

    print(f"\n[bold green]✓ Switch test completed[/bold green]")

    # Save messages at the end of switching test
    save_messages_to_file(messages, "messages_switch.json")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "stream"

    if mode == "stream":
        test_streaming_with_tools()
    elif mode == "invoke":
        test_invoke_with_tools()
    elif mode == "switch":
        test_model_switching()
    else:
        print("Usage: python litellm_tool_call_demo.py [stream|invoke|switch]")
