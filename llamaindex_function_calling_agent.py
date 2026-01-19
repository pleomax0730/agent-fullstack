"""
LlamaIndex Function Calling Agent Workflow
Using Gemini 3 Pro Preview + Simple List History (No Memory)

Based on: https://developers.llamaindex.ai/python/examples/workflow/function_calling_agent/
"""

import asyncio
import json
import os
import base64
from typing import Any, List

from dotenv import load_dotenv
from google.genai import types

# Load environment variables
load_dotenv()

# --- Setup Credentials ---
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE", "service_account.json")
abs_path = os.path.abspath(SERVICE_ACCOUNT_FILE)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = abs_path

with open(abs_path, "r") as f:
    service_account_info = json.load(f)
    project_id = service_account_info["project_id"]

# --- Langfuse Setup (Optional) ---
from langfuse import get_client

langfuse = get_client()
if langfuse.auth_check():
    print("✅ Langfuse connected")

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

LlamaIndexInstrumentor().instrument()

# --- Imports ---
from llama_index.core.llms import ChatMessage
from llama_index.core.base.llms.types import ThinkingBlock, TextBlock, ToolCallBlock
from llama_index.core.tools import FunctionTool, ToolSelection, ToolOutput
from llama_index.core.tools.types import BaseTool
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
    Event,
)
from llama_index.llms.google_genai import GoogleGenAI


# --- Workflow Events ---
class InputEvent(Event):
    """Event containing chat history."""

    input: List[ChatMessage]


class StreamEvent(Event):
    """Event for streaming text output."""

    delta: str


class ThinkingEvent(Event):
    """Event for thinking/reasoning output."""

    delta: str


class ToolCallEvent(Event):
    """Event for tool calls."""

    tool_calls: List[ToolSelection]


class FunctionOutputEvent(Event):
    """Event for tool output."""

    output: ToolOutput


# --- Function Calling Agent Workflow ---
class FunctionCallingAgent(Workflow):
    """
    A function calling agent that uses simple list append for chat history.
    No ChatMemoryBuffer - just a plain Python list.
    """

    def __init__(
        self,
        *args: Any,
        llm: GoogleGenAI,
        tools: List[BaseTool] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tools = tools or []
        self.llm = llm

    @step
    async def prepare_chat_history(self, ctx: Context, ev: StartEvent) -> InputEvent:
        """Prepare chat history from simple list."""
        # Initialize or get existing chat history (simple list)
        chat_history: List[ChatMessage] = await ctx.store.get(
            "chat_history", default=[]
        )

        # Clear sources for this turn
        await ctx.store.set("sources", [])

        # Add user message to history
        user_input = ev.input
        user_msg = ChatMessage(role="user", content=user_input)
        chat_history.append(user_msg)

        # Save updated history
        await ctx.store.set("chat_history", chat_history)

        return InputEvent(input=chat_history)

    @step
    async def handle_llm_input(
        self, ctx: Context, ev: InputEvent
    ) -> ToolCallEvent | StopEvent:
        """Handle LLM input and stream response."""
        chat_history = ev.input

        # Stream the response with tools
        response_stream = await self.llm.astream_chat_with_tools(
            self.tools, chat_history=chat_history
        )

        # Track previous content lengths per block index to support multiple blocks
        # Key: block_index, Value: length processed
        block_progress = {}

        # Stream chunks to output, distinguishing thinking from response
        async for response in response_stream:
            # Check blocks for ThinkingBlock vs TextBlock
            if response.message and response.message.blocks:
                for i, block in enumerate(response.message.blocks):
                    if isinstance(block, ThinkingBlock):
                        content = block.content or ""
                        prev_len = block_progress.get(i, 0)
                        # Only emit the NEW portion (delta)
                        if len(content) > prev_len:
                            delta = content[prev_len:]
                            ctx.write_event_to_stream(ThinkingEvent(delta=delta))
                            block_progress[i] = len(content)
                    elif isinstance(block, TextBlock):
                        text = block.text or ""
                        prev_len = block_progress.get(i, 0)
                        # Only emit the NEW portion (delta)
                        if len(text) > prev_len:
                            delta = text[prev_len:]
                            ctx.write_event_to_stream(StreamEvent(delta=delta))
                            block_progress[i] = len(text)
            elif response.delta:
                # Fallback: use delta directly if no blocks
                ctx.write_event_to_stream(StreamEvent(delta=response.delta))

        # Save assistant message to history
        chat_history = await ctx.store.get("chat_history")

        # Debug prints
        print(f"  💬 Assistant Msg Content: {response.message.content}")
        # Verify if thought signatures or provider specific fields are present
        if response.message.additional_kwargs:
            print(
                f"  🔍 Assistant Msg Args: {response.message.additional_kwargs.keys()}"
            )

        chat_history.append(response.message)

        # Ensure provider_specific_fields are preserved if present (LlamaIndex specific)
        # Some LlamaIndex versions might strip this during serialization if not careful

        await ctx.store.set("chat_history", chat_history)

        # Check for tool calls
        tool_calls = self.llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=False
        )

        if not tool_calls:
            sources = await ctx.store.get("sources", default=[])
            # Return full chat history for inspection
            return StopEvent(
                result={
                    "response": response,
                    "sources": sources,
                    "chat_history": chat_history,
                }
            )
        else:
            return ToolCallEvent(tool_calls=tool_calls)

    @step
    async def handle_tool_calls(self, ctx: Context, ev: ToolCallEvent) -> InputEvent:
        """Execute tool calls and return results."""
        tool_calls = ev.tool_calls
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}
        tool_msgs = []
        sources = await ctx.store.get("sources", default=[])

        for tool_call in tool_calls:
            tool = tools_by_name.get(tool_call.tool_name)
            additional_kwargs = {
                "tool_call_id": tool_call.tool_id,
                "name": tool_call.tool_name,
            }

            if not tool:
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=f"Tool {tool_call.tool_name} does not exist",
                        additional_kwargs=additional_kwargs,
                    )
                )
                continue

            try:
                tool_output = tool(**tool_call.tool_kwargs)
                sources.append(tool_output)
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=tool_output.content,
                        additional_kwargs=additional_kwargs,
                    )
                )
                print(
                    f"  🔧 {tool_call.tool_name}({tool_call.tool_kwargs}) = {tool_output.content}"
                )
            except Exception as e:
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=f"Error in tool call: {e}",
                        additional_kwargs=additional_kwargs,
                    )
                )

        # Update chat history with tool messages
        chat_history = await ctx.store.get("chat_history")
        for msg in tool_msgs:
            chat_history.append(msg)

        await ctx.store.set("sources", sources)
        await ctx.store.set("chat_history", chat_history)

        return InputEvent(input=chat_history)


# --- Define Tools ---
def add(x: int, y: int) -> int:
    """Add two numbers together."""
    return x + y


def multiply(x: int, y: int) -> int:
    """Multiply two numbers together."""
    return x * y


def check_flight(flight: str) -> str:
    """Check the status of a flight."""
    return json.dumps({"flight": flight, "status": "delayed", "departure": "12 PM"})


# --- Main ---
async def main():
    print("\n" + "=" * 60)
    print("LlamaIndex Function Calling Agent + Gemini 3 Pro")
    print("(Using Simple List History)")
    print("=" * 60 + "\n")

    # Create tools
    tools = [
        FunctionTool.from_defaults(add),
        FunctionTool.from_defaults(multiply),
        FunctionTool.from_defaults(check_flight),
    ]

    # Create agent
    llm = GoogleGenAI(
        model="gemini-3-pro-preview",
        vertexai_config={"project": project_id, "location": "global"},
        generation_config=types.GenerateContentConfig(
            temperature=1.0,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.LOW,
                include_thoughts=True,  # Disabled to avoid thought_signature issues
            ),
        ),
    )

    agent = FunctionCallingAgent(
        llm=llm,
        tools=tools,
        timeout=120,
        verbose=True,
    )

    # Test 1: Simple greeting - demonstrates ThinkingEvent vs StreamEvent
    print("📩 User: Hello!")
    print("-" * 40)
    handler = agent.run(input="Hello!")
    async for event in handler.stream_events():
        if isinstance(event, ThinkingEvent):
            # Print thinking in dim/gray style
            print(f"\033[90m🧠 {event.delta}\033[0m", end="", flush=True)
        elif isinstance(event, StreamEvent):
            # Print response in normal style
            print(f"💬 {event.delta}", end="", flush=True)
    result = await handler
    print(f"\n📊 Final: {result['response'].message.content}\n")

    # Test 2: Math with tool calls
    print("📩 User: What is (2123 + 2321) * 312?")
    print("-" * 40)
    handler = agent.run(input="What is (2123 + 2321) * 312?")
    async for event in handler.stream_events():
        if isinstance(event, ThinkingEvent):
            print(f"\033[90m🧠 {event.delta}\033[0m", end="", flush=True)
        elif isinstance(event, StreamEvent):
            print(f"💬 {event.delta}", end="", flush=True)
    result = await handler
    print(f"\n📊 Final: {result['response'].message.content}\n")

    # Test 3: Flight check
    print("📩 User: Check flight AA100")
    print("-" * 40)
    handler = agent.run(input="Check flight AA100")
    async for event in handler.stream_events():
        if isinstance(event, ThinkingEvent):
            print(f"\033[90m🧠 {event.delta}\033[0m", end="", flush=True)
        elif isinstance(event, StreamEvent):
            print(f"💬 {event.delta}", end="", flush=True)
    result = await handler
    print(f"\n📊 Final: {result['response'].message.content}\n")

    # Save chat history to JSON
    if "chat_history" in result:
        save_history_to_json(result["chat_history"], "messages_agent_output.json")
        print("💾 Chat history saved to messages_agent_output.json")

    # Flush Langfuse
    langfuse.flush()
    print("✅ Done! Traces sent to Langfuse.")


def save_history_to_json(chat_history: List[ChatMessage], filename: str):
    """Save chat history to a JSON file in a readable format."""
    output = []
    print(f"DEBUG: Saving {len(chat_history)} messages to {filename}")

    for i, msg in enumerate(chat_history):
        # Base message dict
        msg_dict = {
            "role": msg.role,
        }

        # Handle blocks to separate reasoning (ThinkingBlock) from content (TextBlock)
        tool_calls = []
        if msg.blocks:
            reasoning_parts = []
            content_parts = []
            for block in msg.blocks:
                if isinstance(block, ThinkingBlock):
                    reasoning_parts.append(block.content)
                elif isinstance(block, TextBlock):
                    content_parts.append(block.text)
                elif isinstance(block, ToolCallBlock):
                    # Extract tool call info
                    tc = {
                        "id": block.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": block.tool_name,
                            "arguments": (
                                json.dumps(block.tool_kwargs)
                                if isinstance(block.tool_kwargs, dict)
                                else str(block.tool_kwargs)
                            ),
                        },
                    }
                    tool_calls.append(tc)

            if reasoning_parts:
                msg_dict["reasoning_content"] = "\n\n".join(reasoning_parts)
            if content_parts:
                msg_dict["content"] = "\n\n".join(content_parts)
            if tool_calls:
                msg_dict["tool_calls"] = tool_calls
        else:
            msg_dict["content"] = msg.content
            # Check for tool_calls in additional_kwargs if no blocks
            if msg.additional_kwargs and "tool_calls" in msg.additional_kwargs:
                msg_dict["tool_calls"] = msg.additional_kwargs["tool_calls"]

        # Handle additional kwargs
        if msg.additional_kwargs:
            # Handle tool_call_id for tool messages
            if "tool_call_id" in msg.additional_kwargs:
                msg_dict["tool_call_id"] = msg.additional_kwargs["tool_call_id"]

            # Handle thought_signatures and inject into tool calls if present
            if (
                "thought_signatures" in msg.additional_kwargs
                and "tool_calls" in msg_dict
            ):
                sigs = msg.additional_kwargs["thought_signatures"]
                # Map signatures to tool calls based on index logic (usually first one has it)
                # But here sigs list might correspond to blocks, or just available sigs.
                # LlamaIndex google genai utils puts them effectively 1:1 with blocks or just the needed ones.
                # Let's verify: In messages_invoke.json, it's inside provider_specific_fields.

                # Check if we have parallel calls
                tcs = msg_dict["tool_calls"]

                # Try to distribute signatures
                # Note: sigs might contain "None" strings or actual signatures.
                # We will add them to provider_specific_fields of the tool call.

                # Heuristic: If we have sigs, try to attach.
                # The sigs list usually aligns with blocks that are tool calls or thinking.
                # Since we don't know exact mapping here easily without re-implementing utils logic,
                # we will attach the relevant signature if we can find it.
                # OR just dump thought_signatures at message level as before (safer)
                # USER asked for "like messages_invoke.json" where it is inside tool_call.

                # If we have 1 tool call and 1 valid signature in the list, put it there.

                valid_sigs = [s for s in sigs if s and str(s) != "None"]

                if valid_sigs and tcs:
                    # Provide first valid signature to first tool call (standard Gemini rule)
                    # Convert bytes to base64 string
                    sig_val = valid_sigs[0]
                    if isinstance(sig_val, bytes):
                        sig_str = base64.b64encode(sig_val).decode("utf-8")
                    else:
                        sig_str = str(sig_val)

                    if "provider_specific_fields" not in tcs[0]:
                        tcs[0]["provider_specific_fields"] = {}
                    tcs[0]["provider_specific_fields"]["thought_signature"] = sig_str

            # Keep top-level thought_signatures for completeness if needed, or remove if user wants strict format.
            # I will keep them but decoded, just in case.
            if "thought_signatures" in msg.additional_kwargs:
                sigs = msg.additional_kwargs["thought_signatures"]
                decoded_sigs = []
                for s in sigs:
                    if isinstance(s, bytes):
                        decoded_sigs.append(base64.b64encode(s).decode("utf-8"))
                    else:
                        decoded_sigs.append(str(s))
                msg_dict["thought_signatures"] = decoded_sigs

            # Handle tool_call_id for tool messages
            if "tool_call_id" in msg.additional_kwargs:
                msg_dict["tool_call_id"] = msg.additional_kwargs["tool_call_id"]

            # Handle thought_signatures
            if "thought_signatures" in msg.additional_kwargs:
                # Signatures are bytes, need to decode for JSON
                sigs = msg.additional_kwargs["thought_signatures"]
                decoded_sigs = []
                for s in sigs:
                    if isinstance(s, bytes):
                        # Use Base64 for readable/safe output
                        decoded_sigs.append(base64.b64encode(s).decode("utf-8"))
                    else:
                        decoded_sigs.append(str(s))
                msg_dict["thought_signatures"] = decoded_sigs

        output.append({"index": i, "message": msg_dict})

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    asyncio.run(main())
