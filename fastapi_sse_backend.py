"""
FastAPI backend with SSE streaming compatible with Vercel AI SDK's useChat hook.

This implements the UI Message Stream Protocol so AI Elements components
(Message, Reasoning, Sources, etc.) work seamlessly with a Python backend.

Protocol Reference:
- Header: x-vercel-ai-ui-message-stream: v1
- Message types: start, text-start/delta/end, reasoning-start/delta/end, 
                 tool-call-start/delta/end, finish-step, finish, [DONE]
"""

import json
import os
import uuid
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from google.genai import types
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool
from llama_index.llms.google_genai import GoogleGenAI
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="AI Elements Compatible Backend")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
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
    return {"flight": flight, "status": "delayed", "departure_time": "12 PM"}


def book_taxi(time: str) -> dict:
    """Book a taxi for pickup.

    Args:
        time: Time to book the taxi (e.g. "10 AM")
    """
    return {"booking_status": "success", "pickup_time": time}


TOOLS = [
    FunctionTool.from_defaults(fn=check_flight),
    FunctionTool.from_defaults(fn=book_taxi),
]
TOOLS_BY_NAME = {t.metadata.name: t for t in TOOLS}


# --- SSE Event Helpers ---
def sse_event(data: dict | str) -> str:
    """Format data as SSE event."""
    if isinstance(data, dict):
        return f"data: {json.dumps(data)}\n\n"
    return f"data: {data}\n\n"


def create_message_start(message_id: str) -> dict:
    """Create message start event."""
    return {"type": "start", "messageId": message_id}


def create_text_start(part_id: str) -> dict:
    """Create text start event."""
    return {"type": "text-start", "id": part_id}


def create_text_delta(part_id: str, delta: str) -> dict:
    """Create text delta event."""
    return {"type": "text-delta", "id": part_id, "delta": delta}


def create_text_end(part_id: str) -> dict:
    """Create text end event."""
    return {"type": "text-end", "id": part_id}


def create_reasoning_start(part_id: str) -> dict:
    """Create reasoning start event."""
    return {"type": "reasoning-start", "id": part_id}


def create_reasoning_delta(part_id: str, delta: str) -> dict:
    """Create reasoning delta event."""
    return {"type": "reasoning-delta", "id": part_id, "delta": delta}


def create_reasoning_end(part_id: str) -> dict:
    """Create reasoning end event."""
    return {"type": "reasoning-end", "id": part_id}


def create_tool_input_start(tool_call_id: str, tool_name: str) -> dict:
    """Create tool input start event."""
    return {"type": "tool-input-start", "toolCallId": tool_call_id, "toolName": tool_name}


def create_tool_input_delta(tool_call_id: str, input_text_delta: str) -> dict:
    """Create tool input delta event."""
    return {"type": "tool-input-delta", "toolCallId": tool_call_id, "inputTextDelta": input_text_delta}


def create_tool_input_available(tool_call_id: str, tool_name: str, input_data: dict) -> dict:
    """Create tool input available event."""
    return {"type": "tool-input-available", "toolCallId": tool_call_id, "toolName": tool_name, "input": input_data}


def create_tool_output(tool_call_id: str, output: dict) -> dict:
    """Create tool output available event."""
    return {"type": "tool-output-available", "toolCallId": tool_call_id, "output": output}


def create_source_url(url: str) -> dict:
    """Create source URL event for citations."""
    return {"type": "source-url", "url": url}


def create_finish_step() -> dict:
    """Create finish step event."""
    return {"type": "finish-step"}


def create_finish() -> dict:
    """Create finish message event."""
    return {"type": "finish"}


# --- Request Models ---
class UIMessagePart(BaseModel):
    type: str
    text: Optional[str] = None
    url: Optional[str] = None
    # Add other part types as needed


class UIMessage(BaseModel):
    id: str
    role: str
    content: Optional[str] = None
    parts: Optional[list[UIMessagePart]] = None


class ChatRequest(BaseModel):
    messages: list[UIMessage]
    model: Optional[str] = None
    webSearch: Optional[bool] = False


def convert_ui_messages_to_chat_messages(ui_messages: list[UIMessage]) -> list[ChatMessage]:
    """Convert UI messages from frontend to LlamaIndex ChatMessages."""
    chat_messages = []
    for msg in ui_messages:
        # Extract text from parts if available
        content = ""
        if msg.parts:
            text_parts = [p.text for p in msg.parts if p.type == "text" and p.text]
            content = "\n".join(text_parts)
        elif msg.content:
            content = msg.content

        if content:
            chat_messages.append(ChatMessage(role=msg.role, content=content))

    return chat_messages


# --- LLM Setup ---
_llm_instance = None


def get_llm() -> GoogleGenAI:
    """Get or create LLM instance."""
    global _llm_instance
    if _llm_instance is None:
        creds = load_credentials()
        _llm_instance = GoogleGenAI(
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
    return _llm_instance


async def stream_chat_response(
    messages: list[ChatMessage],
    use_tools: bool = True,
) -> AsyncGenerator[str, None]:
    """
    Stream chat response in UI Message Stream Protocol format.
    
    This generator yields SSE events compatible with useChat hook.
    """
    llm = get_llm()
    message_id = str(uuid.uuid4())
    
    # Send message start
    yield sse_event(create_message_start(message_id))
    
    max_steps = 5
    for step in range(max_steps):
        # Use async streaming - astream_chat instead of stream_chat
        tools = TOOLS if use_tools else None
        
        # Prepare chat with tools if needed
        if tools:
            chat_kwargs = llm._prepare_chat_with_tools(tools, chat_history=messages)
            response_gen = await llm.astream_chat(**chat_kwargs)
        else:
            response_gen = await llm.astream_chat(messages=messages)
        
        full_response = None
        text_part_id = str(uuid.uuid4())
        reasoning_part_id = str(uuid.uuid4())
        text_started = False
        reasoning_started = False
        current_reasoning = ""
        current_text = ""
        
        # Use async for to iterate over the async generator
        async for chunk in response_gen:
            full_response = chunk
            
            # Helper to get parts from chunk
            # LlamaIndex GoogleGenAI wrapper returns chunk.raw as a dict
            parts = []
            if hasattr(chunk, 'raw') and chunk.raw:
                raw = chunk.raw
                if isinstance(raw, dict):
                    if 'content' in raw and 'parts' in raw['content']:
                        parts = raw['content']['parts']

            # Process parts
            for part in parts:
                if isinstance(part, dict):
                    is_thought = part.get('thought', False)
                    text = part.get('text', '')

                    if is_thought:
                        if text:
                            if not reasoning_started:
                                yield sse_event(create_reasoning_start(reasoning_part_id))
                                reasoning_started = True
                            
                            # Google GenAI streaming sends full delta in each chunk
                            yield sse_event(create_reasoning_delta(reasoning_part_id, text))
                                
                    elif text:
                        if not text_started:
                            if reasoning_started:
                                yield sse_event(create_reasoning_end(reasoning_part_id))
                                reasoning_started = False
                            yield sse_event(create_text_start(text_part_id))
                            text_started = True
                        
                        yield sse_event(create_text_delta(text_part_id, text))
                        current_text = text
        
        if not full_response:
            break
        
        # End reasoning if it's still open
        if reasoning_started and not text_started:
            yield sse_event(create_reasoning_end(reasoning_part_id))
        
        # End text part if started
        if text_started:
            yield sse_event(create_text_end(text_part_id))
        
        # Check for tool calls
        if use_tools:
            tool_calls = llm.get_tool_calls_from_response(
                full_response, error_on_no_tool_call=False
            )
            
            if tool_calls:
                # Add assistant message to history
                messages.append(full_response.message)
                
                # Execute tools
                for tc in tool_calls:
                    tool_call_id = tc.tool_id or str(uuid.uuid4())
                    
                    # Send tool input events
                    yield sse_event(create_tool_input_start(tool_call_id, tc.tool_name))
                    yield sse_event(create_tool_input_delta(tool_call_id, json.dumps(tc.tool_kwargs)))
                    yield sse_event(create_tool_input_available(tool_call_id, tc.tool_name, tc.tool_kwargs))
                    
                    # Execute tool
                    tool = TOOLS_BY_NAME.get(tc.tool_name)
                    if tool:
                        result = tool.call(**tc.tool_kwargs)
                        # ToolOutput is not JSON serializable, extract the content
                        if hasattr(result, 'raw_output'):
                            result_data = result.raw_output
                        elif hasattr(result, 'content'):
                            result_data = result.content
                        else:
                            result_data = str(result)
                        yield sse_event(create_tool_output(tool_call_id, result_data))
                        
                        # Add tool response to history
                        messages.append(
                            ChatMessage(
                                role="tool",
                                content=str(result),
                                additional_kwargs={"tool_call_id": tool_call_id},
                            )
                        )
                
                # Send finish step and continue loop for next LLM call
                yield sse_event(create_finish_step())
                continue
        
        # No tool calls, we're done
        break
    
    # Send final events
    yield sse_event(create_finish_step())
    yield sse_event(create_finish())
    yield "data: [DONE]\n\n"


@app.post("/api/chat")
async def chat(request: Request):
    """
    Chat endpoint compatible with useChat hook from @ai-sdk/react.
    
    Implements the UI Message Stream Protocol for streaming responses.
    """
    body = await request.json()
    chat_request = ChatRequest(**body)
    
    # Convert UI messages to LlamaIndex format
    chat_messages = convert_ui_messages_to_chat_messages(chat_request.messages)
    
    # Create streaming response
    return StreamingResponse(
        stream_chat_response(chat_messages, use_tools=True),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            # Critical header for useChat to recognize the protocol
            "x-vercel-ai-ui-message-stream": "v1",
        },
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

