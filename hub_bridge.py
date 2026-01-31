"""
SSE Bridge: agent-fullstack UI <-> hub-agent Socket.IO

Translates Vercel AI SDK useChat requests into hub-agent Socket.IO events
and streams back UI Message Stream Protocol SSE events.

Usage:
    uv run python hub_bridge.py

Env vars:
    HUB_AGENT_URL       (default: http://127.0.0.1:8081)
    INSIGHT_API_URL     (default: http://127.0.0.1:8080)
    TENANT_ID           (default: f17ee335-092b-11f0-9129-4ebe71bd1d01)
    PORT                (default: 8000)
"""

import asyncio
import base64
import hashlib
import json
import os
import uuid
from typing import Any, AsyncGenerator, Optional

import aiohttp
import socketio
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

load_dotenv()

# --- Configuration ---
HUB_AGENT_URL = os.getenv("HUB_AGENT_URL", "http://127.0.0.1:8081")
HUB_AGENT_NAMESPACE = "/api/v1/chat"
INSIGHT_API_URL = os.getenv("INSIGHT_API_URL", "http://127.0.0.1:8080")
TENANT_ID = os.getenv("TENANT_ID", "f17ee335-092b-11f0-9129-4ebe71bd1d01")
PORT = int(os.getenv("PORT", "8000"))

# Global cache: file_hash -> {storage_uri, content_type, filename}
# Persists across requests so we can retrieve GCS URI for previously uploaded files
FILE_CACHE: dict[str, dict] = {}

# Global cache: message_id -> {storage_uri, content_type, filename}
# Some clients may omit the original file part in subsequent requests (to avoid resending base64).
# Persisting by message_id allows us to reconstruct document history items later.
MESSAGE_FILE_CACHE: dict[str, dict] = {}

app = FastAPI(title="Hub Bridge (SSE)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- SSE Event Helpers (UI Message Stream Protocol) ---
def sse_event(data: dict | str) -> str:
    if isinstance(data, dict):
        return f"data: {json.dumps(data)}\n\n"
    return f"data: {data}\n\n"


def create_message_start(message_id: str) -> dict:
    return {"type": "start", "messageId": message_id}


def create_text_start(part_id: str) -> dict:
    return {"type": "text-start", "id": part_id}


def create_text_delta(part_id: str, delta: str) -> dict:
    return {"type": "text-delta", "id": part_id, "delta": delta}


def create_text_end(part_id: str) -> dict:
    return {"type": "text-end", "id": part_id}


def create_tool_input_start(tool_call_id: str, tool_name: str) -> dict:
    return {"type": "tool-input-start", "toolCallId": tool_call_id, "toolName": tool_name}


def create_tool_input_delta(tool_call_id: str, delta: str) -> dict:
    return {"type": "tool-input-delta", "toolCallId": tool_call_id, "inputTextDelta": delta}


def create_tool_input_available(tool_call_id: str, tool_name: str, input_obj: dict) -> dict:
    return {"type": "tool-input-available", "toolCallId": tool_call_id, "toolName": tool_name, "input": input_obj}


def create_tool_output_available(tool_call_id: str, output: Any) -> dict:
    return {"type": "tool-output-available", "toolCallId": tool_call_id, "output": output}


def create_finish_step() -> dict:
    return {"type": "finish-step"}


def create_finish() -> dict:
    return {"type": "finish"}


# --- Request Models (useChat format) ---
class UIMessagePart(BaseModel):
    type: str
    text: Optional[str] = None
    url: Optional[str] = None
    mediaType: Optional[str] = None
    filename: Optional[str] = None
    # Tool invocation fields (from Vercel AI SDK)
    toolCallId: Optional[str] = None
    toolName: Optional[str] = None
    input: Optional[dict] = None
    args: Optional[dict] = None  # alias for input
    output: Optional[Any] = None
    result: Optional[Any] = None  # alias for output
    state: Optional[str] = None  # tool state

    # Pydantic v2 config
    model_config = ConfigDict(extra="allow")


class UIMessage(BaseModel):
    id: str
    role: str
    content: Optional[str] = None
    parts: Optional[list[UIMessagePart]] = None


class ChatRequest(BaseModel):
    messages: list[UIMessage]


# --- File Upload to Insight API ---
async def upload_file_to_insight(
    data_url: str,
    mime_type: str,
    filename: str,
    http_session: aiohttp.ClientSession,
) -> dict | None:
    """
    Upload a data URL file to Insight API.
    Returns {"storage_uri": "gs://...", "content_type": "...", "filename": "..."} on success.
    Uses global FILE_CACHE to avoid re-uploading the same file.
    """
    # Check cache first
    file_hash = compute_file_hash(data_url)
    if file_hash in FILE_CACHE:
        cached = FILE_CACHE[file_hash]
        print(f"[Bridge] File found in cache: {cached.get('storage_uri')}")
        return cached
    
    try:
        # Decode data URL
        if data_url.startswith("data:"):
            header, encoded = data_url.split(",", 1)
        else:
            encoded = data_url
        file_bytes = base64.b64decode(encoded)

        # Prepare multipart upload
        form_data = aiohttp.FormData()
        form_data.add_field(
            "file",
            file_bytes,
            filename=filename or "upload",
            content_type=mime_type,
        )

        headers = {
            "x-tenant-id": TENANT_ID,
            "x-request-id": str(uuid.uuid4()),
        }

        async with http_session.post(
            f"{INSIGHT_API_URL}/files",
            data=form_data,
            headers=headers,
        ) as response:
            if response.status in (200, 201):
                result = await response.json()
                print(f"[Bridge] File uploaded: {result.get('storage_uri')}")
                file_info = {
                    "storage_uri": result.get("storage_uri"),
                    "content_type": result.get("content_type", mime_type),
                    "filename": result.get("filename", filename),
                }
                # Cache for future requests
                FILE_CACHE[file_hash] = file_info
                return file_info
            else:
                error_text = await response.text()
                print(f"[Bridge] File upload failed ({response.status}): {error_text}")
                return None
    except Exception as e:
        print(f"[Bridge] File upload error: {e}")
        return None


def compute_file_hash(data_url: str) -> str:
    """Compute SHA256 of data URL content for deduplication."""
    if data_url.startswith("data:"):
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


# --- Hub Agent Socket.IO Client ---
class HubAgentClient:
    """Async Socket.IO client for hub-agent."""

    def __init__(self):
        self.sio = socketio.AsyncClient()
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.connected = False
        self._setup_handlers()

    def _setup_handlers(self):
        @self.sio.on("connect", namespace=HUB_AGENT_NAMESPACE)
        async def on_connect():
            self.connected = True
            print("[Bridge] Connected to hub-agent")

        @self.sio.on("disconnect", namespace=HUB_AGENT_NAMESPACE)
        async def on_disconnect():
            self.connected = False
            print("[Bridge] Disconnected from hub-agent")

        @self.sio.on("stream_response", namespace=HUB_AGENT_NAMESPACE)
        async def on_stream_response(data):
            await self.event_queue.put(data)

        @self.sio.on("stream_error", namespace=HUB_AGENT_NAMESPACE)
        async def on_stream_error(data):
            await self.event_queue.put({"event": "error", "error": data.get("error", "Unknown error")})

    async def connect(self) -> bool:
        try:
            await asyncio.wait_for(
                self.sio.connect(
                    HUB_AGENT_URL,
                    headers={"x-tenant-id": TENANT_ID},
                    namespaces=[HUB_AGENT_NAMESPACE],
                ),
                timeout=10.0,
            )
            await asyncio.sleep(0.3)
            return self.connected
        except Exception as e:
            print(f"[Bridge] Connection failed: {e}")
            return False

    async def send_chat(self, user_input: str, chat_history: list[dict], active_agent_id: str | None = None):
        payload = {
            "user_input": user_input,
            "chat_history": chat_history,
        }
        if active_agent_id:
            payload["active_agent_id"] = active_agent_id
        await self.sio.emit("chat_stream", payload, namespace=HUB_AGENT_NAMESPACE)

    async def disconnect(self):
        if self.sio.connected:
            await self.sio.disconnect()


# --- Convert UIMessage to hub-agent chat_history format ---
def convert_messages_to_hub_history(messages: list[UIMessage], uploaded_files: dict[str, dict]) -> list[dict]:
    """
    Convert useChat messages to hub-agent chat_history format.
    uploaded_files: {message_id -> file_info} for files that were uploaded this session.
    Also checks FILE_CACHE for files from previous requests.
    
    Follows SOCKETIO_API.md format:
    - Documents: {role, content: "gs://...", type: "document", mime_type}
    - Tool calls: {role: "assistant", content, type, tool_calls: [{id, name, args}]}
    - Tool results: {role: "tool", content, type, tool_call_id}
    """
    history = []

    for msg in messages:
        if msg.role == "user":
            # Check for file/document in uploaded_files (current request)
            file_info = uploaded_files.get(msg.id)

            # Check persistent per-message cache (previous requests)
            if not file_info and msg.id in MESSAGE_FILE_CACHE:
                file_info = MESSAGE_FILE_CACHE[msg.id]
                print(f"[Bridge] Found cached file by message_id: {file_info.get('storage_uri')}")
            
            # Also check message parts for file type and look up in global FILE_CACHE
            if not file_info and msg.parts:
                for part in msg.parts:
                    if part.type == "file" and part.url:
                        file_hash = compute_file_hash(part.url)
                        if file_hash in FILE_CACHE:
                            file_info = FILE_CACHE[file_hash]
                            print(f"[Bridge] Found cached file in history: {file_info.get('storage_uri')}")
                            break
            
            if file_info:
                # Add as document type per API spec
                history.append({
                    "role": "user",
                    "content": file_info["storage_uri"],  # GCS URI as content
                    "type": "document",
                    "mime_type": file_info["content_type"],
                    "filename": file_info.get("filename"),
                })

            # Add text content
            text_content = ""
            if msg.parts:
                for part in msg.parts:
                    if part.type == "text" and part.text:
                        text_content = part.text
                        break
            if not text_content and msg.content:
                text_content = msg.content

            if text_content:
                history.append({
                    "role": "user",
                    "content": text_content,
                    "type": "text",
                })

        elif msg.role == "assistant":
            # Collect tool calls and text content from parts
            tool_calls = []
            text_content = ""
            
            if msg.parts:
                for part in msg.parts:
                    if part.type == "text" and part.text:
                        text_content += part.text
                    # Handle tool invocations (type: "tool-invocation" or "tool-xxx")
                    elif part.type == "tool-invocation" or part.type.startswith("tool-"):
                        tool_name = part.toolName or part.type.replace("tool-", "")
                        tool_call_id = part.toolCallId or f"call_{tool_name}"
                        tool_input = part.input or part.args or {}
                        tool_output = part.output if part.output is not None else part.result
                        
                        # Add tool call
                        tool_calls.append({
                            "id": tool_call_id,
                            "name": tool_name,
                            "args": tool_input if isinstance(tool_input, dict) else {},
                        })
                        
                        # If tool has output, we'll add tool result message after assistant message
                        if tool_output is not None:
                            # Store for later - we need to add assistant msg first, then tool msg
                            if not hasattr(msg, '_tool_results'):
                                msg._tool_results = []
                            msg._tool_results.append({
                                "tool_call_id": tool_call_id,
                                "tool_name": tool_name,
                                "output": tool_output,
                            })
            
            if not text_content and msg.content:
                text_content = msg.content

            # Add assistant message (with tool_calls if any)
            assistant_msg = {
                "role": "assistant",
                "content": text_content,
                "type": "text",
            }
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            history.append(assistant_msg)
            
            # Add tool result messages
            if hasattr(msg, '_tool_results'):
                for tool_result in msg._tool_results:
                    output_str = tool_result["output"]
                    if not isinstance(output_str, str):
                        output_str = json.dumps(output_str, ensure_ascii=False)
                    history.append({
                        "role": "tool",
                        "content": output_str,
                        "type": "text",
                        "tool_call_id": tool_result["tool_call_id"],
                    })

    return history


def normalize_schema_for_frontend(hub_schema: dict) -> dict:
    """
    Convert hub-agent design-agent-schema output to frontend ExtractionSchema format.
    
    Hub format:
        {"name": "...", "task": "...", "fields": [{"name": "...", "display_name": "...", "field_type": "STRING", "description": "..."}]}
    
    Frontend format:
        {"fields": [{"label": "...", "field_name": "...", "value": null, "type": "string", "description": "..."}], "reasoning": "..."}
    """
    fields = []
    for f in hub_schema.get("fields", []):
        fields.append({
            "label": f.get("display_name", f.get("name", "")),
            "field_name": f.get("name", ""),
            "value": None,
            "type": f.get("field_type", "STRING").lower(),
            "description": f.get("description", ""),
        })

    return {
        "fields": fields,
        "reasoning": f"小幫手名稱: {hub_schema.get('name', '')}\n任務: {hub_schema.get('task', '')}",
    }


# --- Main Streaming Handler ---
async def stream_chat_via_hub(
    messages: list[UIMessage],
) -> AsyncGenerator[str, None]:
    """
    Stream chat response by proxying to hub-agent via Socket.IO.
    """
    message_id = str(uuid.uuid4())
    yield sse_event(create_message_start(message_id))

    # Track uploaded files for this session
    uploaded_files: dict[str, dict] = {}

    async with aiohttp.ClientSession() as http_session:
        # Process the latest user message for file uploads
        latest_msg = messages[-1] if messages else None
        if latest_msg and latest_msg.role == "user" and latest_msg.parts:
            for part in latest_msg.parts:
                if part.type == "file" and part.url and part.mediaType:
                    # Upload file to Insight API
                    file_info = await upload_file_to_insight(
                        data_url=part.url,
                        mime_type=part.mediaType,
                        filename=part.filename or "upload",
                        http_session=http_session,
                    )
                    if file_info:
                        uploaded_files[latest_msg.id] = file_info
                        # Persist by message_id for future requests (clients may omit file parts later)
                        MESSAGE_FILE_CACHE[latest_msg.id] = file_info
                    # Only handle first file (one attachment at a time)
                    break

        # Build hub-agent chat history (excluding the current user text, which goes in user_input)
        history = convert_messages_to_hub_history(messages[:-1], uploaded_files)

        # Add file info for current message if present (as document type per API spec)
        if latest_msg and latest_msg.id in uploaded_files:
            file_info = uploaded_files[latest_msg.id]
            history.append({
                "role": "user",
                "content": file_info["storage_uri"],  # GCS URI as content
                "type": "document",
                "mime_type": file_info["content_type"],
                "filename": file_info.get("filename"),
            })

        # Extract user text from latest message
        user_text = ""
        if latest_msg and latest_msg.parts:
            for part in latest_msg.parts:
                if part.type == "text" and part.text:
                    user_text = part.text
                    break
        if not user_text and latest_msg and latest_msg.content:
            user_text = latest_msg.content

        # If file uploaded without text, just indicate file was uploaded
        # Let hub-agent ask the user what they want to do
        if not user_text and uploaded_files:
            user_text = "（已上傳文件）"

        if not user_text:
            # No input at all
            yield sse_event(create_finish())
            yield "data: [DONE]\n\n"
            return

        # Connect to hub-agent
        hub_client = HubAgentClient()
        if not await hub_client.connect():
            error_part_id = str(uuid.uuid4())
            yield sse_event(create_text_start(error_part_id))
            yield sse_event(create_text_delta(error_part_id, "Error: Could not connect to hub-agent"))
            yield sse_event(create_text_end(error_part_id))
            yield sse_event(create_finish_step())
            yield sse_event(create_finish())
            yield "data: [DONE]\n\n"
            return

        try:
        # Send chat request
            print(f"[Bridge] === SENDING TO HUB-AGENT ===")
            print(f"[Bridge] user_input: {user_text}")
            print(f"[Bridge] chat_history ({len(history)} items):")
            for i, item in enumerate(history):
                content_preview = str(item.get('content', ''))[:100]
                print(f"[Bridge]   [{i}] role={item.get('role')}, type={item.get('type')}, content={content_preview}...")
                if item.get('mime_type'):
                    print(f"[Bridge]       mime_type: {item.get('mime_type')}")
                if item.get('filename'):
                    print(f"[Bridge]       filename: {item.get('filename')}")
                if item.get('tool_calls'):
                    print(f"[Bridge]       tool_calls: {item.get('tool_calls')}")
                if item.get('tool_call_id'):
                    print(f"[Bridge]       tool_call_id: {item.get('tool_call_id')}")
            print(f"[Bridge] ===================================")
            
            await hub_client.send_chat(
                user_input=user_text,
                chat_history=history,
                active_agent_id=None,  # Let orchestrator decide
            )

            # Stream events
            text_part_id = str(uuid.uuid4())
            text_started = False
            has_streamed_text = False  # Track if we already streamed via llm_progress
            current_tool_call_id: str | None = None
            current_tool_name: str | None = None

            while True:
                try:
                    event = await asyncio.wait_for(hub_client.event_queue.get(), timeout=120.0)
                except asyncio.TimeoutError:
                    print("[Bridge] Timeout waiting for hub-agent response")
                    break

                event_type = event.get("event")
                print(f"[Bridge] Hub event: {event_type}")

                if event_type == "llm_progress":
                    delta = event.get("delta", "")
                    if delta:
                        if not text_started:
                            yield sse_event(create_text_start(text_part_id))
                            text_started = True
                        
                        # Split large deltas into smaller chunks for smoother animation
                        chunk_size = 4  # Characters per chunk (smaller = smoother)
                        for i in range(0, len(delta), chunk_size):
                            chunk = delta[i:i + chunk_size]
                            yield sse_event(create_text_delta(text_part_id, chunk))
                            await asyncio.sleep(0.05)  # 50ms between chunks (slower)
                        
                        has_streamed_text = True

                elif event_type == "tool_call":
                    # End any open text part
                    if text_started:
                        yield sse_event(create_text_end(text_part_id))
                        text_started = False
                        text_part_id = str(uuid.uuid4())

                    tool_name = event.get("tool_name", "unknown")
                    tool_kwargs = event.get("tool_kwargs", {})
                    current_tool_call_id = f"call_{tool_name}_{uuid.uuid4().hex[:8]}"
                    current_tool_name = tool_name

                    yield sse_event(create_tool_input_start(current_tool_call_id, tool_name))
                    yield sse_event(create_tool_input_delta(current_tool_call_id, json.dumps(tool_kwargs, ensure_ascii=False)))
                    yield sse_event(create_tool_input_available(current_tool_call_id, tool_name, tool_kwargs))

                elif event_type == "tool_progress":
                    # Just acknowledge, no SSE event needed
                    pass

                elif event_type == "tool_result":
                    tool_name = event.get("tool_name", "unknown")
                    result = event.get("result")
                    error = event.get("error")

                    if current_tool_call_id:
                        output = {"error": error} if error else result
                        yield sse_event(create_tool_output_available(current_tool_call_id, output))

                    # Special handling for design-agent-schema: emit normalized schema as text
                    if tool_name == "design-agent-schema" and result and not error:
                        try:
                            hub_schema = json.loads(result) if isinstance(result, str) else result
                            frontend_schema = normalize_schema_for_frontend(hub_schema)

                            schema_part_id = str(uuid.uuid4())
                            yield sse_event(create_text_start(schema_part_id))
                            yield sse_event(create_text_delta(schema_part_id, json.dumps(frontend_schema, ensure_ascii=False, indent=2)))
                            yield sse_event(create_text_end(schema_part_id))
                        except Exception as e:
                            print(f"[Bridge] Failed to normalize schema: {e}")

                    # Special handling for extract-data: emit extracted entity as text
                    if tool_name == "extract-data" and result and not error:
                        try:
                            extracted = json.loads(result) if isinstance(result, str) else result

                            # Normalize various shapes from insight-api
                            entity = None
                            if isinstance(extracted, list):
                                # Take the first item if it's a list of results
                                first = extracted[0] if extracted else None
                                if isinstance(first, dict):
                                    entity = first.get("entity") or first.get("extracted") or first
                            elif isinstance(extracted, dict):
                                entity = extracted.get("entity") or extracted.get("extracted") or extracted

                            if isinstance(entity, dict) and entity:
                                data_part_id = str(uuid.uuid4())
                                yield sse_event(create_text_start(data_part_id))
                                yield sse_event(create_text_delta(data_part_id, json.dumps(entity, ensure_ascii=False, indent=2)))
                                yield sse_event(create_text_end(data_part_id))
                        except Exception as e:
                            print(f"[Bridge] Failed to emit extracted data: {e}")

                    current_tool_call_id = None
                    current_tool_name = None

                elif event_type == "final_response":
                    # Only emit final_response if we haven't already streamed via llm_progress
                    # (hub-agent sends the same content in both, so we'd get duplicates)
                    response = event.get("response", "")
                    if response and not has_streamed_text:
                        final_part_id = str(uuid.uuid4())
                        yield sse_event(create_text_start(final_part_id))
                        yield sse_event(create_text_delta(final_part_id, response))
                        yield sse_event(create_text_end(final_part_id))
                    break

                elif event_type == "error":
                    error_msg = event.get("error", "Unknown error")
                    error_part_id = str(uuid.uuid4())
                    yield sse_event(create_text_start(error_part_id))
                    yield sse_event(create_text_delta(error_part_id, f"Error: {error_msg}"))
                    yield sse_event(create_text_end(error_part_id))
                    break

            # Close any open text part
            if text_started:
                yield sse_event(create_text_end(text_part_id))

        finally:
            await hub_client.disconnect()

    yield sse_event(create_finish_step())
    yield sse_event(create_finish())
    yield "data: [DONE]\n\n"


# --- API Endpoints ---
@app.post("/api/chat")
async def chat(request: Request):
    """
    SSE endpoint compatible with Vercel AI SDK useChat.
    Proxies to hub-agent via Socket.IO.
    """
    body = await request.json()
    chat_request = ChatRequest(**body)

    return StreamingResponse(
        stream_chat_via_hub(chat_request.messages),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "x-vercel-ai-ui-message-stream": "v1",
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok", "bridge": "hub-agent"}


if __name__ == "__main__":
    import uvicorn
    print(f"[Bridge] Starting on port {PORT}")
    print(f"[Bridge] Hub Agent: {HUB_AGENT_URL}")
    print(f"[Bridge] Insight API: {INSIGHT_API_URL}")
    print(f"[Bridge] Tenant ID: {TENANT_ID}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
