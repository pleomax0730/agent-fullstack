"""
FastAPI backend for Extraction Field Generator with SSE streaming.

Uses Vercel AI SDK's UI Message Stream Protocol for:
- Streaming structured output
- Thinking/reasoning display
- Multi-turn conversation
"""

import json
import os
import uuid
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from extraction_field_generator import (
    DocumentBlockFactory,
    ExtractionSchema,
    get_extraction_llm,
    session_manager,
)

load_dotenv()

app = FastAPI(title="Extraction Field Generator Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- SSE Event Helpers ---
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


def create_reasoning_start(part_id: str) -> dict:
    return {"type": "reasoning-start", "id": part_id}


def create_reasoning_delta(part_id: str, delta: str) -> dict:
    return {"type": "reasoning-delta", "id": part_id, "delta": delta}


def create_reasoning_end(part_id: str) -> dict:
    return {"type": "reasoning-end", "id": part_id}


def create_finish_step() -> dict:
    return {"type": "finish-step"}


def create_finish() -> dict:
    return {"type": "finish"}


# --- Request Models ---
class UIMessagePart(BaseModel):
    type: str
    text: Optional[str] = None
    url: Optional[str] = None
    mediaType: Optional[str] = None
    filename: Optional[str] = None


class UIMessage(BaseModel):
    id: str
    role: str
    content: Optional[str] = None
    parts: Optional[list[UIMessagePart]] = None


class ChatRequest(BaseModel):
    messages: list[UIMessage]


# --- Session ID extraction from messages ---
def get_session_id_from_messages(messages: list[UIMessage]) -> str:
    """Generate consistent session ID based on first message ID."""
    if messages:
        return f"session-{messages[0].id}"
    return f"session-{uuid.uuid4()}"


# --- Streaming Extraction ---
async def stream_extraction_chat(
    session_id: str,
    messages: list[UIMessage],
) -> AsyncGenerator[str, None]:
    """Stream extraction response with thinking and structured output."""
    message_id = str(uuid.uuid4())
    yield sse_event(create_message_start(message_id))

    session = session_manager.get_or_create(session_id)

    # Get latest user message
    latest_msg = messages[-1] if messages else None
    if not latest_msg or latest_msg.role != "user":
        yield sse_event(create_finish())
        yield "data: [DONE]\n\n"
        return

    # Process message parts
    file_block = None
    user_text = ""

    if latest_msg.parts:
        for part in latest_msg.parts:
            if part.type == "file" and part.url and part.mediaType:
                if DocumentBlockFactory.is_supported_mime(part.mediaType):
                    file_block = DocumentBlockFactory.from_data_url(
                        part.url, part.mediaType
                    )
            elif part.type == "text" and part.text:
                user_text = part.text

    # Build user message
    print(f"[DEBUG] file_block: {file_block}")
    print(f"[DEBUG] user_text: {user_text}")

    if file_block:
        session.set_document(file_block)
        if not user_text:
            user_text = "Analyze this document and suggest extraction fields."
        session.add_user_message_with_document(user_text)
    elif user_text:
        session.add_user_message(user_text)
    else:
        print("[DEBUG] No file_block and no user_text, returning early")
        yield sse_event(create_finish())
        yield "data: [DONE]\n\n"
        return

    print(f"[DEBUG] Session messages count: {len(session.get_messages())}")

    # Stream LLM response
    import asyncio

    llm = get_extraction_llm()
    sllm = llm.as_structured_llm(ExtractionSchema)

    reasoning_part_id = str(uuid.uuid4())
    text_part_id = str(uuid.uuid4())
    reasoning_started = False
    text_started = False
    final_schema: Optional[ExtractionSchema] = None

    try:
        print("[DEBUG] Starting astream_chat...")
        stream = await sllm.astream_chat(session.get_messages())
        print(f"[DEBUG] stream type: {type(stream)}")

        print("[DEBUG] Entering async for loop...")
        chunk_count = 0

        # Use anext with timeout to avoid hanging forever
        stream_iter = stream.__aiter__()
        timeout_seconds = 120  # 2 minutes timeout for LLM response

        while True:
            try:
                # Wait for next chunk with timeout
                chunk = await asyncio.wait_for(
                    stream_iter.__anext__(), timeout=timeout_seconds
                )
                chunk_count += 1
                print(f"[DEBUG] Got chunk #{chunk_count}")
                # Debug: print chunk structure
                print(f"[DEBUG] chunk type: {type(chunk)}")
                print(f"[DEBUG] chunk: {chunk}")
                if hasattr(chunk, "raw"):
                    print(f"[DEBUG] chunk.raw type: {type(chunk.raw)}")
                    print(f"[DEBUG] chunk.raw: {chunk.raw}")
                if hasattr(chunk, "message"):
                    print(f"[DEBUG] chunk.message: {chunk.message}")
                if hasattr(chunk, "delta"):
                    print(f"[DEBUG] chunk.delta: {chunk.delta}")
                print("---")

                if hasattr(chunk, "raw") and chunk.raw:
                    raw = chunk.raw

                    # Handle thinking parts
                    if isinstance(raw, dict):
                        parts = raw.get("content", {}).get("parts", [])
                        for part in parts:
                            if isinstance(part, dict) and part.get("thought"):
                                text = part.get("text", "")
                                if text:
                                    if not reasoning_started:
                                        yield sse_event(
                                            create_reasoning_start(reasoning_part_id)
                                        )
                                        reasoning_started = True
                                    yield sse_event(
                                        create_reasoning_delta(reasoning_part_id, text)
                                    )

                    # Handle structured output
                    elif isinstance(raw, ExtractionSchema):
                        if reasoning_started and not text_started:
                            yield sse_event(create_reasoning_end(reasoning_part_id))
                            reasoning_started = False

                        if not text_started:
                            yield sse_event(create_text_start(text_part_id))
                            text_started = True

                        schema_json = raw.model_dump_json(indent=2)
                        yield sse_event(create_text_delta(text_part_id, schema_json))
                        final_schema = raw

            except StopAsyncIteration:
                print("[DEBUG] Stream iteration complete")
                break
            except asyncio.TimeoutError:
                print(f"[DEBUG] Timeout after {timeout_seconds}s waiting for LLM response")
                error_part_id = str(uuid.uuid4())
                yield sse_event(create_text_start(error_part_id))
                yield sse_event(
                    create_text_delta(
                        error_part_id,
                        f"Error: LLM response timeout after {timeout_seconds} seconds. The model may be taking too long to process the request.",
                    )
                )
                yield sse_event(create_text_end(error_part_id))
                break

        if reasoning_started:
            yield sse_event(create_reasoning_end(reasoning_part_id))

        if text_started:
            yield sse_event(create_text_end(text_part_id))

        if final_schema:
            session.add_assistant_response(final_schema)

    except Exception as e:
        import traceback

        print(f"[DEBUG] Exception: {e}")
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        error_part_id = str(uuid.uuid4())
        yield sse_event(create_text_start(error_part_id))
        yield sse_event(create_text_delta(error_part_id, f"Error: {str(e)}"))
        yield sse_event(create_text_end(error_part_id))

    yield sse_event(create_finish_step())
    yield sse_event(create_finish())
    yield "data: [DONE]\n\n"


# --- API Endpoints ---


@app.post("/api/chat")
async def chat(request: Request):
    """
    Extraction field generator chat endpoint.

    - Upload PDF/image: generates extraction field suggestions
    - Send text: refines the schema based on feedback
    - Streams thinking + structured output via SSE
    """
    body = await request.json()
    chat_request = ChatRequest(**body)

    session_id = get_session_id_from_messages(chat_request.messages)

    return StreamingResponse(
        stream_extraction_chat(session_id, chat_request.messages),
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
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
