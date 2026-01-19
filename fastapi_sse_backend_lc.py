"""
FastAPI backend for Extraction Field Generator with SSE streaming (LangChain Version).

Uses Vercel AI SDK's UI Message Stream Protocol for:
- Streaming structured output
- Thinking/reasoning display (via LangChain)
- Multi-turn conversation
"""

import json
import os
import uuid
import asyncio
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import BaseMessage

from extraction_field_generator_lc import (
    ContentFactory,
    ExtractionSchema,
    get_extraction_llm,
    session_manager,
)

load_dotenv()

app = FastAPI(title="Extraction Field Generator Backend (LangChain)")

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
    media_block = None
    user_text = ""

    if latest_msg.parts:
        for part in latest_msg.parts:
            if part.type == "file" and part.url and part.mediaType:
                if ContentFactory.is_supported_mime(part.mediaType):
                    media_block = ContentFactory.from_data_url(part.url, part.mediaType)
            elif part.type == "text" and part.text:
                user_text = part.text

    # Build user message
    print(f"[DEBUG] media_block present: {bool(media_block)}")
    print(f"[DEBUG] user_text: {user_text}")

    if media_block:
        session.set_media(media_block)
        if not user_text:
            user_text = "Analyze this document and suggest extraction fields."
        session.add_user_message_with_media(user_text)
    elif user_text:
        session.add_user_message(user_text)
    else:
        print("[DEBUG] No media and no user_text, returning early")
        yield sse_event(create_finish())
        yield "data: [DONE]\n\n"
        return

    print(f"[DEBUG] Session messages count: {len(session.get_messages())}")

    # Initialize LangChain LLM with structured output + streaming
    # We enable include_thoughts=True by default for LangChain as it supports it well
    llm = get_extraction_llm(include_thoughts=True)

    # Configure structured output with streaming support (method='json_schema')
    structured_llm = llm.with_structured_output(
        ExtractionSchema,
        method="json_schema",
        include_raw=False,  # We want the Pydantic object directly
    )

    reasoning_part_id = str(uuid.uuid4())
    text_part_id = str(uuid.uuid4())
    reasoning_started = False
    text_started = False
    final_schema: Optional[ExtractionSchema] = None

    try:
        print("[DEBUG] Starting astream...")
        # Note: structured_llm.astream yields chunks of extraction schema (partial or complete)
        # However, to get thinking blocks, we might need to access the raw response.
        # But 'with_structured_output' typically hides raw chunks unless include_raw=True.
        # If we use include_raw=True, it yields (raw_output, parsed_output).
        # Let's verify if ChatGoogleGenerativeAI yields thinking chunks in raw output.

        # Actually, to get streaming thinking + structured output, standard .astream() on a structured LLM
        # might mask the thinking parts if they are not part of the schema.
        #
        # A better approach for thinking + structured is to parse chunks manually from the raw stream.
        # But let's try strict json_schema first. If we need thinking, we might need 'include_raw=True'
        # or separate the logic.

        # Strategy: Use include_raw=True to access the raw chunks (which contain thinking)
        # and the parsed schema.
        structured_with_raw = llm.with_structured_output(
            ExtractionSchema, method="json_schema", include_raw=True
        )

        chunk_count = 0
        async for chunk in structured_with_raw.astream(session.get_messages()):
            chunk_count += 1
            # chunk is dict with 'raw', 'parsed', 'parsing_error'

            # Access raw message chunk to get thinking
            raw_msg = chunk.get("raw")

            if raw_msg and isinstance(raw_msg, BaseMessage):
                content = raw_msg.content
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "thinking":
                                text = block.get("thinking", "")
                                if text:
                                    if not reasoning_started:
                                        yield sse_event(
                                            create_reasoning_start(reasoning_part_id)
                                        )
                                        reasoning_started = True
                                    yield sse_event(
                                        create_reasoning_delta(reasoning_part_id, text)
                                    )

            # Helper to extract thinking if content is simple text but has special handling (unlikely for Gemini)

            # Handle structured output
            parsed = chunk.get("parsed")
            if parsed and isinstance(parsed, ExtractionSchema):
                if reasoning_started and not text_started:
                    yield sse_event(create_reasoning_end(reasoning_part_id))
                    reasoning_started = False

                if not text_started:
                    yield sse_event(create_text_start(text_part_id))
                    text_started = True

                # Generate JSON delta
                # Since we get full object updates usually, we might send the full JSON repeatedly
                # or try to diff it. For simplicity, we send the full JSON as a delta replacement
                # (Frontend needs to handle receiving full JSONs or we rely on Vercel AI SDK to handle it)
                # Actually, Vercel AI SDK expects deltas. Sending full JSON repeatedly is okay-ish
                # if the frontend overwrites, but huge overhead.
                # Ideally we send the diff, but let's send the full JSON for now.
                schema_json = parsed.model_dump_json(indent=2)

                # Trick: We can just send the full JSON and let the frontend replace it.
                # Or we can compute usage.
                # For this demo, let's send the full JSON in the last step, or send updates.
                # Since chunk.parsed is the Accumulated object, we can just update 'final_schema'.
                final_schema = parsed

                # To make it look like streaming text, we could try to yield the delta.
                # But 'parsed' is the full object.
                # Let's just output the final schema at the end or update periodically?
                # A common pattern is to just store it and yield the final result.
                # But the user wants to see it appear.

                # Sending full json as a delta in one go for now
                # We will send it at the end to avoid flickering if we can't diff.
                pass

        if reasoning_started:
            yield sse_event(create_reasoning_end(reasoning_part_id))

        if final_schema:
            if not text_started:
                yield sse_event(create_text_start(text_part_id))
                text_started = True

            schema_json = final_schema.model_dump_json(indent=2)
            yield sse_event(create_text_delta(text_part_id, schema_json))
            session.add_assistant_response(final_schema)

        if text_started:
            yield sse_event(create_text_end(text_part_id))

    except Exception as e:
        import traceback

        print(f"[DEBUG] Exception: {e}")
        traceback.print_exc()
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
    Extraction field generator chat endpoint (LangChain).
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
    return {"status": "ok", "backend": "langchain"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "127.0.0.1")
    uvicorn.run(app, host=host, port=port)
