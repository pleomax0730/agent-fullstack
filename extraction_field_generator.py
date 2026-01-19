"""
Extraction Field Generator - Backend Service

Integrated with the chat streaming infrastructure for:
- Streaming structured output
- Thinking/reasoning display
- Multi-turn conversation
"""

import base64
import json
import os
from enum import Enum
from pathlib import Path
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
from google.genai import types
from llama_index.core.base.llms.types import (
    ChatMessage,
    DocumentBlock,
    ImageBlock,
    MessageRole,
    TextBlock,
)
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.llms.google_genai import GoogleGenAI

load_dotenv()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

CREDENTIALS_FILE = "service_account.json"
LLM_MODEL = "gemini-3-pro-preview"

SYSTEM_INSTRUCTION = """You are an expert data extraction schema designer.
Your task is to analyze documents (PDFs or images) and suggest extraction fields.

When analyzing a document:
1. Identify key data points that could be extracted
2. Suggest appropriate field names (snake_case)
3. Determine the correct data type (string, int, float)
4. Provide clear descriptions for each field

When the user provides feedback:
- Add new fields they request
- Remove fields they don't need
- Modify field names, types, or descriptions as requested
- Keep existing fields unless explicitly asked to change them

Always respond with a complete updated schema."""


def load_credentials() -> dict:
    """Load VertexAI credentials from file."""
    if not os.path.exists(CREDENTIALS_FILE):
        raise FileNotFoundError(f"Credentials file '{CREDENTIALS_FILE}' not found.")
    with open(CREDENTIALS_FILE, "r") as f:
        creds = json.load(f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_FILE
    return creds


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------


class FieldType(str, Enum):
    """Supported field types for extraction."""

    STRING = "string"
    INT = "int"
    FLOAT = "float"


class ExtractionField(BaseModel):
    """A single extraction field definition."""

    field_name: str = Field(description="Name of the field to extract (snake_case)")
    type: FieldType = Field(description="Data type of the field")
    description: str = Field(description="Description of what this field represents")


class ExtractionSchema(BaseModel):
    """Complete extraction schema with multiple fields."""

    fields: list[ExtractionField] = Field(
        description="List of fields to extract from the document"
    )
    # reasoning: str = Field(
    #    description="Brief explanation of why these fields were identified"
    # )


# -----------------------------------------------------------------------------
# Document Block Factory
# -----------------------------------------------------------------------------


class DocumentBlockFactory:
    """Creates LlamaIndex blocks from various input sources."""

    IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
    DOC_MIME_TYPES = {"application/pdf"}

    @classmethod
    def from_data_url(cls, data_url: str, mime_type: str) -> ImageBlock | DocumentBlock:
        """Create block from data URL (base64 encoded)."""
        if data_url.startswith("data:"):
            _, data = data_url.split(",", 1)
            raw_bytes = base64.b64decode(data)
        else:
            raw_bytes = base64.b64decode(data_url)

        if mime_type in cls.IMAGE_MIME_TYPES:
            return ImageBlock(image=raw_bytes, image_mimetype=mime_type)
        elif mime_type in cls.DOC_MIME_TYPES:
            return DocumentBlock(data=raw_bytes, mimetype=mime_type)
        else:
            raise ValueError(f"Unsupported mime type: {mime_type}")

    @classmethod
    def from_file(cls, file_path: str | Path) -> ImageBlock | DocumentBlock:
        """Create block from file path."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
            return ImageBlock(path=path)
        elif suffix == ".pdf":
            return DocumentBlock(path=path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    @classmethod
    def is_supported_mime(cls, mime_type: str) -> bool:
        """Check if mime type is supported."""
        return mime_type in (cls.IMAGE_MIME_TYPES | cls.DOC_MIME_TYPES)


# -----------------------------------------------------------------------------
# Extraction LLM Client (Singleton)
# -----------------------------------------------------------------------------

_extraction_llm: Optional[GoogleGenAI] = None


def get_extraction_llm() -> GoogleGenAI:
    """Get or create extraction-specific LLM instance."""
    global _extraction_llm
    if _extraction_llm is None:
        creds = load_credentials()
        _extraction_llm = GoogleGenAI(
            model=LLM_MODEL,
            vertexai_config={"project": creds["project_id"], "location": "global"},
            generation_config=types.GenerateContentConfig(
                temperature=1.0,
                system_instruction=SYSTEM_INSTRUCTION,
                thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
            ),
        )
    return _extraction_llm


# -----------------------------------------------------------------------------
# Extraction Session
# -----------------------------------------------------------------------------


class ExtractionSession:
    """Manages a single extraction field generation session."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: list[ChatMessage] = []
        self.current_schema: Optional[ExtractionSchema] = None
        self.document_block: Optional[ImageBlock | DocumentBlock] = None

    def set_document(self, block: ImageBlock | DocumentBlock) -> None:
        """Set the document to analyze."""
        self.document_block = block

    def add_user_message_with_document(self, text: str) -> None:
        """Add user message with document attachment."""
        blocks = []
        if self.document_block:
            blocks.append(self.document_block)
        blocks.append(TextBlock(text=text))

        self.messages.append(ChatMessage(role=MessageRole.USER, blocks=blocks))

    def add_user_message(self, text: str) -> None:
        """Add text-only user message."""
        self.messages.append(
            ChatMessage(role=MessageRole.USER, blocks=[TextBlock(text=text)])
        )

    def add_assistant_response(self, schema: ExtractionSchema) -> None:
        """Add assistant response with schema."""
        self.current_schema = schema
        self.messages.append(
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=schema.model_dump_json(),
            )
        )

    def get_messages(self) -> list[ChatMessage]:
        """Get all messages for LLM context."""
        return self.messages

    def to_dict(self) -> dict:
        """Serialize session state."""
        return {
            "session_id": self.session_id,
            "schema": self.current_schema.model_dump() if self.current_schema else None,
            "message_count": len(self.messages),
        }


# -----------------------------------------------------------------------------
# Session Manager
# -----------------------------------------------------------------------------


class SessionManager:
    """Manages multiple extraction sessions."""

    def __init__(self):
        self._sessions: dict[str, ExtractionSession] = {}

    def create_session(self, session_id: str) -> ExtractionSession:
        """Create a new session."""
        session = ExtractionSession(session_id)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[ExtractionSession]:
        """Get existing session."""
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str) -> ExtractionSession:
        """Get existing session or create new one."""
        if session_id not in self._sessions:
            return self.create_session(session_id)
        return self._sessions[session_id]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False


# Global session manager
session_manager = SessionManager()


# -----------------------------------------------------------------------------
# Streaming Extraction Generator
# -----------------------------------------------------------------------------


async def stream_extraction_response(
    session: ExtractionSession,
    is_initial: bool = False,
) -> AsyncGenerator[tuple[str, dict | ExtractionSchema | None], None]:
    """
    Stream extraction response with thinking and structured output.

    Yields tuples of (event_type, data):
    - ("reasoning_start", None)
    - ("reasoning_delta", {"text": "..."})
    - ("reasoning_end", None)
    - ("schema_start", None)
    - ("schema_delta", {"partial": {...}})
    - ("schema_end", ExtractionSchema)
    """
    llm = get_extraction_llm()
    sllm = llm.as_structured_llm(ExtractionSchema)

    # Stream the response
    stream = sllm.stream_chat(session.get_messages())

    reasoning_started = False
    schema_started = False
    final_schema: Optional[ExtractionSchema] = None

    for chunk in stream:
        # Check for thinking/reasoning in raw response
        if hasattr(chunk, "raw") and chunk.raw:
            raw = chunk.raw
            if isinstance(raw, dict):
                parts = raw.get("content", {}).get("parts", [])
                for part in parts:
                    if isinstance(part, dict) and part.get("thought"):
                        text = part.get("text", "")
                        if text:
                            if not reasoning_started:
                                yield ("reasoning_start", None)
                                reasoning_started = True
                            yield ("reasoning_delta", {"text": text})

        # Check for structured output
        if chunk.raw and isinstance(chunk.raw, ExtractionSchema):
            if reasoning_started and not schema_started:
                yield ("reasoning_end", None)
                reasoning_started = False

            if not schema_started:
                yield ("schema_start", None)
                schema_started = True

            yield ("schema_delta", {"partial": chunk.raw.model_dump()})
            final_schema = chunk.raw

    # End any open sections
    if reasoning_started:
        yield ("reasoning_end", None)

    if schema_started and final_schema:
        yield ("schema_end", final_schema)
        session.add_assistant_response(final_schema)
