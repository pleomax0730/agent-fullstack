"""
Extraction Field Generator - Backend Service (LangChain Version)

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
from typing import AsyncGenerator, Optional, List, Union, Dict, Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

load_dotenv()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

CREDENTIALS_FILE = "service_account.json"
LLM_MODEL = "gemini-3-pro-preview"  # Supports thinking

SYSTEM_INSTRUCTION = """You are an expert data extraction agent specialized in analyzing documents (PDFs or images).
Your task is to identify key information, define a schema, and extract the actual values from the document.

Output Requirements:
1. **Language**: Use Traditional Chinese (Taiwan) for all `label` and `description` fields.
2. **Schema Design**:
   - `label`: A human-readable display name in Traditional Chinese (e.g., "發票號碼", "總金額").
   - `field_name`: A technical key in snake_case (e.g., "invoice_number", "total_amount").
   - `type`: The data type (string, int, float).
   - `value`: The actual value extracted from the document corresponding to this field.
   - `description`: A brief description of the field in Traditional Chinese.

When the user provides feedback:
- Adapt the schema and re-extract values based on their requests.
- Maintain consistency in field naming.
"""


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

    label: str = Field(
        description="Human-readable field name in Traditional Chinese (Taiwan)"
    )
    field_name: str = Field(description="Technical field key (snake_case)")
    value: Union[str, int, float, None] = Field(
        description="The extracted value from the document"
    )
    type: FieldType = Field(description="Data type of the field")
    description: str = Field(description="Description in Traditional Chinese (Taiwan)")


class ExtractionSchema(BaseModel):
    """Complete extraction schema with multiple fields."""

    fields: list[ExtractionField] = Field(
        description="List of fields to extract from the document"
    )
    # Removing reasoning field for now to match current LlamaIndex version
    # reasoning: str = Field(description="Brief explanation of why these fields were identified")


# -----------------------------------------------------------------------------
# LangChain Content Factory
# -----------------------------------------------------------------------------


class ContentFactory:
    """Creates LangChain message content from various input sources."""

    IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
    DOC_MIME_TYPES = {"application/pdf"}

    @classmethod
    def from_data_url(cls, data_url: str, mime_type: str) -> Dict[str, Any]:
        """Create media content block from data URL."""
        if data_url.startswith("data:"):
            _, data = data_url.split(",", 1)
        else:
            data = data_url

        # LangChain with Gemini supports base64 for images
        # For PDFs, typically we might need to convert to images or text,
        # but ChatGoogleGenerativeAI handles some base64 inputs directly in image_url standard
        # or we pass it as a specific media block depending on the provider.
        # For LangChain Google GenAI, we often use the 'image_url' format standard with data protocol.

        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{data}"},
        }

    @classmethod
    def is_supported_mime(cls, mime_type: str) -> bool:
        """Check if mime type is supported."""
        return mime_type in cls.IMAGE_MIME_TYPES or mime_type in cls.DOC_MIME_TYPES


# -----------------------------------------------------------------------------
# Extraction LLM Client
# -----------------------------------------------------------------------------

_extraction_llm: Optional[ChatGoogleGenerativeAI] = None


def get_extraction_llm(include_thoughts: bool = True) -> ChatGoogleGenerativeAI:
    """Get or create extraction-specific LLM instance."""
    global _extraction_llm
    if _extraction_llm is None or _extraction_llm.include_thoughts != include_thoughts:
        creds = load_credentials()
        _extraction_llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            vertexai=True,
            project=creds["project_id"],
            location="global",
            include_thoughts=include_thoughts,
            thinking_level="low",
            temperature=1.0,
        )
    return _extraction_llm


# -----------------------------------------------------------------------------
# Extraction Session
# -----------------------------------------------------------------------------


class ExtractionSession:
    """Manages a single extraction field generation session."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_INSTRUCTION)]
        self.current_schema: Optional[ExtractionSchema] = None
        self.media_block: Optional[Dict[str, Any]] = None

    def set_media(self, block: Dict[str, Any]) -> None:
        """Set the document/image to analyze."""
        self.media_block = block

    def add_user_message_with_media(self, text: str) -> None:
        """Add user message with media attachment."""
        content = []
        if text:
            content.append({"type": "text", "text": text})

        if self.media_block:
            content.append(self.media_block)

        self.messages.append(HumanMessage(content=content))

    def add_user_message(self, text: str) -> None:
        """Add text-only user message."""
        self.messages.append(HumanMessage(content=text))

    def add_assistant_response(self, schema: ExtractionSchema) -> None:
        """Add assistant response with schema."""
        self.current_schema = schema
        # We store the JSON representation as the message content
        self.messages.append(AIMessage(content=schema.model_dump_json()))

    def get_messages(self) -> List[BaseMessage]:
        """Get all messages for LLM context."""
        return self.messages


# -----------------------------------------------------------------------------
# Session Manager
# -----------------------------------------------------------------------------


class SessionManager:
    """Manages multiple extraction sessions."""

    def __init__(self):
        self._sessions: dict[str, ExtractionSession] = {}

    def get_or_create(self, session_id: str) -> ExtractionSession:
        if session_id not in self._sessions:
            self._sessions[session_id] = ExtractionSession(session_id)
        return self._sessions[session_id]


session_manager = SessionManager()
