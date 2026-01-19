"""
Test async streaming for structured LLM using LangChain.

Tests both with and without include_thoughts to verify compatibility.
Uses with_structured_output with method='json_schema' for best streaming support.
"""

import asyncio
import json
import os
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from rich import print

load_dotenv()

SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE", "service_account.json")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")


def load_credentials() -> dict:
    """Load Google Cloud service account credentials from file."""
    abs_path = os.path.abspath(SERVICE_ACCOUNT_FILE)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Service account file not found: {abs_path}")

    with open(abs_path, "r") as f:
        service_account_info = json.load(f)

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = abs_path
    return service_account_info


# --- Pydantic models for structured output ---


class ExtractionField(BaseModel):
    """A single extraction field definition."""

    field_name: str = Field(description="Name of the field to extract (snake_case)")
    field_type: str = Field(description="Data type: string, int, or float")
    description: str = Field(description="Description of what this field represents")


class ExtractionSchema(BaseModel):
    """Complete extraction schema with multiple fields."""

    fields: List[ExtractionField] = Field(description="List of fields to extract")


async def test_async_stream_structured(include_thoughts: bool = False):
    """Test async streaming with structured output."""
    print(
        f"\n[bold green]═══ Async Streaming Structured Output (include_thoughts={include_thoughts}) ═══[/bold green]\n"
    )

    creds = load_credentials()

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        vertexai=True,
        project=creds["project_id"],
        location="global",
        include_thoughts=include_thoughts,
        temperature=1.0,
    )

    # Use with_structured_output with json_schema method for better streaming
    structured_llm = llm.with_structured_output(
        ExtractionSchema,
        method="json_schema",  # Use json_schema for streaming support
    )

    messages = [
        HumanMessage(
            content="Suggest extraction fields for an invoice document. Include fields for invoice number, date, total amount, and vendor name."
        )
    ]

    print("[DEBUG] Starting astream...")
    chunk_count = 0
    final_result = None

    try:
        async for chunk in structured_llm.astream(messages):
            chunk_count += 1
            print(f"[DEBUG] Chunk #{chunk_count}")
            print(f"  Type: {type(chunk)}")
            print(f"  Value: {chunk}")
            print("---")
            final_result = chunk
    except Exception as e:
        print(f"[red]Error during streaming: {e}[/red]")
        import traceback

        traceback.print_exc()
        return

    print(f"\n[DEBUG] Total chunks: {chunk_count}")

    if final_result:
        print(f"\n[bold cyan]Final Result:[/bold cyan]")
        print(f"  Type: {type(final_result)}")
        if isinstance(final_result, ExtractionSchema):
            print(f"  Fields: {len(final_result.fields)}")
            for field in final_result.fields:
                print(
                    f"    - {field.field_name} ({field.field_type}): {field.description}"
                )
        else:
            print(f"  Value: {final_result}")


async def test_sync_stream_structured(include_thoughts: bool = False):
    """Test sync streaming with structured output (for comparison)."""
    print(
        f"\n[bold green]═══ Sync Streaming Structured Output (include_thoughts={include_thoughts}) ═══[/bold green]\n"
    )

    creds = load_credentials()

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        vertexai=True,
        project=creds["project_id"],
        location="global",
        include_thoughts=include_thoughts,
        temperature=1.0,
    )

    structured_llm = llm.with_structured_output(
        ExtractionSchema,
        method="json_schema",
    )

    messages = [
        HumanMessage(
            content="Suggest extraction fields for an invoice document. Include fields for invoice number, date, total amount, and vendor name."
        )
    ]

    print("[DEBUG] Starting stream...")
    chunk_count = 0
    final_result = None

    try:
        for chunk in structured_llm.stream(messages):
            chunk_count += 1
            print(f"[DEBUG] Chunk #{chunk_count}")
            print(f"  Type: {type(chunk)}")
            print(f"  Value: {chunk}")
            print("---")
            final_result = chunk
    except Exception as e:
        print(f"[red]Error during streaming: {e}[/red]")
        import traceback

        traceback.print_exc()
        return

    print(f"\n[DEBUG] Total chunks: {chunk_count}")

    if final_result:
        print(f"\n[bold cyan]Final Result:[/bold cyan]")
        if isinstance(final_result, ExtractionSchema):
            print(f"  Fields: {len(final_result.fields)}")


async def test_raw_async_stream(include_thoughts: bool = False):
    """Test raw async streaming without structured output (for comparison)."""
    print(
        f"\n[bold green]═══ Raw Async Streaming (include_thoughts={include_thoughts}) ═══[/bold green]\n"
    )

    creds = load_credentials()

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        vertexai=True,
        project=creds["project_id"],
        location="global",
        include_thoughts=include_thoughts,
        temperature=1.0,
    )

    messages = [
        HumanMessage(
            content="Say hello and pick a random number between 1 and 100. Keep it brief."
        )
    ]

    print("[DEBUG] Starting astream...")
    chunk_count = 0

    try:
        async for chunk in llm.astream(messages):
            chunk_count += 1
            content = chunk.content

            # Handle thinking blocks
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "thinking":
                            print(
                                f"[dim][Thinking] {block.get('thinking', '')[:100]}...[/dim]"
                            )
                        elif block.get("type") == "text":
                            print(f"[Text] {block.get('text', '')}", end="", flush=True)
            elif content:
                print(content, end="", flush=True)
    except Exception as e:
        print(f"\n[red]Error during streaming: {e}[/red]")
        import traceback

        traceback.print_exc()
        return

    print(f"\n\n[DEBUG] Total chunks: {chunk_count}")


async def main():
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    thoughts = "--thoughts" in sys.argv

    if mode == "async" or mode == "all":
        await test_async_stream_structured(include_thoughts=thoughts)

    if mode == "sync" or mode == "all":
        await test_sync_stream_structured(include_thoughts=thoughts)

    if mode == "raw" or mode == "all":
        await test_raw_async_stream(include_thoughts=thoughts)

    if mode not in ["async", "sync", "raw", "all"]:
        print(
            "Usage: python test_langchain_async_streaming.py [async|sync|raw|all] [--thoughts]"
        )


if __name__ == "__main__":
    asyncio.run(main())
