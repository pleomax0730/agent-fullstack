"""
Test: Minimal required fields for multi-turn tool call conversations.

Tests:
1. Gemini 3 without signature (LiteLLM auto-injects dummy)
2. Gemini 3 without tool_call_id
3. OpenAI without signature
4. OpenAI without tool_call_id
"""

import json
import os
from dotenv import load_dotenv
from rich import print
from litellm import completion

load_dotenv()

SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE", "service_account.json")
GEMINI_MODEL = "vertex_ai/gemini-3-pro-preview"
OPENAI_MODEL = "gpt-4o-mini"


def load_credentials():
    abs_path = os.path.abspath(SERVICE_ACCOUNT_FILE)
    with open(abs_path, "r") as f:
        creds = json.load(f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = abs_path
    return creds


# Tool schema
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "check_flight",
            "description": "Gets the current status of a flight.",
            "parameters": {
                "type": "object",
                "properties": {
                    "flight": {"type": "string", "description": "Flight number"}
                },
                "required": ["flight"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_taxi",
            "description": "Book a taxi for pickup.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {"type": "string", "description": "Pickup time"},
                    "location": {
                        "type": "string",
                        "enum": ["airport", "hotel", "home"],
                    },
                },
                "required": ["time"],
            },
        },
    },
]


def run_test(model: str, messages: list, test_name: str, extra_params: dict = None):
    """Run a single test case."""
    print(f"\n[bold]Testing: {test_name}[/bold]")
    print(f"Model: {model}")

    params = {
        "model": model,
        "messages": messages,
        "tools": TOOLS_SCHEMA,
        "stream": False,
    }
    if extra_params:
        params.update(extra_params)

    try:
        response = completion(**params)
        assistant_msg = response.choices[0].message

        print(f"[green]✓ SUCCESS[/green]")

        if assistant_msg.tool_calls:
            for tc in assistant_msg.tool_calls:
                print(f"  [Tool Call] {tc.function.name}")
        if assistant_msg.content:
            print(f"  [Response] {assistant_msg.content[:100]}...")

        return True

    except Exception as e:
        print(f"[red]✗ FAILED[/red]")
        print(f"  Error: {str(e)[:200]}")
        return False


def test_gemini():
    """Test Gemini 3 with various missing fields."""
    print("\n[bold yellow]" + "=" * 60 + "[/bold yellow]")
    print("[bold yellow]  GEMINI 3 TESTS[/bold yellow]")
    print("[bold yellow]" + "=" * 60 + "[/bold yellow]")

    creds = load_credentials()
    extra = {
        "vertex_location": "global",
        "vertex_project": creds["project_id"],
        "reasoning_effort": "low",
    }

    # Test 1: Without signature (should work - LiteLLM injects dummy)
    messages_no_sig = [
        {"role": "user", "content": "Check flight AA100 and book taxi if delayed."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_001",
                    "type": "function",
                    "function": {
                        "name": "check_flight",
                        "arguments": '{"flight": "AA100"}',
                    },
                    # NO provider_specific_fields
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_001",
            "content": '{"status": "delayed"}',
        },
    ]
    run_test(GEMINI_MODEL, messages_no_sig, "Gemini - No Signature", extra)

    # Test 2: Without tool_call_id
    messages_no_id = [
        {"role": "user", "content": "Check flight AA100 and book taxi if delayed."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    # NO id field
                    "type": "function",
                    "function": {
                        "name": "check_flight",
                        "arguments": '{"flight": "AA100"}',
                    },
                }
            ],
        },
        {"role": "tool", "content": '{"status": "delayed"}'},  # NO tool_call_id
    ]
    run_test(GEMINI_MODEL, messages_no_id, "Gemini - No tool_call_id", extra)

    # Test 3: Without both signature AND tool_call_id
    messages_minimal = [
        {"role": "user", "content": "Check flight AA100 and book taxi if delayed."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "check_flight",
                        "arguments": '{"flight": "AA100"}',
                    },
                }
            ],
        },
        {"role": "tool", "content": '{"status": "delayed"}'},
    ]
    run_test(GEMINI_MODEL, messages_minimal, "Gemini - Minimal (no sig, no id)", extra)


def test_openai():
    """Test OpenAI with various missing fields."""
    print("\n[bold cyan]" + "=" * 60 + "[/bold cyan]")
    print("[bold cyan]  OPENAI TESTS[/bold cyan]")
    print("[bold cyan]" + "=" * 60 + "[/bold cyan]")

    if not os.getenv("OPENAI_API_KEY"):
        print("[red]OPENAI_API_KEY not set, skipping OpenAI tests[/red]")
        return

    # Test 1: Standard (with tool_call_id)
    messages_standard = [
        {"role": "user", "content": "Check flight AA100 and book taxi if delayed."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_openai_001",
                    "type": "function",
                    "function": {
                        "name": "check_flight",
                        "arguments": '{"flight": "AA100"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_openai_001",
            "content": '{"status": "delayed"}',
        },
    ]
    run_test(OPENAI_MODEL, messages_standard, "OpenAI - Standard (with id)")

    # Test 2: Without tool_call_id
    messages_no_id = [
        {"role": "user", "content": "Check flight AA100 and book taxi if delayed."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "check_flight",
                        "arguments": '{"flight": "AA100"}',
                    },
                }
            ],
        },
        {"role": "tool", "content": '{"status": "delayed"}'},
    ]
    run_test(OPENAI_MODEL, messages_no_id, "OpenAI - No tool_call_id")


def test_random_uuid():
    """Test if random UUID can be used as tool_call_id."""
    import uuid

    print("\n[bold magenta]" + "=" * 60 + "[/bold magenta]")
    print("[bold magenta]  RANDOM UUID AS TOOL_CALL_ID TESTS[/bold magenta]")
    print("[bold magenta]" + "=" * 60 + "[/bold magenta]")

    creds = load_credentials()

    # Generate random UUID
    random_id = str(uuid.uuid4())
    print(f"\n[dim]Generated random UUID: {random_id}[/dim]")

    # Test Gemini with random UUID
    messages_gemini = [
        {"role": "user", "content": "Check flight AA100 and book taxi if delayed."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": random_id,  # Random UUID
                    "type": "function",
                    "function": {
                        "name": "check_flight",
                        "arguments": '{"flight": "AA100"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": random_id, "content": '{"status": "delayed"}'},
    ]

    extra_gemini = {
        "vertex_location": "global",
        "vertex_project": creds["project_id"],
        "reasoning_effort": "low",
    }
    run_test(
        GEMINI_MODEL,
        messages_gemini,
        f"Gemini - Random UUID: {random_id[:8]}...",
        extra_gemini,
    )

    # Test OpenAI with random UUID
    if os.getenv("OPENAI_API_KEY"):
        random_id_2 = str(uuid.uuid4())
        print(f"\n[dim]Generated random UUID: {random_id_2}[/dim]")

        messages_openai = [
            {"role": "user", "content": "Check flight AA100 and book taxi if delayed."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": random_id_2,  # Random UUID
                        "type": "function",
                        "function": {
                            "name": "check_flight",
                            "arguments": '{"flight": "AA100"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": random_id_2,
                "content": '{"status": "delayed"}',
            },
        ]
        run_test(
            OPENAI_MODEL, messages_openai, f"OpenAI - Random UUID: {random_id_2[:8]}..."
        )


if __name__ == "__main__":
    test_gemini()
    test_openai()
    test_random_uuid()

    print("\n[bold]" + "=" * 60 + "[/bold]")
    print("[bold]  SUMMARY[/bold]")
    print("[bold]" + "=" * 60 + "[/bold]")
    print(
        """
    Key findings:
    - Signature: LiteLLM auto-injects dummy for Gemini 3
    - tool_call_id: Required, but can be random UUID
    """
    )
