"""
Multi-turn Benchmark: Compare Gemini 3 models and reasoning efforts across a conversation.

Scenario:
1. User: Check flight AA100.
2. Tool: flight is delayed (12 PM instead of 10 AM).
3. Assistant: Reports delay and waits.
4. User: If it's delayed that much, I need a taxi. Book it 2 hours before the new time.
5. Assistant: Should reason and call book_taxi for 10 AM.
"""

import json
import os
import time
from typing import List, Dict
from dotenv import load_dotenv
from rich import print
from rich.table import Table
from litellm import completion

load_dotenv()

SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE", "service_account.json")


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
                    "time": {
                        "type": "string",
                        "description": "Pickup time (e.g. 10 AM)",
                    },
                    "location": {
                        "type": "string",
                        "enum": ["airport", "hotel", "home"],
                        "default": "home",
                        "description": "Pickup location (defaults to home if not specified)",
                    },
                },
                "required": ["time"],
            },
        },
    },
]


def run_multiturn_test(model: str, effort: str, creds: dict):
    """Executes a 3-turn interactive conversation and measures time."""
    messages = []
    timings = []

    # --- TURN 1: Check Flight ---
    messages.append(
        {"role": "user", "content": "Hi, please check the status of flight AA100."}
    )

    start = time.time()
    try:
        # Step 1.1: Assistant calls check_flight
        response = completion(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            reasoning_effort=effort,
            vertex_location="global",
            vertex_project=creds["project_id"],
        )
        msg1 = response.choices[0].message
        messages.append(msg1)

        # Step 1.2: Mock Tool Result
        if msg1.tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": msg1.tool_calls[0].id,
                    "content": json.dumps(
                        {
                            "flight": "AA100",
                            "status": "delayed",
                            "new_departure": "12 PM",
                        }
                    ),
                }
            )

            # Step 1.3: Assistant reports status
            response = completion(
                model=model,
                messages=messages,
                tools=TOOLS_SCHEMA,
                reasoning_effort=effort,
                vertex_location="global",
                vertex_project=creds["project_id"],
            )
            messages.append(response.choices[0].message)

        t1 = time.time() - start
        timings.append(t1)

        # --- TURN 2: Reasoning & Booking ---
        messages.append(
            {
                "role": "user",
                "content": "Since it's delayed until 12 PM, please book a taxi for me 2 hours before the flight.",
            }
        )

        start2 = time.time()
        # Step 2.1: Assistant calculates time and calls book_taxi
        response = completion(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            reasoning_effort=effort,
            vertex_location="global",
            vertex_project=creds["project_id"],
        )
        msg2 = response.choices[0].message
        messages.append(msg2)

        # Verify if it correctly calculated 10 AM
        is_correct = False
        args = {}
        if msg2.tool_calls:
            args = json.loads(msg2.tool_calls[0].function.arguments)
            if "10" in args.get("time", ""):
                is_correct = True

        # DEBUG: Print reasoning and args if incorrect or for all if needed
        if not is_correct:
            print(f"      [yellow]Logic Failure Analysis ({model} - {effort})[/yellow]")
            if hasattr(msg2, "reasoning_content") and msg2.reasoning_content:
                print(f"      [dim]Reasoning:[/dim] {msg2.reasoning_content[:200]}...")
            print(f"      [dim]Tool Args:[/dim] {args}")

        if msg2.tool_calls:
            # Step 2.2: Mock Tool Result
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": msg2.tool_calls[0].id,
                    "content": json.dumps(
                        {"status": "success", "pickup": args.get("time")}
                    ),
                }
            )

            # Step 2.3: Final response
            response = completion(
                model=model,
                messages=messages,
                tools=TOOLS_SCHEMA,
                reasoning_effort=effort,
                vertex_location="global",
                vertex_project=creds["project_id"],
            )
            messages.append(response.choices[0].message)

        t2 = time.time() - start2
        timings.append(t2)

        return {
            "total_time": sum(timings),
            "turn1": t1,
            "turn2": t2,
            "correct": is_correct,
        }

    except Exception as e:
        print(f"  [red]Failed: {str(e)[:100]}[/red]")
        return None


def main():
    creds = load_credentials()
    configs = [
        ("vertex_ai/gemini-3-pro-preview", "low"),
        ("vertex_ai/gemini-3-flash-preview", "high"),
        ("vertex_ai/gemini-3-flash-preview", "medium"),
        ("vertex_ai/gemini-3-flash-preview", "low"),
        ("vertex_ai/gemini-3-flash-preview", "minimal"),
    ]

    results = []
    print(
        "\n[bold yellow]═══ Multi-turn Reasoning Benchmark (2 Turns + Tools) ═══[/bold yellow]\n"
    )

    for model, effort in configs:
        m_name = model.split("/")[-1]
        print(f"🔄 Testing {m_name} ({effort})...")
        res = run_multiturn_test(model, effort, creds)
        if res:
            res["model"] = m_name
            res["effort"] = effort
            results.append(res)
            print(
                f"   ✅ Done: Total {res['total_time']:.2f}s | Correct logic: {'[green]Yes[/green]' if res['correct'] else '[red]No[/red]'}"
            )

    # Display results Table
    table = Table(title="Multi-turn Performance Comparison")
    table.add_column("Model", style="cyan")
    table.add_column("Effort", style="magenta")
    table.add_column("Total Time", justify="right", style="green")
    table.add_column("Turn 1 (s)", justify="right")
    table.add_column("Turn 2 (s)", justify="right")
    table.add_column("Correct?", justify="center")

    for r in results:
        table.add_row(
            r["model"],
            r["effort"],
            f"{r['total_time']:.2f}",
            f"{r['turn1']:.2f}",
            f"{r['turn2']:.2f}",
            "✅" if r["correct"] else "❌",
        )

    print("\n")
    print(table)


if __name__ == "__main__":
    main()
