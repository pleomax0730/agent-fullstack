"""
Error Handling Test: Can the model self-correct after a tool error?

Scenario:
1. User: Book taxi for 10 AM.
2. Assistant: Calls book_taxi(10 AM).
3. Tool Result: {"error": "Pickup must be at least 3 hours before flight. Current flight: 12 PM."}
4. Expectation: Assistant should REASON that it needs updated time (9 AM) and CALL book_taxi again.
"""

import json
import os
from dotenv import load_dotenv
from rich import print
from litellm import completion

load_dotenv()

SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE", "service_account.json")


def load_credentials():
    abs_path = os.path.abspath(SERVICE_ACCOUNT_FILE)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = abs_path
    with open(abs_path, "r") as f:
        return json.load(f)


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "book_taxi",
            "description": "Book a taxi for pickup.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {"type": "string", "description": "Pickup time"}
                },
                "required": ["time"],
            },
        },
    }
]


def test_recovery(model: str, effort: str):
    creds = load_credentials()
    print(f"\n[bold cyan]🧪 Testing Error Recovery: {model} ({effort})[/bold cyan]")

    messages = [
        {
            "role": "user",
            "content": "My flight is at 12 PM. Please book a taxi for 10 AM.",
        }
    ]

    # Step 1: Initial tool call
    resp1 = completion(
        model=model,
        messages=messages,
        tools=TOOLS_SCHEMA,
        reasoning_effort=effort,
        vertex_location="global",
        vertex_project=creds["project_id"],
    )
    msg1 = resp1.choices[0].message
    messages.append(msg1)

    if msg1.tool_calls:
        tc = msg1.tool_calls[0]
        print(f"   [Step 1] Model requested: {tc.function.arguments}")

        # Step 2: Inject Tool Error (The "Constraint" Error)
        error_msg = {
            "error": "Constraint Violated: Pickup must be at least 3 hours before flight for security. Flight is 12 PM."
        }
        messages.append(
            {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(error_msg)}
        )
        print(f"   [Step 2] Injected Error: {error_msg['error']}")

        # Step 3: See if Assistant corrects itself
        resp2 = completion(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            reasoning_effort=effort,
            vertex_location="global",
            vertex_project=creds["project_id"],
        )
        msg2 = resp2.choices[0].message

        if hasattr(msg2, "reasoning_content") and msg2.reasoning_content:
            print(f"\n[bold]Reasoning:[/bold]\n{msg2.reasoning_content}\n")

        if msg2.tool_calls:
            new_args = json.loads(msg2.tool_calls[0].function.arguments)
            print(f"   [Step 3] [green]SUCCESS![/green] Model retried with: {new_args}")
        else:
            print(
                f"   [Step 3] [red]FAILED[/red] - Model did not retry tool call. Content: {msg2.content}"
            )
    else:
        print("   [Step 1] Failed: Model did not call tool initially.")


from rich.table import Table


def main():
    creds = load_credentials()
    configs = [
        ("vertex_ai/gemini-3-pro-preview", "low"),
        ("vertex_ai/gemini-3-flash-preview", "high"),
        ("vertex_ai/gemini-3-flash-preview", "medium"),
        ("vertex_ai/gemini-3-flash-preview", "low"),
        ("vertex_ai/gemini-3-flash-preview", "minimal"),
    ]

    print("\n[bold yellow]--- Final Results ---[/bold yellow]")
    for model, effort in configs:
        m_short = model.split("/")[-1]
        print(f"Testing {m_short} ({effort})...")

        messages = [
            {
                "role": "user",
                "content": "My flight is at 12 PM. Please book a taxi for 10 AM.",
            }
        ]

        try:
            # Turn 1
            resp1 = completion(
                model=model,
                messages=messages,
                tools=TOOLS_SCHEMA,
                reasoning_effort=effort,
                vertex_location="global",
                vertex_project=creds["project_id"],
            )
            msg1 = resp1.choices[0].message
            messages.append(msg1)

            init_call = "✅" if msg1.tool_calls else "❌"
            retry = "❌"
            asks = "❌"
            mentions_9am = "❌"

            if msg1.tool_calls:
                # Turn 2: Inject Error
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg1.tool_calls[0].id,
                        "content": json.dumps(
                            {
                                "error": "Constraint Violated: Pickup must be at least 3 hours before flight. Flight is 12 PM."
                            }
                        ),
                    }
                )

                resp2 = completion(
                    model=model,
                    messages=messages,
                    tools=TOOLS_SCHEMA,
                    reasoning_effort=effort,
                    vertex_location="global",
                    vertex_project=creds["project_id"],
                )
                msg2 = resp2.choices[0].message

                # Analysis
                if msg2.tool_calls:
                    retry = "✅"
                if msg2.content and (
                    "9" in msg2.content
                    or "instead" in msg2.content
                    or "?" in msg2.content
                ):
                    asks = "✅"

                reasoning = getattr(msg2, "reasoning_content", "") or ""
                if "9" in reasoning:
                    mentions_9am = "✅"

            res_str = f"{m_short} ({effort}): Init Call: {init_call}, Auto-Retry?: {retry}, Asks User?: {asks}, Reasoning Mention 9 AM?: {mentions_9am}"
            print(f"  {res_str}")
            with open("error_results.txt", "a", encoding="utf-8") as f:
                f.write(res_str + "\n")

        except Exception as e:
            err_str = f"{m_short} ({effort}): Error: {e}"
            print(f"  {err_str}")
            with open("error_results.txt", "a", encoding="utf-8") as f:
                f.write(err_str + "\n")


if __name__ == "__main__":
    if os.path.exists("error_results.txt"):
        os.remove("error_results.txt")
    main()
