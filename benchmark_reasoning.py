"""
Benchmark: Compare inference time across Gemini models and reasoning levels.

Tests:
- vertex_ai/gemini-3-pro-preview: low
- vertex_ai/gemini-3-flash-preview: high, medium, low, minimal
"""

import json
import os
import time
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


# Test prompt - complex enough to trigger thinking
TEST_PROMPT = """
Analyze this problem step by step:
A train leaves Station A at 9:00 AM traveling at 60 mph towards Station B.
Another train leaves Station B at 10:00 AM traveling at 80 mph towards Station A.
The stations are 280 miles apart.
At what time will the two trains meet?
"""


def run_benchmark(model: str, reasoning_effort: str, creds: dict, runs: int = 3):
    """Run benchmark for a specific model and reasoning level."""
    times = []

    for i in range(runs):
        start = time.time()
        try:
            response = completion(
                model=model,
                messages=[{"role": "user", "content": TEST_PROMPT}],
                stream=False,
                vertex_location="global",
                vertex_project=creds["project_id"],
                reasoning_effort=reasoning_effort,
            )
            elapsed = time.time() - start
            times.append(elapsed)

            # Print thinking content length if available
            msg = response.choices[0].message
            thinking_len = (
                len(msg.reasoning_content)
                if hasattr(msg, "reasoning_content") and msg.reasoning_content
                else 0
            )
            content_len = len(msg.content) if msg.content else 0

            print(
                f"  [dim]Run {i+1}: {elapsed:.2f}s (thinking: {thinking_len} chars, response: {content_len} chars)[/dim]"
            )

        except Exception as e:
            print(f"  [red]Run {i+1}: FAILED - {str(e)[:100]}[/red]")
            times.append(None)

    valid_times = [t for t in times if t is not None]
    if valid_times:
        return {
            "avg": sum(valid_times) / len(valid_times),
            "min": min(valid_times),
            "max": max(valid_times),
            "runs": len(valid_times),
        }
    return None


def main():
    print(
        "\n[bold yellow]═══ Gemini Model & Reasoning Level Benchmark ═══[/bold yellow]\n"
    )

    creds = load_credentials()

    # Test configurations
    tests = [
        ("vertex_ai/gemini-3-pro-preview", "low"),
        ("vertex_ai/gemini-3-flash-preview", "high"),
        ("vertex_ai/gemini-3-flash-preview", "medium"),
        ("vertex_ai/gemini-3-flash-preview", "low"),
        ("vertex_ai/gemini-3-flash-preview", "minimal"),
    ]

    results = []

    for model, effort in tests:
        model_short = model.replace("vertex_ai/", "")
        print(f"\n[bold]Testing: {model_short} | reasoning_effort={effort}[/bold]")

        result = run_benchmark(model, effort, creds, runs=2)

        if result:
            results.append({"model": model_short, "effort": effort, **result})

    # Display results table
    print("\n")
    table = Table(title="Benchmark Results")
    table.add_column("Model", style="cyan")
    table.add_column("Reasoning Level", style="magenta")
    table.add_column("Avg Time (s)", justify="right", style="green")
    table.add_column("Min (s)", justify="right")
    table.add_column("Max (s)", justify="right")

    for r in results:
        table.add_row(
            r["model"],
            r["effort"],
            f"{r['avg']:.2f}",
            f"{r['min']:.2f}",
            f"{r['max']:.2f}",
        )

    print(table)

    # Summary
    print("\n[bold]Key Observations:[/bold]")
    if len(results) >= 2:
        sorted_results = sorted(results, key=lambda x: x["avg"])
        fastest = sorted_results[0]
        slowest = sorted_results[-1]

        print(
            f"  🚀 Fastest: {fastest['model']} ({fastest['effort']}) - {fastest['avg']:.2f}s"
        )
        print(
            f"  🐢 Slowest: {slowest['model']} ({slowest['effort']}) - {slowest['avg']:.2f}s"
        )
        print(
            f"  📊 Difference: {slowest['avg'] - fastest['avg']:.2f}s ({slowest['avg']/fastest['avg']:.1f}x slower)"
        )


if __name__ == "__main__":
    main()
