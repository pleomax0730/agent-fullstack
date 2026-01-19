"""
Test: Gemini 3 thought_signature Behavior with Function Calling
================================================================

【核心發現 Key Findings】

1. Signature 必要性 (Signature Requirement):
   - 當使用 Thinking Mode (thinking_config.thinking_level != None) 時，
     多輪對話中的 Function Call 必須帶有 thought_signature，否則會報 400 錯誤。
   - 這個要求與 include_thoughts 設定無關（即使 include_thoughts=False 也需要）。

2. Parallel Tool Calls 的 Signature 數量:
   - 官方文檔說明：只有第一個 function_call 需要 signature。
   - 實測結果：提供 1 個或多個 signature 都能成功，API 對多餘的 signature 寬容接受。

3. Token 成本:
   - 1 個 Signature vs 2 個 Signature：Prompt Tokens 完全相同 (59 tokens)。
   - 結論：thought_signature 不計入 input tokens，可能被視為 metadata。

4. 特殊 Signature 值:
   - "skip_thought_signature_validator" 可跳過驗證（用於測試/開發）。
   - "context_engineering_is_the_way_to_go" 也可用於跳過驗證。
   - 注意：傳入時需轉為 bytes，例如 b"skip_thought_signature_validator"。

5. LlamaIndex Role Handling (Fixed in v0.8.3):
   - Previous versions had a bug checking only MessageRole.MODEL.
   - Fixed in llama-index-llms-google-genai v0.8.3+.
   - MessageRole.ASSISTANT is now correctly handled and recommended.

【實作建議 Implementation Recommendations】

1. DB 設計：不需要儲存真實的 thought_signature，讀取時動態注入 dummy signature。

2. 最簡單的注入策略 (支援 LlamaIndex 標準 Role):
   ```python
   # 使用 MessageRole.ASSISTANT
   DUMMY_SIG = b"skip_thought_signature_validator"
   blocks = [...] # ToolCallBlocks
   additional_kwargs = {"thought_signatures": [DUMMY_SIG] * len(blocks)}
   msg = ChatMessage(role=MessageRole.ASSISTANT, blocks=blocks, additional_kwargs=additional_kwargs)
   ```

3. 不管 Assistant Message 有幾個 ToolCallBlock，只要列表裡有 1 個 dummy signature 即可通過驗證。

【測試場景 Test Scenarios】

- Test 1: 1 Signature for 2 Parallel Calls → SUCCESS
- Test 2: 2 Signatures for 2 Parallel Calls → SUCCESS
- Test 3: DB Reload (ASSISTANT Role + Dummy Sig) → SUCCESS
- Token Usage: thought_signature apparently adds NO token cost.
"""

import asyncio
import json
import os
from dotenv import load_dotenv
from google.genai import types

load_dotenv()

SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE", "service_account.json")
SKIP_SIGNATURE = "skip_thought_signature_validator"


def load_credentials():
    abs_path = os.path.abspath(SERVICE_ACCOUNT_FILE)
    with open(abs_path, "r") as f:
        creds = json.load(f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = abs_path
    return creds


# --- LlamaIndex Imports ---
from llama_index.core.llms import ChatMessage
from llama_index.core.base.llms.types import MessageRole, ToolCallBlock
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.openai import OpenAI


async def run_parallel_test(test_name, signature_count):
    print(f"\n{test_name}")
    print("-" * 50)

    creds = load_credentials()
    project_id = creds["project_id"]

    llm = GoogleGenAI(
        model="gemini-3-pro-preview",
        vertexai_config={"project": project_id, "location": "global"},
        generation_config=types.GenerateContentConfig(
            system_instruction="You are a helpful assistant.",
            temperature=1.0,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.LOW,
            ),
        ),
    )

    # Prepare Signatures list (Bytes)
    sigs = [SKIP_SIGNATURE.encode("utf-8") for _ in range(signature_count)]

    additional_kwargs = {"thought_signatures": sigs}

    # Construct PARALLEL Tool Calls (2 calls in 1 message)
    messages = [
        ChatMessage(role=MessageRole.USER, content="Check flights AA100 and BA200"),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            blocks=[
                ToolCallBlock(
                    tool_call_id="call_001",
                    tool_name="check_flight",
                    tool_kwargs={"flight": "AA100"},
                ),
                ToolCallBlock(
                    tool_call_id="call_002",
                    tool_name="check_flight",
                    tool_kwargs={"flight": "BA200"},
                ),
            ],
            additional_kwargs=additional_kwargs,
        ),
        ChatMessage(
            role=MessageRole.TOOL,
            content='{"status": "delayed"}',
            additional_kwargs={"tool_call_id": "call_001"},
        ),
        ChatMessage(
            role=MessageRole.TOOL,
            content='{"status": "on_time"}',
            additional_kwargs={"tool_call_id": "call_002"},
        ),
    ]

    try:
        response = await llm.achat(messages)

        prompt_tokens = 0
        completion_tokens = 0

        # Correct way: response.raw is a dict, use .get()
        if hasattr(response, "raw") and response.raw:
            usage_metadata = response.raw.get("usage_metadata")
            if usage_metadata:
                prompt_tokens = usage_metadata.get("prompt_token_count", 0)
                completion_tokens = usage_metadata.get("candidates_token_count", 0)

        print(f"RESULT: SUCCESS")
        print(f"Prompt Tokens: {prompt_tokens}")
        print(f"Completion Tokens: {completion_tokens}")
        return prompt_tokens

    except Exception as e:
        error_msg = str(e)
        print(f"RESULT: FAILED")
        print(f"Error: {error_msg}")
        return None


async def main():
    print("Running Gemini Parallel Function Call Signature Token Usage Tests")

    # TEST 1: Standard (1 Signature)
    print("\n>> TEST 1: Standard (1 Signature)")
    tokens_1 = await run_parallel_test(test_name="TEST 1: Standard", signature_count=1)

    # TEST 2: All Filled (2 Signatures)
    print("\n>> TEST 2: All Filled (2 Signatures)")
    tokens_2 = await run_parallel_test(
        test_name="TEST 2: All Filled", signature_count=2
    )

    print(f"\n" + "=" * 60)
    print("TOKEN USAGE SUMMARY")
    print("=" * 60)
    print(f"Test 1 (1 Sig):  {tokens_1} prompt tokens")
    print(f"Test 2 (2 Sigs): {tokens_2} prompt tokens")

    # TEST 3: Simulate DB Reload (Best Practice)
    print("\n" + "=" * 60)
    print(">> TEST 3: DB Reload Simulation (Injecting Dummy)")
    print("=" * 60)

    # Simulate data stored in DB (clean, no signatures)
    db_rows = [
        {"role": "user", "content": "Check flight AA100"},
        {
            "role": "model",
            "tool_calls": [
                {"id": "call_99", "name": "check_flight", "args": {"flight": "AA100"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_99", "content": '{"status": "delayed"}'},
    ]

    print("1. Loading from DB (No signatures stored)...")

    # Reconstruct ChatMessage with Dummy Injection
    reloaded_messages = []
    for row in db_rows:
        role = row["role"]

        if role == "model" and "tool_calls" in row:
            # THIS IS THE KEY STEP
            # Inject Dummy Signature when loading Assistant Tool Calls
            blocks = []
            for tc in row["tool_calls"]:
                blocks.append(
                    ToolCallBlock(
                        tool_call_id=tc["id"],
                        tool_name=tc["name"],
                        tool_kwargs=tc["args"],  # Stringify for OpenAI - json.dumps(tc["args"])
                    )
                )

            # Simple Strategy: Inject for ALL blocks (easier implementation)
            # DUMMY_SIG must be bytes
            dummy_sigs = [SKIP_SIGNATURE.encode("utf-8")] * len(blocks)

            msg = ChatMessage(
                role=MessageRole.ASSISTANT,  # Or ASSISTANT, mapping happens later
                blocks=blocks,
                additional_kwargs={"thought_signatures": dummy_sigs},
            )
        elif role == "tool":
            msg = ChatMessage(
                role=MessageRole.TOOL,
                content=row["content"],
                additional_kwargs={"tool_call_id": row["tool_call_id"]},
            )
        else:
            msg = ChatMessage(role=MessageRole.USER, content=row["content"])

        reloaded_messages.append(msg)

    print("2. Injected Dummy Signatures into ChatMessages")
    print("3. Sending to Gemini...")

    # Run the reloaded history
    result3 = False
    try:
        creds = load_credentials()
        llm = GoogleGenAI(
            model="gemini-3-pro-preview",
            vertexai_config={"project": creds["project_id"], "location": "global"},
            generation_config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.LOW
                ),
            ),
        )

        # Switch to OpenAI for compatibility testing
        # OpenAI usually ignores extra kwargs, but let's verify bytes serialization behavior
        # llm = OpenAI(model="gpt-4o-mini", temperature=0)

        response = await llm.achat(reloaded_messages)
        print(f"RESULT: ✅ SUCCESS")
        print(f"Response: {response.message.content}")
        result3 = True
    except Exception as e:
        print(f"RESULT: ❌ FAILED - {e}")

    if result3:
        print("\nCONCLUSION: DB Reload Strategy works perfectly with Dummy Signatures.")


if __name__ == "__main__":
    asyncio.run(main())
