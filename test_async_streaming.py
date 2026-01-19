"""
Test async streaming for structured LLM
"""

import asyncio
import json
import os

from dotenv import load_dotenv
from google.genai import types
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.llms import ChatMessage
from llama_index.llms.google_genai import GoogleGenAI

load_dotenv()

CREDENTIALS_FILE = "service_account.json"
LLM_MODEL = "gemini-2.5-flash"


def load_credentials() -> dict:
    if not os.path.exists(CREDENTIALS_FILE):
        raise FileNotFoundError(f"Credentials file '{CREDENTIALS_FILE}' not found.")
    with open(CREDENTIALS_FILE, "r") as f:
        creds = json.load(f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_FILE
    return creds


class SimpleResponse(BaseModel):
    """A simple response."""
    message: str
    count: int


async def test_astream_chat():
    creds = load_credentials()
    
    llm = GoogleGenAI(
        model=LLM_MODEL,
        vertexai_config={"project": creds["project_id"], "location": "global"},
        generation_config=types.GenerateContentConfig(
            temperature=1.0,
            thinking_config=types.ThinkingConfig(include_thoughts=True),
        ),
    )
    
    sllm = llm.as_structured_llm(SimpleResponse)
    
    input_msg = ChatMessage.from_str("Say hello and pick a random number between 1 and 100")
    
    print("[DEBUG] Calling astream_chat...")
    stream = await sllm.astream_chat([input_msg])
    print(f"[DEBUG] stream type: {type(stream)}")
    
    print("[DEBUG] Starting async iteration...")
    chunk_count = 0
    async for chunk in stream:
        chunk_count += 1
        print(f"[DEBUG] Chunk #{chunk_count}")
        print(f"  delta: {chunk.delta}")
        print(f"  raw type: {type(chunk.raw)}")
        print(f"  raw: {chunk.raw}")
        print("---")
    
    print(f"[DEBUG] Total chunks: {chunk_count}")


if __name__ == "__main__":
    asyncio.run(test_astream_chat())
