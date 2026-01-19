"""
Structured Prediction

LlamaIndex provides an intuitive interface for converting any LLM into a
structured LLM through structured_predict - simply define the target Pydantic
class (can be nested), and given a prompt, we extract out the desired object.
"""

import json
import os
from typing import List

from dotenv import load_dotenv
from google.genai import types
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.llms import ChatMessage
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.google_genai import GoogleGenAI
from rich import print

load_dotenv()

CREDENTIALS_FILE = "service_account.json"
LLM_MODEL = "gemini-2.5-flash"


def load_credentials() -> dict:
    """Load VertexAI credentials from file."""
    if not os.path.exists(CREDENTIALS_FILE):
        raise FileNotFoundError(f"Credentials file '{CREDENTIALS_FILE}' not found.")
    with open(CREDENTIALS_FILE, "r") as f:
        creds = json.load(f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_FILE
    return creds


class MenuItem(BaseModel):
    """A menu item in a restaurant."""

    course_name: str
    is_vegetarian: bool


class Restaurant(BaseModel):
    """A restaurant with name, city, and cuisine."""

    name: str
    city: str
    cuisine: str
    menu_items: List[MenuItem]


creds = load_credentials()

llm = GoogleGenAI(
    model=LLM_MODEL,
    vertexai_config={"project": creds["project_id"], "location": "global"},
    generation_config=types.GenerateContentConfig(
        temperature=1.0,
        thinking_config=types.ThinkingConfig(include_thoughts=True)
    ),
)

prompt_tmpl = PromptTemplate("Generate a restaurant in a given city {city_name}")

# Option 1: Use `as_structured_llm`
restaurant_obj = (
    llm.as_structured_llm(Restaurant)
    .complete(prompt_tmpl.format(city_name="Miami"))
    .raw
)
# Option 2: Use `structured_predict`
# restaurant_obj = llm.structured_predict(Restaurant, prompt_tmpl, city_name="Miami")

print(restaurant_obj)

# Example output:
# name='Pasta Mia' city='Miami' cuisine='Italian' menu_items=[MenuItem(course_name='pasta', is_vegetarian=False)]


# -----------------------------------------------------------------------------
# Structured Prediction with Streaming
# Any LLM wrapped with as_structured_llm supports streaming through stream_chat.
# -----------------------------------------------------------------------------

input_msg = ChatMessage.from_str("Generate a restaurant in San Francisco")

sllm = llm.as_structured_llm(Restaurant)
stream_output = sllm.stream_chat([input_msg])
for partial_output in stream_output:
    print(partial_output.delta)
    print("---" * 20)
    restaurant_obj = partial_output.raw

print(restaurant_obj)

# Example output:
# {'city': 'San Francisco',
#  'cuisine': 'Italian',
#  'menu_items': [{'course_name': 'pasta', 'is_vegetarian': False}],
#  'name': 'Italian Delight'}
#
# Restaurant(name='Italian Delight', city='San Francisco', cuisine='Italian', menu_items=[MenuItem(course_name='pasta', is_vegetarian=False)])
