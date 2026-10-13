import os
from functools import cache
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from services.search import web_search

load_dotenv()  # GOOGLE_API_KEY comes from .env

ROUTER_MODEL = (os.getenv("GEMINI_ROUTER_MODEL") or "gemini-2.5-flash").strip()
LIGHT_MODEL = (os.getenv("GEMINI_LIGHT_MODEL") or "gemini-2.5-flash").strip()
HEAVY_MODEL = (os.getenv("GEMINI_HEAVY_MODEL") or "gemini-2.5-pro").strip()
MAX_SEARCHES = 3

HEAVY_PROMPT = SystemMessage(
    "Answer the user's question. If you are unsure or need current facts, call web_search, "
    "and keep searching with refined queries until you can answer confidently. "
    "Once you can answer, reply directly without calling a tool."
)


class Intent(BaseModel):
    """'translate' for simple tasks like translation; 'heavy' for complex reasoning."""

    route: Literal["translate", "heavy"]


@cache
def _llm(model: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=model)


async def classify(text: str) -> str:
    result = await _llm(ROUTER_MODEL).with_structured_output(Intent).ainvoke(text)
    return result.route


async def ask(text: str, model: str) -> str:
    return (await _llm(model).ainvoke(text)).text


async def heavy_step(messages: list) -> AIMessage:
    """One turn of the heavy model: either a web_search call or the final answer."""
    model = _llm(HEAVY_MODEL)
    if sum(m.type == "tool" for m in messages) < MAX_SEARCHES:
        model = model.bind_tools([web_search])  # past the cap the model must answer
    return await model.ainvoke([HEAVY_PROMPT, *messages])
