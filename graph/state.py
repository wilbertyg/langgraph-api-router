from typing import Annotated, Literal, TypedDict
from langgraph.graph import END

from langgraph.graph.message import add_messages


class State(TypedDict, total=False):
    audio_path: str  # set => audio input, otherwise text
    text: str  # user text, or the transcript of the audio
    intent: Literal["translate", "heavy"]
    messages: Annotated[list, add_messages]  # heavy-LLM search loop history
    model: str
    output: str


def check_file_type(state: State) -> str:
    return "transcribe" if state.get("audio_path") else "evaluate_intent"


def route_intent(state: State) -> str:
    """Routing 2: simple tasks (translation) -> light model, everything else -> heavy."""
    return "light_llm" if state["intent"] == "translate" else "heavy_llm"


def route_heavy(state: State) -> str:
    """Loop: if the heavy model asked to search, run the tool and come back; else finish."""
    return "tools" if state["messages"][-1].tool_calls else END