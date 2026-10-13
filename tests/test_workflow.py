import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from backend.api import app
from graph.builder import graph
from services import llm, search, stt


@pytest.fixture(autouse=True)
def fake_services(monkeypatch):
    queries = []

    async def classify(text):
        return "translate" if "translate" in text else "heavy"

    async def ask(text, model):
        return f"{model}: {text}"

    async def heavy_step(messages):
        # First turn: ask to search. After the tool result arrives: answer.
        if any(m.type == "tool" for m in messages):
            return AIMessage(f"{llm.HEAVY_MODEL}: found it")
        return AIMessage("", tool_calls=[{"name": "web_search", "args": {"query": "q"}, "id": "1"}])

    class FakeDDGS:
        def text(self, query, max_results):
            queries.append(query)
            return [{"title": "t", "href": "u", "body": "b"}]

    monkeypatch.setattr(stt, "transcribe", lambda path: "translate hello" if "translate" in path else "prove theorem")
    monkeypatch.setattr(llm, "classify", classify)
    monkeypatch.setattr(llm, "ask", ask)
    monkeypatch.setattr(llm, "heavy_step", heavy_step)
    monkeypatch.setattr(search, "DDGS", FakeDDGS)
    return queries


@pytest.mark.parametrize(
    "payload, text",
    [
        ({"text": "translate hello"}, "translate hello"),
        ({"audio_path": "translate.wav"}, "translate hello"),  # via transcribe
    ],
)
async def test_light_route(payload, text, fake_services):
    result = await graph.ainvoke(payload)
    assert result["text"] == text
    assert result["model"] == llm.LIGHT_MODEL
    assert result["output"] == f"{llm.LIGHT_MODEL}: {text}"
    assert fake_services == []  # no searching on the light path


@pytest.mark.parametrize(
    "payload, text",
    [
        ({"text": "prove theorem"}, "prove theorem"),
        ({"audio_path": "hard.wav"}, "prove theorem"),  # via transcribe
    ],
)
async def test_heavy_route_searches_then_answers(payload, text, fake_services):
    result = await graph.ainvoke(payload)
    assert result["text"] == text
    assert result["model"] == llm.HEAVY_MODEL
    assert result["output"] == f"{llm.HEAVY_MODEL}: found it"
    assert fake_services == ["q"]  # looped once through the search tool


def test_api():
    client = TestClient(app)
    assert client.post("/process", data={"text": "translate hi"}).json()["model"] == llm.LIGHT_MODEL
    audio = {"file": ("hard.wav", b"RIFF", "audio/wav")}
    heavy = client.post("/process", files=audio).json()
    assert heavy["model"] == llm.HEAVY_MODEL
    assert heavy["output"].endswith("found it")
    assert client.post("/process").status_code == 400
