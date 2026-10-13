import logging
from langchain_core.messages import HumanMessage

from graph.state import State
from services import llm, stt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("agentic")


def transcribe(state: State) -> dict:
    logger.info(f"Audio input received: '{state.get('audio_path')}'. Model: {stt.MODEL_NAME}")
    text = stt.transcribe(state["audio_path"])
    logger.info(f"Completed. Transcript: '{text}'")
    return {"text": text}


async def evaluate_intent(state: State) -> dict:
    logger.info(f"Evaluating intent using Router Model: [{llm.ROUTER_MODEL}]")
    intent = await llm.classify(state["text"])
    logger.info(f"Intent evaluated as: '{intent}'")
    return {"intent": intent}


async def light_llm(state: State) -> dict:
    logger.info(f"[{llm.LIGHT_MODEL}] for prompt: '{state.get('text')}'")
    output = await llm.ask(state["text"], llm.LIGHT_MODEL)
    logger.info(f"[{llm.LIGHT_MODEL}] output generated")
    return {"output": output, "model": llm.LIGHT_MODEL}


async def heavy_llm(state: State) -> dict:
    """One loop turn: search again, or answer. Runs repeatedly via the tools node."""
    new = [] if state.get("messages") else [HumanMessage(state["text"])]
    turn_num = len(state.get("messages", [])) + 1
    logger.info(f"[State: heavy_llm] Invoking Heavy Model: [{llm.HEAVY_MODEL}] (loop turn {turn_num})")
    reply = await llm.heavy_step([*state.get("messages", []), *new])
    if reply.tool_calls:
        tool_names = [call.get("name") for call in reply.tool_calls]
        logger.info(f"[{llm.HEAVY_MODEL}] requested tools: {tool_names}")
    else:
        logger.info(f"[{llm.HEAVY_MODEL}] finished with final answer.")
    return {"messages": [*new, reply], "model": llm.HEAVY_MODEL, "output": reply.text}
