from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from graph.nodes import evaluate_intent, heavy_llm, light_llm, transcribe
from graph.state import State, check_file_type, route_heavy, route_intent
from services.search import web_search

def build_graph():
    g = StateGraph(State)
    g.add_node("transcribe", transcribe)
    g.add_node("evaluate_intent", evaluate_intent)
    g.add_node("light_llm", light_llm)
    g.add_node("heavy_llm", heavy_llm)
    g.add_node("tools", ToolNode([web_search]))

    g.add_conditional_edges(START, check_file_type, ["transcribe", "evaluate_intent"])
    g.add_edge("transcribe", "evaluate_intent")
    g.add_conditional_edges("evaluate_intent", route_intent, ["light_llm", "heavy_llm"])
    g.add_edge("light_llm", END)
    g.add_conditional_edges("heavy_llm", route_heavy, ["tools", END])
    g.add_edge("tools", "heavy_llm")
    return g.compile()


graph = build_graph()
