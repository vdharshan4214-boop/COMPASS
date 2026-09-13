from __future__ import annotations

from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END

from ml_utils import classify_category, compute_urgency, compute_priority, find_duplicate, infer_department
from campus_rag import retrieve_campus_context


class ComplaintState(TypedDict):
    description: str
    location: str
    photo_path: Optional[str]
    category: str
    urgency: float
    department: str
    dedup_match: Optional[int]
    priority_score: float


def classify_node(state: ComplaintState) -> ComplaintState:
    state["category"] = classify_category(state["description"])
    state["urgency"] = compute_urgency(state["description"])
    return state


def dedup_node(state: ComplaintState, config: dict | None = None, *, open_complaints: list | None = None) -> ComplaintState:
    complaints = open_complaints
    if not complaints and config and isinstance(config, dict) and "configurable" in config:
        complaints = config["configurable"].get("open_complaints")
    complaints = complaints or []
    state["dedup_match"] = find_duplicate(
        state["description"],
        state["category"],
        state["location"],
        [(c.get("id"), c.get("description"), c.get("category"), c.get("location")) for c in complaints],
    )
    return state



def route_node(state: ComplaintState) -> ComplaintState:
    campus_matches = retrieve_campus_context(state["location"], state["description"])
    state["department"] = campus_matches[0]["team"] if campus_matches else infer_department(state["category"], state["description"])
    return state


def priority_node(state: ComplaintState, *, frequency: int = 1, days_open: int = 0) -> ComplaintState:
    state["priority_score"] = compute_priority(state["urgency"], frequency, days_open)
    return state


def build_reactive_graph():
    graph = StateGraph(ComplaintState)
    graph.add_node("classify_node", classify_node)
    graph.add_node("dedup_node", dedup_node)
    graph.add_node("route_node", route_node)
    graph.add_node("priority_node", priority_node)

    graph.set_entry_point("classify_node")
    graph.add_edge("classify_node", "dedup_node")
    graph.add_conditional_edges(
        "dedup_node",
        lambda state: "update_existing" if state.get("dedup_match") else "route_node",
        {
            "update_existing": END,
            "route_node": "route_node",
        },
    )
    graph.add_edge("route_node", "priority_node")
    graph.add_edge("priority_node", END)
    return graph.compile()


reactive_graph = build_reactive_graph()


def run_reactive_pipeline(description: str, location: str, photo_path: str | None = None, open_complaints: list | None = None):
    initial_state: ComplaintState = {
        "description": description,
        "location": location,
        "photo_path": photo_path,
        "category": "other",
        "urgency": 0.0,
        "department": "Facilities Team",
        "dedup_match": None,
        "priority_score": 0.0,
    }
    if open_complaints is not None:
        return reactive_graph.invoke(initial_state, config={"configurable": {"open_complaints": open_complaints}})
    return reactive_graph.invoke(initial_state)
