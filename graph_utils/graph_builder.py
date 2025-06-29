# graph_builder.py

from langgraph.graph import StateGraph, END 
from .state_types import AgentState
from typing import Callable, Dict, Any
from agents.multi_assistant import MultiAssistant, MultiAssistantWithLeader

def build_multi_agent_graph(agent_runner: MultiAssistant | MultiAssistantWithLeader, flow_config: Dict[str, Any]) -> StateGraph:
    """
    Builds a LangGraph state machine from a MultiAssistant or MultiAssistantWithLeader instance.
    """
    graph = StateGraph(flow_config.get("state_type", AgentState))

    for node in flow_config["nodes"]:
        graph.add_node(node["name"], node["function"])

    for edge in flow_config["edges"]:
        graph.add_edge(edge["from"], edge["to"])

    graph.set_entry_point(flow_config["entry"])
    graph.set_finish_point(flow_config.get("end", END))

    return graph.compile()



# Example flow_config structure (for reference):
"""
flow_config = {
    "state_type": AgentState,
    "entry": "Expert_Investor",
    "end": "Thesis_CoT",
    "nodes": [
        {"name": "Expert_Investor", "function": expert_investor_agent.chat},
        {"name": "Data_CoT_US", "function": data_collector_us.chat},
        {"name": "Concept_CoT", "function": concept_agent.chat},
        {"name": "Thesis_CoT", "function": thesis_writer.chat},
    ],
    "edges": [
        {"from": "Expert_Investor", "to": "Data_CoT_US"},
        {"from": "Data_CoT_US", "to": "Concept_CoT"},
        {"from": "Concept_CoT", "to": "Thesis_CoT"},
    ]
}
"""
