# workflows/orchestrator.py

from typing import Dict, Any, List, Callable
from functools import partial
from langgraph.graph import StateGraph, END
from graph_utils.state_types import AgentState
from agents.single_agent import SingleAssistant



# Import agent creation functions
from config.agent_library import (
    get_data_collector_us, get_data_collector_india,
    get_concept_summarizer_global, get_thesis_creator,
    get_shadow_auditor, get_io_agent, get_summarizer_agent,
    get_validation_agent
)

# Import the node logic and the actual tool functions
from . import graph_nodes
from tools import report_utils

def run_orchestration(company: str, year: str, config: Dict[str, Any]):
    """
    Initializes agents and builds the graph for the financial analysis workflow.
    """
    # 1. Instantiate agents
    thesis_agent = get_thesis_creator(config)
    audit_agent = get_shadow_auditor(config)
    io_agent = get_io_agent(config)

    summarizer_agent = get_summarizer_agent(config)
    concept_agent = get_concept_summarizer_global(config)
    validation_agent = get_validation_agent(config)

    # 2. Define the Graph
    workflow = StateGraph(AgentState)

    # 3. Add Nodes to the Graph
    workflow.add_node("resolve_company", graph_nodes.resolve_company_node)
    workflow.add_node("data_collection", graph_nodes.data_collection_node)
    workflow.add_node("validate_collected_data", partial(graph_nodes.validate_collected_data_node, agent=validation_agent))
    workflow.add_node("synchronize_data", lambda state: {})  # Stub for now
    workflow.add_node("summarization", partial(graph_nodes.summarization_node, agent=summarizer_agent))
    workflow.add_node("conceptual_analysis", partial(graph_nodes.conceptual_analysis_node, agent=concept_agent))
    workflow.add_node("generate_report", partial(graph_nodes.generate_report_node, agent=thesis_agent))
    workflow.add_node("save_report", partial(graph_nodes.save_report_node, io_agent=io_agent))
    workflow.add_node("run_evaluation", partial(graph_nodes.run_evaluation_node, audit_agent=audit_agent))
    workflow.add_node("save_evaluation_report", partial(graph_nodes.save_evaluation_report_node, io_agent=io_agent))

    # LLM Decision Node (pre-data)
    workflow.add_node("llm_decision", partial(graph_nodes.llm_decision_node, agent=validation_agent))

    # 4. Wire Up Edges
    workflow.set_entry_point("resolve_company")
    workflow.add_edge("resolve_company", "llm_decision")

    workflow.add_conditional_edges(
        "llm_decision",
        lambda state: getattr(state, "llm_decision", "end"),
        {
            "continue": "data_collection",
            "valid": "data_collection",     
            "approved": "data_collection",     
            "end": END
        }
    )

    # After data collection → run validation on collected files
    workflow.add_edge("data_collection", "validate_collected_data")

    workflow.add_conditional_edges(
        "validate_collected_data",
        lambda state: getattr(state, "llm_decision", "end"),
        {
            "data_collection_continue": "synchronize_data",
            "valid": "synchronize_data", 
            "end": END
        }
    )

    # Standard downstream flow
    workflow.add_edge("synchronize_data", "summarization")
    workflow.add_edge("summarization", "conceptual_analysis")
    workflow.add_edge("conceptual_analysis", "generate_report")
    workflow.add_edge("generate_report", "save_report")
    workflow.add_edge("save_report", "run_evaluation")
    workflow.add_edge("run_evaluation", "save_evaluation_report")
    workflow.add_edge("save_evaluation_report", END)

    # Consolidate initial state definition
    report_work_dir = f"report/{company}_{year}"
    initial_state_for_graph = {
        "company": company,
        "year": year,
        "work_dir": report_work_dir,
        "messages": [], 
        "llm_decision": "continue", 
    }

    print(f"Report directory initialized to: {initial_state_for_graph['work_dir']}")

    # 5. Compile and Run
    app = workflow.compile()
    print("Graph compiled. Starting execution...")

    # Pass the correctly populated initial_state_for_graph
    for event in app.stream(initial_state_for_graph): 
        print(event)
        print("---")

    print("\n✅ Orchestration completed. Check logs and output files.\n")