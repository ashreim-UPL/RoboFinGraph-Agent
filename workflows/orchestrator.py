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
    get_shadow_auditor, get_io_agent, get_summarizer_agent
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

    # --- FIX #1: Create a real TOOL_MAP dictionary to map task names to functions ---
    # --- MODIFICATION: Updated TOOL_MAP and task list with descriptive filenames ---
    TOOL_MAP: Dict[str, Callable] = {
        # Note: 'get_risk_assessment' and others are now consolidated into get_company_profile
        "get_sec_10k_sections": report_utils.get_sec_10k_sections,
        "get_company_profile": report_utils.get_company_profile,
        "get_key_data": report_utils.get_key_data,
        "get_competitors": report_utils.get_competitor_analysis,
        "get_income_statement": partial(report_utils.get_financial_statement, statement_type="income_statement"),
        "get_balance_sheet": partial(report_utils.get_financial_statement, statement_type="balance_sheet"),
        "get_cash_flow": partial(report_utils.get_financial_statement, statement_type="cash_flow_statement"),
        "get_pe_eps_chart": report_utils.generate_pe_eps_chart,
        "get_share_performance_chart": report_utils.generate_share_performance_chart,
    }

    data_collection_tasks: List[Dict[str, str]] = [
        {'task': 'get_sec_10k_sections', 'file': 'report/sec_filings'},
        {'task': 'get_key_data',                  'file': 'report/key_data.json'},
        {'task': 'get_company_profile',           'file': 'report/company_profile.json'},
        {'task': 'get_competitors',               'file': 'report/competitors.json'},
        {'task': 'get_income_statement',          'file': 'report/income_statement.json'},
        {'task': 'get_balance_sheet',             'file': 'report/balance_sheet.json'},
        {'task': 'get_cash_flow',                 'file': 'report/cash_flow.json'},
        {'task': 'get_pe_eps_chart',              'file': 'report/pe_eps_performance.png'},
        {'task': 'get_share_performance_chart',   'file': 'report/share_performance.png'}
    ]

    # 2. Define the Graph
    workflow = StateGraph(AgentState)

    # 3. Add Nodes to the Graph
    workflow.add_node("resolve_company", graph_nodes.resolve_company_node)

    for item in data_collection_tasks:
        node_name = item['task'].split('.')[-1]
        task_function = TOOL_MAP[item['task']]
        output_file = item['file']
        
        # --- FIX #3: Pass the callable 'task_function' to the partial, not the 'task_name' string ---
        node_runnable = partial(
            graph_nodes.data_collection_node,
            task_function=task_function,
            output_file=output_file
        )
        workflow.add_node(node_name, node_runnable)

    # Add other nodes

    workflow.add_node("synchronize_data", lambda state: {})
    workflow.add_node("summarization", partial(graph_nodes.summarization_node, agent=summarizer_agent))

    workflow.add_edge('synchronize_data', 'summarization')              # ✅ Flow: sync -> summarization
    workflow.add_edge('summarization', 'conceptual_analysis')           # ✅ Summarization -> concept (or report)

    workflow.add_node("conceptual_analysis", partial(graph_nodes.conceptual_analysis_node, agent=concept_agent))
    workflow.add_node("generate_report", partial(graph_nodes.generate_report_node, agent=thesis_agent))
    workflow.add_node("save_report", partial(graph_nodes.save_report_node, io_agent=io_agent))
    workflow.add_node("run_evaluation", partial(graph_nodes.run_evaluation_node, audit_agent=audit_agent))
    workflow.add_node("save_evaluation_report", partial(graph_nodes.save_evaluation_report_node, io_agent=io_agent))

    # ---
    # 4. Set the Entry Point
    # ---

    workflow.set_entry_point("resolve_company")
    workflow.add_conditional_edges(
        "resolve_company",
        graph_nodes.decide_to_continue,
        {
            # Use the first task's generated node_name for the entry point to the parallel tasks
            "continue": data_collection_tasks[0]['task'].split('.')[-1],
            "end": END
        }
    )
    
    # The rest of the wiring is correct and uses the generated node names
    for item in data_collection_tasks:
        node_name = item['task'].split('.')[-1]
        workflow.add_edge("resolve_company", node_name)
        workflow.add_edge(node_name, "synchronize_data")

    workflow.add_edge('synchronize_data', 'conceptual_analysis')
    workflow.add_edge('conceptual_analysis', 'generate_report')
    workflow.add_edge('generate_report', 'save_report')
    workflow.add_edge('save_report', 'run_evaluation')
    workflow.add_edge('run_evaluation', 'save_evaluation_report')
    workflow.add_edge('save_evaluation_report', END)


    # 5. Compile and Run
    app = workflow.compile()
    print("Graph compiled. Starting execution...")
    initial_state = {"company": company, "year": year}
    for event in app.stream(initial_state):
        print(event)
        print("---")
    print("\n✅ Orchestration completed. Check logs and output files.\n")