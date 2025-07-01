# workflows/orchestrator.py

import os
import json
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from graph_utils.state_types import AgentState
from functools import partial
from . import graph_nodes

import workflows.graph_nodes as graph_nodes
from config.agent_library import (
    get_company_resolver_agent,
    get_data_collector_us,
    get_data_collector_india,
    get_thesis_creator,
    get_shadow_auditor,
    get_io_agent,
    get_summarizer_agent,
    get_concept_summarizer_global,
    get_validation_agent,
)
from utils.logger import get_logger, log_event
from utils.config_utils import resolve_model_config, inject_model_env

logger = get_logger()



def run_orchestration(
    company: str,
    year: str,
    config: Dict[str, Any],
    report_type: str,
    verbose: bool
):
    
    resolver_agent      = get_company_resolver_agent(config)
    data_collector_us   = get_data_collector_us(config)
    data_collector_in   = get_data_collector_india(config)
    thesis_agent        = get_thesis_creator(config)
    audit_agent         = get_shadow_auditor(config)
    io_agent            = get_io_agent(config)
    summarizer_agent    = get_summarizer_agent(config)
    concept_agent       = get_concept_summarizer_global(config)
    validation_agent    = get_validation_agent(config)

    """data_collection_runnable = partial(
        graph_nodes.data_collection_node,
        tool_map=graph_nodes.TOOL_MAP,
        tasks_fn=graph_nodes.get_data_collection_tasks
    )"""

    validate_data_runnable = partial(
        graph_nodes.validate_collected_data_node,
        validator_agent=validation_agent,
        tasks_fn=graph_nodes.get_data_collection_tasks,
        gen_prompt=graph_nodes.generate_validation_prompt,
        extract=graph_nodes.extract_route
        )    

    # 1. Kickoff
    logger.info(f"Orchestration start → {company} ({year}), report_type={report_type}, verbose={verbose}")
    log_event("orchestration_start", {
        "company": company,
        "year": year,
        "report_type": report_type,
        "verbose": verbose
    })

    # --- Progress: APIs ---
    print(json.dumps({"event_type": "setup_progress", "step": "Setting up APIs"}, ensure_ascii=False))

    # 2. Inject every LLM API key
    llm_models = config.get("llm_models", {})
    providers = config.get("model_providers", [])
    print(json.dumps({"event_type": "setup_progress", "step": "Setting up LLM Models"}, ensure_ascii=False))
    for agent_name, model_name in llm_models.items():
        if not model_name:
            continue
        try:
            mc = resolve_model_config(model_name, providers)
            inject_model_env(mc)
            logger.info(f"Injected env for agent '{agent_name}' → model '{model_name}'")
            log_event("model_env_injected", {"agent": agent_name, "model": model_name})
        except Exception as e:
            logger.error(f"Env injection failed for '{agent_name}': {e}")
            log_event("model_env_error", {"agent": agent_name, "error": str(e)})

    # --- Progress: Agents ---
    print(json.dumps({"event_type": "setup_progress", "step": "Setting up Agents"}, ensure_ascii=False))


    logger.info("All agents instantiated")
    log_event("agents_ready", {"agents": list(llm_models.keys())})

    # --- Progress: Tools / Graph construction ---
    print(json.dumps({"event_type": "setup_progress", "step": "Setting up Tools"}, ensure_ascii=False))

    # 3. Build graph
    workflow = StateGraph(AgentState)

    # these lists will mirror exactly what you register in the graph
    registered_nodes: List[str] = []
    registered_edges: List[tuple[str, str]] = []

    # 1) Resolve → validator decision
    registered_nodes.append("resolve_company")
    workflow.add_node("resolve_company",        graph_nodes.resolve_company_node)
    workflow.add_node("get_sec_metadata",       graph_nodes.get_sec_metadata_node)
    #workflow.add_node("data_collection",        data_collection_runnable)
    workflow.add_node("data_collection", graph_nodes.data_collection_node)
    registered_nodes.append("llm_decision_node")
    workflow.add_node(
        "llm_decision_node",
        partial(graph_nodes.llm_decision_node, agent=validation_agent)
    )

    # 3) Validation & sync
    #workflow.add_node("validate_collected_data", validate_data_runnable)
    def validate_with_debug(state, _):
        # first run your real validator
        result = graph_nodes.validate_collected_data_node(state, validation_agent)
        # then dump the state so you can see everything
        print("[DEBUG validate_collected_data] state at exit:")
        for k, v in state.__dict__.items():
            print(f"  {k} = {v!r}")
        print("[DEBUG validate_collected_data] node returned:", result)
        return result

    workflow.add_node("validate_collected_data", validate_with_debug)    

    registered_nodes.append("synchronize_data")
    workflow.add_node("synchronize_data", lambda state, _: {})

    # 4) Summaries and report generation
    registered_nodes.extend([
        "summarization",
        "conceptual_analysis",
        "generate_report",
        "save_report"
    ])
    workflow.add_node(
        "summarization",
        lambda state, _: graph_nodes.summarization_node(state, summarizer_agent)
    )
    workflow.add_node(
        "conceptual_analysis",
        lambda state, _: graph_nodes.conceptual_analysis_node(state, concept_agent)
    )
    workflow.add_node(
        "generate_report",
        lambda state, _: graph_nodes.generate_report_node(state, thesis_agent)
    )
    workflow.add_node(
        "save_report",
        lambda state, _: graph_nodes.save_report_node(state, io_agent)
    )

    # 5) Audit & finish
    registered_nodes.extend([
        "run_evaluation",
        "save_evaluation_report"
    ])
    workflow.add_node(
        "run_evaluation",
        lambda state, _: graph_nodes.run_evaluation_node(state, audit_agent)
    )
    workflow.add_node(
        "save_evaluation_report",
        lambda state, _: graph_nodes.save_evaluation_report_node(state, io_agent)
    )

    logger.info("Graph nodes added")
    log_event("graph_nodes_added", {})

    # --- now wire up edges, mirroring what you already have ---
    workflow.set_entry_point("resolve_company")
    workflow.add_edge("resolve_company", "llm_decision_node")
    workflow.add_conditional_edges(
        "llm_decision_node",
        lambda state: getattr(state, "llm_decision", "end"),
        {"continue": "get_sec_metadata", "end": END}
    )
    workflow.add_edge("get_sec_metadata", "data_collection")
    # data collection → validate
    workflow.add_edge("data_collection",      "validate_collected_data")
    
    workflow.add_conditional_edges(
        "validate_collected_data",
        lambda state: getattr(state, "__route__", "end"),
        {"valid": "synchronize_data", "end": END}
    )


    # straight-line flow from sync through report
    workflow.add_edge("synchronize_data",      "summarization")
    workflow.add_edge("summarization",         "conceptual_analysis")
    workflow.add_edge("conceptual_analysis",   "generate_report")
    workflow.add_edge("generate_report",       "save_report")
    workflow.add_edge("save_report",           "run_evaluation")
    workflow.add_edge("run_evaluation",        "save_evaluation_report")
    workflow.add_edge("save_evaluation_report", END)

    logger.info("Graph edges configured")
    log_event("graph_edges_configured", {})

    # ——— DEBUG DUMP ———
    print("\n=== WORKFLOW NODES ===")
    for n in registered_nodes:
        print(f" • {n}")

    print("\n=== WORKFLOW EDGES ===")
    for src, dst in registered_edges:
        print(f" {src} → {dst}")
    print(json.dumps(
        {"event_type": "setup_progress", "step": "Resolving Company Region & Peers"},
        ensure_ascii=False
    ))

    # 4. Prepare work dir & initial state
    report_dir = f"report/{company}_{year}"
    os.makedirs(report_dir, exist_ok=True)
    initial_state = {
        "company": company,
        "year": year,
        "work_dir": report_dir,
        "report_type": report_type,
        "verbose": verbose,
        
        # --- Add default values for ALL other required fields ---
        "company_details": {},
        "region": "",
        "messages": [],
        "llm_decision": "",
        "raw_data_files": [],
        
        # --- Specifically add the two missing fields ---
        "filing_date": None,  # Initialize as None, since it's Optional
        "error_log": []       # Initialize as an empty list
    }

    logger.info(f"Workdir prepared at {report_dir}")
    log_event("initial_state_ready", initial_state)

    # 5. Compile & stream
    try:
        app = workflow.compile()
        logger.info("Graph compiled; starting execution")
        log_event("graph_compiled", {})

        print(json.dumps({"event_type": "pipeline_start"}, ensure_ascii=False))
        for event in app.stream(initial_state):
            # forward every event to front-end
            print(json.dumps({"event_type": "graph_event", "payload": str(event)}, ensure_ascii=False))
            logger.info(f"Event: {event}")

        print(json.dumps({"event_type": "pipeline_complete"}, ensure_ascii=False))
        logger.info("Orchestration completed successfully")
        log_event("orchestration_completed", {"company": company, "year": year})

    except Exception as e:
        logger.error(f"Orchestration failed: {e}")
        log_event("orchestration_failed", {"error": str(e)})
        raise
