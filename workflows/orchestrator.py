# workflows/orchestrator.py

import os
import json
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from agents.state_types import AgentState
from agents.agent_utils import inject_model_env
from functools import partial

import tools.graph_tools as graph_tools
from agents.agent_library import (
    resolve_company_node,
    llm_decision_node,
    data_collection_us_node,
    data_collection_indian_node,
    validate_collected_data_node,
    synchronize_data_node,
    summarization_node,
    validate_summarized_data_node,
    concept_analysis_node,
    validate_analyzed_data_node,
    generate_report_node,
    run_evaluation_node,
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

    g = StateGraph(AgentState)

    g.add_node("Resolve Company", resolve_company_node)
    g.add_node("Branch Decision", llm_decision_node)
    g.add_node("Data Collection US", data_collection_us_node)
    g.add_node("Data Collection India", data_collection_indian_node)
    g.add_conditional_edges(
        "Branch Decision",
        {"branch":"us"}    , "Data Collection US",
        {"branch":"india"}, "Data Collection India",
        {"branch":"end"}   , END,
    )
    g.add_node("Validate Collected Data", validate_collected_data_node)
    g.add_edge("Data Collection US",   "Validate Collected Data")
    g.add_edge("Data Collection India","Validate Collected Data")
    g.add_node("Synchronize Data", synchronize_data_node)
    g.add_edge("Validate Collected Data", "Synchronize Data")
    g.add_node("Summarize", summarization_node)
    g.add_edge("Synchronize Data", "Summarize")
    g.add_node("Validate Summaries", validate_summarized_data_node)
    g.add_edge("Summarize", "Validate Summaries")
    g.add_node("Conceptual Analysis", concept_analysis_node)
    g.add_edge("Validate Summaries", "Conceptual Analysis")
    g.add_node("Validate Analyzed Data", validate_analyzed_data_node)
    g.add_edge("Conceptual Analysis", "Validate Analyzed Data")
    g.add_node("Generate Report", generate_report_node)
    g.add_edge("Validate Analyzed Data", "Generate Report")
    g.add_node("Run Evaluation", run_evaluation_node)
    g.add_edge("Generate Report", "Run Evaluation")
    g.add_edge("Run Evaluation", END)

    logger.info("Graph edges configured")
    log_event("graph_edges_configured", {})

    # 4. Prepare work dir & initial state
    report_dir = os.path.join("report", f"{company}_{year}")
    os.makedirs(report_dir, exist_ok=True)

    initial_state_object = AgentState(
        company=company,
        year=year,
        user_input=company, # or user_input if different from company
        work_dir=report_dir # AgentState's __init__ might set this if not provided, but being explicit is good
    )

    logger.info(f"Workdir prepared at {report_dir}")
    log_event("initial_state_ready", initial_state)

    # 5. Compile & stream
    try:
        app = g.compile()
        logger.info("Graph compiled; starting execution")
        log_event("graph_compiled", {})

        print(json.dumps({"event_type": "pipeline_start"}, ensure_ascii=False))
        for event in app.stream(initial_state_object):
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
