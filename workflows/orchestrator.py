# workflows/orchestrator.py

import os
import json
import re
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END, START
from agents.state_types import AgentState,NodeStatus
#from agents.agent_utils import inject_model_env
from functools import partial
import traceback

import tools.graph_tools as graph_tools
from agents.agent_library import (
    resolve_company_node,
    region_decision_node,
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
from utils.config_utils import inject_model_env, get_node_model, get_provider_creds, _init_llm

logger = get_logger()



def run_orchestration(
    company: str,
    year: str,
    config: dict,
    report_type: str,
    verbose: bool
):
    final_state = None
    logger.info(f"Orchestration start → {company} ({year}), report_type={report_type}, verbose={verbose}")
    log_event("orchestration_start", {
        "company": company,
        "year": year,
        "report_type": report_type,
        "verbose": verbose
    })

    log_event("setup_progress", {"step": "Setting up APIs"})

    # === 1. Inject all required LLM envs for this pipeline
    llm_models = config.get("llm_models", {})
    for agent_name, model_name in llm_models.items():
        if not model_name:
            continue
        try:
            provider, _ = get_node_model(agent_name, config)
            creds = get_provider_creds(provider, config)
            model_config = {
                "api_type": provider,
                "model": model_name,
                "api_key": creds.get("api_key"),
                "base_url": creds.get("base_url"),
            }
            inject_model_env(model_config)
            logger.info(f"Injected env for agent '{agent_name}' → model '{model_name}'")
            log_event("model_env_injected", {"agent": agent_name, "model": model_name})
        except Exception as e:
            logger.error(f"Env injection failed for '{agent_name}': {e}")
            log_event("model_env_error", {"agent": agent_name, "error": str(e)})

    log_event("setup_progress", {"step": "Setting up Agents"})
    logger.info("All agents instantiated")
    log_event("agents_ready", {"agents": list(llm_models.keys())})

    # --- Progress: Tools / Graph construction ---
    log_event("setup_progress", {"step": "Setting up Tools"})

  # === 2. Build StateGraph

    g = StateGraph(AgentState)
    #g.add_edge(START, "Resolve Company")
    g.add_node("Resolve Company", resolve_company_node)
    g.set_entry_point("Resolve Company")
    g.add_node("Branch Decision", region_decision_node)
    g.add_edge("Resolve Company", "Branch Decision")
    g.add_node("Data Collection US", data_collection_us_node)
    g.add_node("Data Collection India", data_collection_indian_node)
    g.add_conditional_edges(
        "Branch Decision",
        lambda state: state.llm_decision,  # This lambda reads the 'llm_decision' from the state
                                            # which was updated by llm_decision_node.
        {
            "us": "Data Collection US",
            "india": "Data Collection India",
            "end": END,
        }
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

    # === 3. Prepare work dir & initial state
    provider = config.get("default_provider", "openai")
    # include provider in driectory to run evaluation tests
    report_dir = os.path.join("report", f"{company}_{year}_{provider}")

    os.makedirs(report_dir, exist_ok=True)
    initial_state = {
        "company": company,
        "year": year,
        "work_dir": report_dir,
        "messages": [],
        "llm_decision": "continue",
    }
    logger.info(f"Workdir prepared at {report_dir}")
    log_event("initial_state_ready", {"company": company, "year": year, "work_dir": report_dir})

    # === 4. Compile & stream
    final_state = None
    last_agent_state = None
    try:
        app = g.compile()
        logger.info("Graph compiled; starting execution")
        print(json.dumps({"event_type": "pipeline_start"}, ensure_ascii=False))
        
        # --- NEW: Get Mermaid graph definition and send to frontend ---
        mermaid_graph = app.get_graph().draw_mermaid_png() # Or .draw_mermaid_yaml(), etc.
        # .draw_mermaid_png() might return bytes, you might need .draw_mermaid_yaml() or .draw_mermaid_svg()
        # Let's assume .draw_mermaid_yaml() gives a string
        # If it returns a string, you can directly log it.
        # If it returns something else, you might need to convert.
        # Check LangGraph docs for the best method to get string output.

        # For demonstration, let's use the simplest .get_graph().draw_mermaid() if available
        # or mock it if a direct string output method isn't immediately obvious from docs.
        # The best method is app.get_graph().draw_mermaid() which generates the string.
        try:
            mermaid_definition = app.get_graph().draw_mermaid()
            clean_mermaid = re.sub(r'<[^>]+>', '', mermaid_definition)
            log_event("graph_definition", {"mermaid_syntax": clean_mermaid})
            logger.info("Sent Mermaid graph definition to frontend.")
        except Exception as graph_err:
            logger.error(f"Failed to get Mermaid graph definition: {graph_err}")
            log_event("graph_definition_error", {"error": str(graph_err)})
        # --- END NEW ---

        for event in app.stream(initial_state):
            logger.info(f"Event: {event}")

            # Save the latest state (dict with .memory) if available
            payload = event.get("payload") if isinstance(event, dict) else None
            if isinstance(payload, dict) and 'memory' in payload:
                last_agent_state = payload

            if isinstance(event, dict) and event.get("type") == "pipeline_complete":
                final_state = event.get("payload")
        log_event("pipeline_complete", {})
        logger.info("Orchestration completed successfully")
        log_event("orchestration_completed", {"company": company, "year": year})

        # === Save/Print Pipeline Metrics at the end ===
        # Prefer the last agent state if it has pipeline_data
        metrics_state = last_agent_state or final_state
        pipeline_data = None
        if metrics_state and 'memory' in metrics_state and 'pipeline_data' in metrics_state['memory']:
            pipeline_data = metrics_state['memory']['pipeline_data']
            # Print metrics as table
            from tabulate import tabulate
            log_event("pipeline_metrics_table", {"table": pipeline_data})

            # Save metrics as JSON
            metrics_path = os.path.join(report_dir, "pipeline_metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(pipeline_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved pipeline node metrics to {metrics_path}")

    except Exception as e:
        err_trace = traceback.format_exc()
        logger.error(f"Orchestration failed: {e}")
        log_event("orchestration_failed", {"error": str(e), "traceback": err_trace})
        log_event("orchestration_failed", {"error": str(e)})
        raise

    return last_agent_state or final_state