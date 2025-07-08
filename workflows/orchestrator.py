# workflows/orchestrator.py

import os
import json
import re
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END, START
from agents.state_types import AgentState,NodeStatus
#from agents.agent_utils import inject_model_env
from functools import partial
from tabulate import tabulate
from datetime import datetime
import sys
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
    # Initialize current_graph_state as the single source of truth for the evolving state.
    # Ensure it includes all expected keys of AgentState, especially 'memory' and 'termination_reason'.
    # This also sets up the report_dir early and consistently.
    provider = config.get("default_provider", "openai")
    report_dir = os.path.join("report", f"{company}_{year}_{provider}")

    os.makedirs(report_dir, exist_ok=True) # Ensure workdir exists

    current_graph_state = {
        "company": company,
        "year": year,
        "work_dir": report_dir,
        "messages": [],
        "llm_decision": "continue",  # Default initial decision
        "termination_reason": "",    # Default initial termination reason
        "memory": {}                 # Ensure 'memory' key exists as an empty dict initially
    }

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

    log_event("setup_progress", {"step": "Setting up Tools"})

    # === 2. Build StateGraph (Graph definition remains the same)
    g = StateGraph(AgentState)
    g.add_node("Resolve Company", resolve_company_node)
    g.set_entry_point("Resolve Company")
    g.add_node("Branch Decision", region_decision_node)
    g.add_edge("Resolve Company", "Branch Decision")
    g.add_node("Data Collection US", data_collection_us_node)
    g.add_node("Data Collection India", data_collection_indian_node)
    g.add_conditional_edges(
        "Branch Decision",
        lambda state: state.llm_decision,
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

    # === 3. Prepare work dir (already done by initializing current_graph_state)
    logger.info(f"Workdir prepared at {report_dir}")
    log_event("initial_state_ready", {"company": company, "year": year, "work_dir": report_dir})

    # === 4. Compile & stream
    try:
        app = g.compile()
        # Graph PNG and Mermaid generation code
        try:
            png_bytes = app.get_graph().draw_mermaid_png()
            graph_path = os.path.join(report_dir, "orchestration_graph.png")
            with open(graph_path, "wb") as f:
                f.write(png_bytes)
            logger.info(f"Saved orchestration graph to {graph_path}")
            log_event("graph_png_saved", {"path": graph_path})
        except Exception as e:
            logger.error(f"Failed to generate graph PNG: {e}")
            log_event("graph_png_error", {"error": str(e)})

        logger.info("Graph compiled; starting execution")
        print(json.dumps({"event_type": "pipeline_start"}, ensure_ascii=False))

        try:
            mermaid_definition = app.get_graph().draw_mermaid()
            clean_mermaid = re.sub(r'<[^>]+>', '', mermaid_definition)
            log_event("graph_definition", {"mermaid_syntax": clean_mermaid})
            logger.info("Sent Mermaid graph definition to frontend.")
        except Exception as graph_err:
            logger.error(f"Failed to get Mermaid graph definition: {graph_err}")
            log_event("graph_definition_error", {"error": str(graph_err)})

        # Stream events and always update current_graph_state with the latest full state
        for event in app.stream(current_graph_state): # Start stream with the initialized state
            logger.info(f"Event: {event}")

            # LangGraph stream yields {node_name: updated_state_dict} or {'__end__': final_state_dict}
            for node_name, node_output_state in event.items():
                # node_output_state is the full updated AgentState at that point
                current_graph_state = node_output_state
                if node_name == "__end__":
                    # If it's the __end__ event, we've received the final state.
                    # The loop will terminate naturally after this.
                    break # Exit inner loop, outer loop will finish

        # --- AFTER THE STREAM LOOP ---
        # current_graph_state now holds the actual final state of the graph.
        return_state = current_graph_state

        # --- Debugging point for the warning ---
        logger.debug(f"DEBUG: return_state chosen for processing: {return_state}")
        # --- End debugging point ---

        # Check for early termination reason
        llm_decision = return_state.get("llm_decision")
        if llm_decision == "end":
            reason = return_state.get("termination_reason", "No specific reason provided for early termination.")
            logger.info(f"Orchestration terminated early: {reason}")
            log_event("orchestration_completed", {"company": company, "year": year, "status": "terminated_early", "reason": reason})

            # Explicitly state that metrics/tables are skipped for early termination
            logger.info("Skipping metrics extraction and table printing for early termination (llm_decision='end').")
            return return_state # Exit here, as no further metrics/tables are expected

        else: # This path is for a full, successful graph completion (llm_decision is 'us' or 'india')
            logger.info("Orchestration completed successfully")
            log_event("orchestration_completed", {"company": company, "year": year, "status": "success"})


        # --- Guard against missing memory for all subsequent operations for SUCCESSFUL runs ---
        # This block only executes if llm_decision was NOT 'end'.
        if return_state and 'memory' in return_state:
            # Check for specific keys before printing
            region_value = return_state.get("llm_decision", "N/A") # llm_decision is at root level of state

            # Calculate actual duration from start/end times in memory
            duration_value = "N/A"
            if 'pipeline_start_time' in return_state['memory'] and 'pipeline_end_time' in return_state['memory']:
                try:
                    start = datetime.fromisoformat(return_state['memory']['pipeline_start_time'])
                    end = datetime.fromisoformat(return_state['memory']['pipeline_end_time'])
                    duration_value = (end - start).total_seconds()
                except Exception as dur_err:
                    logger.error(f"Error calculating duration: {dur_err}")
                    duration_value = "Calculation Error"

            print("region: ", region_value)
            print("duration:", duration_value)

            # Extract ICAIF Ratings data (assuming it's in memory.final_evaluation.icaif_scores)
            icaif_ratings_data = return_state['memory'].get('final_evaluation', {}).get('icaif_scores', [])
            if isinstance(icaif_ratings_data, dict): # If icaif_scores is a dict, convert to items for tabulate
                icaif_ratings_data = list(icaif_ratings_data.items())

            pipeline_data = return_state['memory'].get('pipeline_data')

            # --- Conditional table printing logic (from main.py, adapted for run_orchestration) ---
            # This logic should ideally be in main.py, but if run_orchestration prints directly:
            if sys.stdout.isatty(): # Only print if in an interactive terminal
                print("\n--- Output Summary ---")

                is_icaif_empty = not bool(icaif_ratings_data)
                is_pipeline_empty = not bool(pipeline_data)

                if is_icaif_empty and is_pipeline_empty:
                    # This case should ideally not be reached if llm_decision was not 'end'
                    # and the full pipeline ran successfully to produce data.
                    # If it is reached, it means the full pipeline ran but produced no data.
                    print("Full orchestration completed, but no relevant data found for output.")
                else:
                    print("\nICAIF Ratings:")
                    if not is_icaif_empty:
                        print(tabulate(
                            tabular_data=icaif_ratings_data,
                            headers=["Criterion", "Score"],
                            tablefmt="plain" # Or "github" as per your node's print
                        ))
                    else:
                        print("[Data for ICAIF Ratings not available.]")

                    print("\nPipeline Matrix:")
                    if not is_pipeline_empty:
                        print(tabulate(
                            tabular_data=pipeline_data,
                            headers=["Agent", "Requests", "Avg Latency (s)", "Errors", "Provider", "Model"],
                            tablefmt="plain" # Or "github" as per your node's print
                        ))
                    else:
                        print("[Data for Pipeline Matrix not available.]")

                print("\n--- End of Summary ---")
            # --- End of conditional table printing logic ---

            if pipeline_data: # Save pipeline metrics if available
                metrics_path = os.path.join(report_dir, "pipeline_metrics.json")
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(pipeline_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved pipeline node metrics to {metrics_path}")
            else:
                logger.warning("No pipeline metrics data available in final state to save.")

        else: # This else block is for when a successful run (not 'end') still has missing memory
            logger.warning("Final state or its memory is missing. Cannot extract metrics for printing/saving for successful run.")
            logger.debug(f"DEBUG: Full return_state for missing memory (unexpected for success): {return_state}")


    except Exception as e:
        err_trace = traceback.format_exc()
        logger.error(f"Orchestration failed: {e}")
        log_event("orchestration_failed", {"error": str(e), "traceback": err_trace})
        raise # Re-raise the exception so main.py can handle it if needed

    return return_state # Always return the final state