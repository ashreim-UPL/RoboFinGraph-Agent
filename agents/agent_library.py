import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from functools import wraps
from datetime import datetime
import sys
from utils.logger import get_logger, log_event
from langchain.schema import HumanMessage

from utils.config_utils import LangGraphLLMExecutor
from agents.state_types import AgentState, NodeState, TOOL_MAP, NodeStatus
from tools.report_utils import (
    get_sec_10k_section_1, get_sec_10k_section_1a, get_sec_10k_section_7,
    get_company_profile, get_key_data, get_competitors,
    get_income_statement, get_balance_sheet, get_cash_flow,
    get_pe_eps_chart, get_share_performance_chart, get_financial_metrics
)
from tools.company_search import process_company_data
from tools.graph_tools import summarize_sections, generate_concept_insights, validate_summaries, validate_insights, evaluate_pipeline
from pathlib import Path
from prompts.summarization_intsruction import summarization_prompt_library
from agents.state_types import FILE_TO_TEMPLATE_KEY
from prompts.report_summaries import report_section_specs
from tools.report_writer import ReportLabUtils


# === Pipeline Metrics Recorder ===
def _record_metrics(agent_state: AgentState, node_state: NodeState, node_key: str) -> None:
    """
    Append NodeState metrics to agent_state.memory['pipeline_data'] for KPI calculation.
    """
    metrics = {
        "node": node_key,
        "status": node_state.status.value,
        "duration": node_state.duration,
        "cost_llm": node_state.cost_llm,
        "tokens_sent": node_state.tokens.sent,
        "tokens_generated": node_state.tokens.generated,
        "tools_used": list(node_state.tools_used),
        "errors": node_state.errors.copy(),
        "files_created": node_state.files_created.copy(),
        "files_created_count": len(node_state.files_created)
    }
    pipeline = agent_state.memory.setdefault("pipeline_data", [])
    pipeline.append(metrics)

# === Decorator for Node Functions ===
import json

def record_node(node_key: str):
    """
    Decorator to wrap graph node functions: sets status, logs start/end events,
    captures timing and errors, and records metrics.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(agent_state: AgentState, node_state: NodeState, *args, **kwargs):
            node_state.status = NodeStatus.RUNNING
            start_time = datetime.now()
            # Emit start event for front-end
            start_event = {"event_type": "node_start", "data": {"node": node_key, "timestamp": start_time.isoformat()}}
            sys.stdout.write(json.dumps(start_event) + "\n")
            sys.stdout.flush()
            # Log start internally
            log_event("node_start", {"node": node_key, "timestamp": start_time.isoformat()})

            try:
                result = fn(agent_state, node_state, *args, **kwargs)
                node_state.status = NodeStatus.SUCCESS
                return result
            except Exception as e:
                node_state.status = NodeStatus.ERROR
                node_state.errors.append(str(e))
                raise
            finally:
                end_time = datetime.now()
                node_state.end_time = end_time
                node_state.duration = (end_time - start_time).total_seconds()
                # Record metrics in memory
                _record_metrics(agent_state, node_state, node_key)
                # Emit end event
                metrics = agent_state.memory.get("pipeline_data", [])[-1]
                end_event = {"event_type": "node_end", "data": metrics}
                sys.stdout.write(json.dumps(end_event) + "\n")
                sys.stdout.flush()
                # Log end internally
                log_event("node_end", metrics)
        return wrapper
    return decorator

# === Graph Node Implementations ===

@record_node("resolve_company")
async def resolve_company_node(agent_state: AgentState, node_state: NodeState) -> AgentState: 
    """
    Resolves company details using the specialized 'process_company_data' tool (LLM-only).
    This node updates agent_state with comprehensive company_details directly from the LLM.
    """
    # 1. Get the company query from AgentState.
    company_query = agent_state.company 
    year  = agent_state.year

    if not company_query:
        error_msg = "Error: No company query provided in agent_state.company."
        print(error_msg)
        agent_state.error_log.append(f"resolve_company_node: {error_msg}") 
        agent_state.company_details = {"error": error_msg} #
        agent_state.validation_result_key = "No_Company_Query" # New specific status
        agent_state.accuracy_score = 0
        return agent_state
    try:
        # 2. Call the asynchronous process_company_data function.
        result = await process_company_data(company_query, year) 

        # 3. Update the agent_state with the LLM's results.
        data   = result.get("final_data", {})
        agent_state.company_details       = data
        agent_state.region                = data.get("region", agent_state.region)
        agent_state.filing_date           = data.get("filing_date")  # if returned
        agent_state.validation_result_key = result.get("validation_status", "valid")
        agent_state.accuracy_score        = result.get("accuracy_score", 0.0)
        if result.get("message"):
            agent_state.error_log.append(result["message"])

        # Optionally, update agent_state.company with the LLM's resolved name
        # if it's deemed more official or for downstream consistency.
        official = data.get("official_name") 
        if official and official.lower() != company_query.lower():
            agent_state.company = official #

    except Exception as e:
        error_msg = f"resolve_company_node error: {e}"
        agent_state.error_log.append(error_msg) 
        agent_state.company_details = {"error": error_msg} 
        agent_state.validation_result_key = "Node_Error" 
        agent_state.accuracy_score = 0.0

    return agent_state

@record_node("llm_decision")
def llm_decision_node(agent_state: AgentState, node_state: NodeState) -> AgentState:
    """
    Determine data‐collection branch based on company region (no LLM).
    Sets both agent_state.llm_decision and agent_state.memory['branch'].
    """
    # 1) Derive region from resolved company details or fallback to top‐level
    region = (
        agent_state.company_details.get("region")
        if agent_state.company_details
        else agent_state.region
    ) or "US"
    region_norm = region.strip().lower()

    # 2) Map to one of the supported branches
    if region_norm in ("us", "usa", "united states"):
        branch = "us"
    elif region_norm in ("india", "in"):
        branch = "india"
    else:
        branch = "us"

    # 3) Persist both the “decision” and the explicit branch flag
    agent_state.llm_decision           = branch
    agent_state.memory["branch"]      = branch

    return agent_state

@record_node("data_collection_switch")
def data_collection_switch_node(agent_state: AgentState, node_state: NodeState) -> AgentState:
    """Route pipeline based on LLM decision: 'us', 'india', or 'end'."""
    decision = (agent_state.llm_decision or "").strip().lower()
    if decision in ("us", "continue"):  branch = "us"
    elif decision == "india":             branch = "india"
    elif decision == "end":               branch = "end"
    else:                                  branch = "us"
    agent_state.memory["branch"] = branch
    return agent_state

@record_node("data_collection_us")
def data_collection_us_node(agent_state: AgentState, node_state: NodeState) -> AgentState:
    """
    Invoke all data‐collection tools (US); identical to the Indian variant for now.
    """
    data = {}
    for tool_name, fn in TOOL_MAP.items():
        result = fn(agent_state.company)
        data[tool_name] = result
        node_state.tools_used.add(tool_name)
    agent_state.memory["raw_data"] = data
    return agent_state

@record_node("data_collection_indian")
def data_collection_indian_node(agent_state: AgentState, node_state: NodeState) -> AgentState:
    """
    Invoke all data‐collection tools (India); currently identical to US.
    """
    data = {}
    for tool_name, fn in TOOL_MAP.items():
        result = fn(agent_state.company)
        data[tool_name] = result
        node_state.tools_used.add(tool_name)
    agent_state.memory["raw_data"] = data
    return agent_state

@record_node("validate_collected_data")
def validate_collected_data_node(agent_state: AgentState, node_state: NodeState) -> AgentState:
    """
    Verify each expected raw file exists and is valid JSON via our shared tool.
    Compute accuracy = 1.0 only if all pass; otherwise accuracy = 0.0, flag 'invalid'
    and set branch to 'end'.
    """
    # 1. Build the list of expected files
    tasks = agent_state.get_data_collection_tasks()

    # 2. Delegate existence + JSON validity to helper
    from tools.graph_tools import validate_raw_data
    result = validate_raw_data(tasks)

    # 3. Capture which files we actually read
    for path in result["files_read"]:
        node_state.files_read.append(path)

    # 4. Compute pass/fail
    total      = result["total"]
    valid_cnt  = result["valid_count"]
    if valid_cnt == total:
        agent_state.accuracy_score        = 1.0
        agent_state.validation_result_key = "valid"
    else:
        agent_state.accuracy_score        = 0.0
        agent_state.validation_result_key = "invalid"

        missing = result.get("missing_files", [])
        corrupt = result.get("corrupt_files", [])
        reasons = []
        if missing: reasons.append(f"missing: {missing}")
        if corrupt: reasons.append(f"corrupt: {corrupt}")

        agent_state.error_log.append(
            f"validate_collected_data failed ({'; '.join(reasons)})"
        )
        # tell the orchestrator to stop
        agent_state.memory["branch"] = "end"

    return agent_state


@record_node("synchronize_data")
def synchronize_data_node(agent_state: AgentState, node_state: NodeState) -> AgentState:
    agent_state.memory["normalized_data"] = agent_state.memory.get("raw_data", {})
    return agent_state


@record_node("summarization")
def summarization_node(agent_state: AgentState, node_state: NodeState) -> AgentState:
    executor = LangGraphLLMExecutor("summarizer")

    # 1) Pull prompts from our prompt library
    prompts = summarization_prompt_library

    # 2) Summarize each section
    summaries, count = summarize_sections(
        raw_data_dir=Path(agent_state.raw_data_dir),         
        filing_dir=Path(agent_state.filing_dir),             
        prompts=prompts,                                    
        file_key_map=FILE_TO_TEMPLATE_KEY,                   
        llm_executor=executor
    )

    # 3) Store results and an easy KPI
    agent_state.memory["summaries"] = summaries
    node_state.custom_metrics["sections_processed"] = count

    return agent_state

@record_node("validate_summarized_data")
def validate_summarized_data_node(agent_state: AgentState, node_state: NodeState) -> AgentState:
    """
    Check that each section summary exists and meets basic criteria.
    Uses `validate_summaries` helper to compute per-section scores, availability,
    and an overall score. Updates agent_state.accuracy_score and validation_result_key.
    """
    # 1) Pull our summaries and the specs we used to generate them
    summaries = agent_state.memory.get("summaries", {})
    specs     = summarization_prompt_library

    # 2) Delegate to our validation utility
    result = validate_summaries(summaries, specs)

    # 3) Store the raw validation results
    agent_state.memory["summary_validation"] = result

    # 4) Update accuracy and decide pass/fail
    overall = result.get("overall_score", 0.0)
    agent_state.accuracy_score        = overall
    agent_state.validation_result_key = "valid" if overall >= 1.0 else "invalid"

    # 5) If any section failed, we end the pipeline
    if overall < 1.0:
        agent_state.memory["branch"] = "end"

    # 6) Attach detailed scores for observability
    node_state.custom_metrics["section_scores"]     = result.get("section_scores", {})
    node_state.custom_metrics["availability_score"] = result.get("availability_score", 0.0)

    return agent_state

@record_node("concept_analysis")
def concept_analysis_node(agent_state: AgentState, node_state: NodeState) -> AgentState:
    """
    Generate conceptual insights for each summary section using our shared helper.
    """
    executor = LangGraphLLMExecutor("concept")
    # Attach state so executor can track metrics
    executor.agent_state = agent_state
    executor.node_state  = node_state

    summaries = agent_state.memory.get("summaries", {})
    insights, count = generate_concept_insights(summaries, executor)

    agent_state.memory["concepts"] = insights
    node_state.custom_metrics["concepts_generated"] = count

    return agent_state

@record_node("validate_analyzed_data")
def validate_analyzed_data_node(agent_state: AgentState, node_state: NodeState) -> AgentState:
    """
    Validate each conceptual insight section:
      - Checks presence/non-emptiness via `validate_insights`
      - Computes overall score and per-section flags
      - On any failure, routes branch to 'end'
    """
    # 1) Grab the insights dict
    insights = agent_state.memory.get("concepts", {})

    # 2) Call our shared validation util
    result = validate_insights(insights)

    # 3) Persist raw validation results
    agent_state.memory["insight_validation"] = result

    # 4) Update accuracy & validation flag
    overall = result.get("overall_score", 0.0)
    agent_state.accuracy_score        = overall
    agent_state.validation_result_key = "valid" if overall >= 1.0 else "invalid"

    # 5) If any section failed, terminate the pipeline
    if overall < 1.0:
        agent_state.memory["branch"] = "end"

    # 6) Attach detailed metrics for observability
    node_state.custom_metrics["insight_section_scores"] = result.get("section_scores", {})
    node_state.custom_metrics["insight_availability"]   = result.get("availability_score", 0.0)

    return agent_state


@record_node("generate_report")
def generate_report_node(agent_state: AgentState, node_state: NodeState) -> AgentState:
    """
    Build final report sections from summaries using report_section_specs,
    write each section to disk, and record file paths and counts.
    """
    # 1) Prepare executor
    executor = LangGraphLLMExecutor("thesis")
    
    # Attach state references so helper can emit metrics
    executor.agent_state = agent_state
    executor.node_state  = node_state

    # 2) Generate all sections
    output_dir = Path(agent_state.work_dir) / "report_sections"
    sections, files = ReportLabUtils.build_annual_report(
        company_name=agent_state.company,
        summaries=agent_state.memory.get("summaries", {}),
        specs=report_section_specs,
        llm_executor=executor,
        output_dir=output_dir
    )

    # 3) Persist into state
    agent_state.memory["report_sections"] = sections
    agent_state.memory["report_files"]    = files

    # 4) Record files created
    for f in files.values():
        node_state.files_created.append(f)

    # Optionally track a custom metric
    node_state.custom_metrics["sections_generated"] = len(sections)

    return agent_state


@record_node("run_evaluation")
def run_evaluation_node(agent_state: AgentState, node_state: NodeState) -> AgentState:
    """
    Perform final pipeline evaluation: quantitative KPIs + qualitative LLM audit.
    Delegates logic to tools.evaluation_utils.evaluate_pipeline.
    """
    # Attach executor for qualitative call
    executor = LangGraphLLMExecutor("audit")
    node_state.llm_executor = executor

    # Delegate both quantitative and qualitative evaluation
    result = evaluate_pipeline(agent_state, node_state)

    # Store full evaluation
    agent_state.memory["final_evaluation"] = result

    # Optionally break out scores for easy access
    agent_state.accuracy_score = result["kpis"].get("total_pipeline_cost_usd", 0.0)  # or choose a different metric

    return agent_state