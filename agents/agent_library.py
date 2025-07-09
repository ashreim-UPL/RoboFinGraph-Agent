import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from functools import wraps
from datetime import datetime, timezone
from tabulate import tabulate
from collections import defaultdict
import re
import sys
from utils.logger import get_logger, log_event
from langchain.schema import HumanMessage
import uuid
from agents.agent_utils import normalize_date

from utils.config_utils import LangGraphLLMExecutor
from agents.state_types import AgentState, NodeState, TOOL_MAP, NodeStatus
from tools.company_search import process_company_data
from tools.graph_tools import generate_concept_insights, validate_summaries, validate_insights, evaluate_pipeline, collect_us_financial_data, validate_raw_data, collect_indian_financial_data
from pathlib import Path
from prompts.summarization_intsruction import summarization_prompt_library
from prompts.report_summaries import report_section_specs
from tools.report_writer import ReportLabUtils


# === Pipeline Metrics Recorder ===
def _record_metrics(agent_state: AgentState, node_key: str, start_time, end_time) -> None:
    """
    Append metrics from agent_state directly to memory['pipeline_data'].
    """
    metrics = {
        "node": node_key,
        "status": getattr(agent_state, "status", "unknown"),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration": (end_time - start_time).total_seconds(),
        "cost_llm": getattr(agent_state, "cost_llm", 0.0),
        "tokens_sent": getattr(agent_state, "tokens", {}).sent if hasattr(agent_state, "tokens") else 0,
        "tokens_generated": getattr(agent_state, "tokens", {}).generated if hasattr(agent_state, "tokens") else 0,
        "tools_used": list(getattr(agent_state, "tools_used", set())),
        "errors": list(getattr(agent_state, "errors", [])),
        "files_created": list(getattr(agent_state, "files_created", [])),
        "files_created_count": len(getattr(agent_state, "files_created", [])),
        # Optional extras:
        "agent_name": getattr(agent_state, "agent_name", None),
        "models_used": agent_state.memory.get("models_used", []),
        # Add summaries/hashes of input/output if desired
    }
    agent_state.memory.setdefault("pipeline_data", []).append(metrics)


# === Decorator for Node Functions ===

def record_node(node_key: str):
    def decorator(fn):
        @wraps(fn)
        def wrapper(agent_state: AgentState, *args, **kwargs):
            node_start = datetime.now(timezone.utc)
            # --- TRACK NODE SEQUENCE ---
            pipeline_nodes = agent_state.memory.setdefault("pipeline_nodes", [])
            if not pipeline_nodes or pipeline_nodes[-1] != node_key:
                pipeline_nodes.append(node_key)

            record = {
                "current_node": node_key,  
                "status": NodeStatus.RUNNING,
                "start_time": node_start.isoformat(),
                "end_time": None,
                "duration": None,
                "cost_llm": 0.0,
                "tokens_sent": 0,
                "tokens_generated": 0,
                "tools_used": [],
                "functions_used": [],
                "errors": [],
                "files_read": [],
                "files_created": [],
                "custom_metrics": {},
            }
            pipeline = agent_state.memory.setdefault("pipeline_data", [])
            # Set pipeline_start_time ONLY ONCE
            if not agent_state.memory.get("pipeline_start_time"):
                agent_state.memory["pipeline_start_time"] = node_start.isoformat()

            sys.stdout.write(json.dumps({
                "event_type": "node_start",
                "data": {"node": node_key, "timestamp": record["start_time"]}
            }) + "\n")
            sys.stdout.flush()

            try:
                setattr(agent_state, "_current_node_record", record)
                result = fn(agent_state, *args, **kwargs)
                record["status"] = NodeStatus.SUCCESS
            except Exception as e:
                record["status"] = NodeStatus.ERROR
                record["errors"].append(str(e))
                agent_state.error_log.append(f"{node_key}: {e}")
                raise
            finally:
                # finish timing
                node_end = datetime.now(timezone.utc)
                record["end_time"]   = node_end.isoformat()
                record["duration"]   = (node_end - node_start).total_seconds()
                agent_state.memory["pipeline_end_time"] = node_end.isoformat()

                # — Inject LLM KPIs only if this node actually ran an LLM call —
                models = agent_state.memory.get("models_used", [])
                if models:
                    last = models[-1]
                    # only pull in metrics if it’s our node
                    if last.get("agent") == node_key:
                        record["tokens_sent"]      = last.get("tokens_sent", 0)
                        record["tokens_generated"] = last.get("tokens_generated", 0)
                        record["cost_llm"]         = last.get("cost_llm", 0.0)
                        record["tools_used"]       = [f"{last.get('provider')}:{last.get('model')}"]


                # append & stream node_end
                pipeline.append(record)
                sys.stdout.write(json.dumps({
                    "event_type": "node_end",
                    "data": record
                }) + "\n")
                sys.stdout.flush()

            return result
        return wrapper
    return decorator

# === Graph Node Implementations ===

@record_node("Get Company Details")
def resolve_company_node(agent_state: AgentState) -> AgentState: 
    """
    Resolves company details using the specialized 'process_company_data' tool (LLM-only).
    This node updates agent_state with comprehensive company_details directly from the LLM.
    """
    model_start = datetime.now(timezone.utc)
    # 1. Get the company query from AgentState.
    company_query = agent_state.company 
    year  = agent_state.year

    if not company_query:
        error_msg = "Error: No company query provided in agent_state.company."
        agent_state.error_log.append(f"resolve_company_node: {error_msg}") 
        agent_state.company_details = {"error": error_msg} #
        agent_state.validation_result_key = "No_Company_Query" # New specific status
        agent_state.accuracy_score = 0
        return agent_state
    try:
        # 2. Call the  process_company_data function.
        result = process_company_data(company_query, year) 

        # <-- record that function call -->
        rec = getattr(agent_state, "_current_node_record", None)
        if rec is not None:
            rec.setdefault("functions_used", []).append("process_company_data")

        # 3. Update the agent_state with the LLM's results.
        data = result.get("llm_result", {}) if result else {}

        agent_state.company_details = data
        agent_state.region = data.get("region", agent_state.region) if data else agent_state.region
        raw_filing_date = (
            data.get(f"filing_date_{agent_state.year}") or data.get("filing_date") if data else None
        )
        agent_state.filing_date = normalize_date(raw_filing_date)        
        agent_state.validation_result_key = result.get("validation_status", "valid") if result else "Node_Error"
        agent_state.accuracy_score = result.get("accuracy_score", 0.0) if result else 0.0
        agent_state.sec_report_address= data.get("sec_report_address") if data else None

        if result and result.get("message"):
            agent_state.error_log.append(result["message"])

        official = data.get("official_name") if data else None
        if official and official.lower() != company_query.lower():
            agent_state.company = official

        model_end = datetime.now(timezone.utc)
        model_duration = (model_end - model_start).total_seconds()
        agent_state.memory.setdefault("models_used", []).append({
            "agent": "Get Company Details",
            "provider": "OpenAI",
            "model": "gpt-4o-search-preview-2025-03-11",
            "tokens_sent": result.get("tokens_sent", 0),
            "tokens_generated": result.get("tokens_generated", 0),
            "cost_llm": result.get("cost_llm", 0.0),
            "model_start_time": model_start.isoformat(),
            "model_end_time": model_end.isoformat(),
            "model_duration": model_duration, 
        })

    except Exception as e:
        error_msg = f"resolve_company_node error: {e}"
        agent_state.error_log.append(error_msg) 
        agent_state.company_details = {"error": error_msg} 
        agent_state.validation_result_key = "Node_Error" 
        agent_state.accuracy_score = 0.0
    return agent_state

@record_node("Check Region")
def region_decision_node(agent_state: AgentState) -> AgentState:
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


    if not region:
        branch = "end"
    else:
        region_norm = region.strip().lower()
        # 2) Map to one of the supported branches
        if region_norm in ("us", "usa", "united states"):
            branch = "us"
        elif region_norm in ("india", "in"):
            branch = "india"
        else:
            # If region is detected but not 'us' or 'india', route to 'end'
            agent_state.termination_reason = "Unsupported region"
            agent_state.error_log.append(f"Unsupported region: {region}")
            branch = "end"

    # 3) Persist both the “decision” and the explicit branch flag
    agent_state.llm_decision       = branch
    agent_state.memory["branch"]      = branch

    return agent_state

@record_node("US Data Collection")
def data_collection_us_node(agent_state: AgentState) -> Dict[str, Any]:
    """
    High-level node to orchestrate US-specific data collection using a dedicated tool.
    """

    try:
        # Prepare the high-level arguments needed by the comprehensive tool
        # Ensure these match the parameters expected by collect_us_financial_data
        
        # Determine the primary ticker to pass to the collection tool
        primary_ticker = agent_state.company_details.get("sec_ticker") or \
                         agent_state.company_details.get("fmp_ticker") or \
                         agent_state.company_details.get("yfinance_ticker") or \
                         agent_state.company # Fallback to generic company name

        if not primary_ticker:
            raise ValueError("Could not determine a valid ticker for data collection.")
            
        # Get filing_date, with a robust fallback
        filing_date_val = getattr(agent_state, "filing_date", None)
        if not filing_date_val:
            filing_date_val = f"{agent_state.year}-12-31" # Default to end of year if not set
    
        # Call the comprehensive data collection tool
        collection_results = collect_us_financial_data(agent_state)
        # <-- record that function call -->
        rec = getattr(agent_state, "_current_node_record", None)
        if rec is not None:
            rec.setdefault("functions_used", []).append("collect_us_financial_data")

        # Update the AgentState based on the results from the tool
        # Ensure raw_data_files and error_log are lists in AgentState
        updated_raw_data_files = list(agent_state.raw_data_files) + collection_results["collected_files"]
        updated_error_log = list(agent_state.error_log) + collection_results["errors"]

        messages = list(agent_state.messages) # Start with existing messages
        now = datetime.now(timezone.utc).isoformat()

        for f in collection_results["collected_files"]:
            agent_state.messages.append({
            "role":    "assistant",
            "content": f"Successfully collected: {os.path.basename(f)}",
            "ts":      now
            })

        for err in collection_results["errors"]:
            agent_state.messages.append({
            "role":    "assistant",
            "content": f"Error during data collection: {err}",
            "ts":      now
            })
            
        errors_list = collection_results.get("errors", []) # Safely get 'errors', default to empty list if not present
        status = NodeStatus.SUCCESS.value if not errors_list else NodeStatus.PARTIAL_FAILURE.value

        agent_state.llm_provider = "NA"
        agent_state.llm_model = "NA"

        # === LOG LLM/AGENT USAGE ===
        agent_state.memory.setdefault("models_used", []).append({
            "agent": "US collection",
            "provider": agent_state.llm_provider,
            "model": agent_state.llm_model,
            "tokens_sent": agent_state.tokens_sent,
            "tokens_generated": agent_state.tokens_generated,
            "cost_llm": agent_state.cost_llm,
        })
        
        # Return the updated state as a dictionary
        return {
            "raw_data_files": updated_raw_data_files,
            "error_log": updated_error_log,
            "messages": messages,
            "status": status
        }

    except Exception as e:
        return {
            "error_log": agent_state.error_log + [f"Critical error in Data Collection US Node: {e}"],
            "status": NodeStatus.ERROR.value
        }

@record_node("India Data Collection")
def data_collection_indian_node(agent_state: AgentState) -> AgentState:
    """
    Invoke all data‐collection tools (India); currently identical to US.
    """
    try:
        # Prepare the high-level arguments needed by the comprehensive tool
        # Ensure these match the parameters expected by collect_Indian_financial_data
        
        # Determine the primary ticker to pass to the collection tool
        primary_ticker = agent_state.company_details.get("sec_ticker") or \
                         agent_state.company_details.get("fmp_ticker") or \
                         agent_state.company_details.get("yfinance_ticker") or \
                         agent_state.company # Fallback to generic company name

        if not primary_ticker:
            raise ValueError("Could not determine a valid ticker for data collection.")
            
        # Get filing_date, with a robust fallback
        filing_date_val = getattr(agent_state, "filing_date", None)
        if not filing_date_val:
            filing_date_val = f"{agent_state.year}-12-31" # Default to end of year if not set
    
        # Call the comprehensive data collection tool
        collection_results = collect_indian_financial_data(agent_state)

        # <-- record that function call -->
        rec = getattr(agent_state, "_current_node_record", None)
        if rec is not None:
            rec.setdefault("functions_used", []).append("collect_indian_financial_data")

        # Update the AgentState based on the results from the tool
        # Ensure raw_data_files and error_log are lists in AgentState
        updated_raw_data_files = list(agent_state.raw_data_files) + collection_results["collected_files"]
        updated_error_log = list(agent_state.error_log) + collection_results["errors"]

        messages = list(agent_state.messages) # Start with existing messages
        now = datetime.now(timezone.utc).isoformat()

        for f in collection_results["collected_files"]:
            agent_state.messages.append({
            "role":    "assistant",
            "content": f"Successfully collected: {os.path.basename(f)}",
            "ts":      now
            })

        for err in collection_results["errors"]:
            agent_state.messages.append({
            "role":    "assistant",
            "content": f"Error during data collection: {err}",
            "ts":      now
            })

        errors_list = collection_results.get("errors", []) # Safely get 'errors', default to empty list if not present
        status = NodeStatus.SUCCESS.value if not errors_list else NodeStatus.PARTIAL_FAILURE.value

        agent_state.llm_provider = "NA"
        agent_state.llm_model = "NA"

        # === LOG LLM/AGENT USAGE ===
        agent_state.memory.setdefault("models_used", []).append({
            "agent": "IN collection",
            "provider": agent_state.llm_provider,
            "model": agent_state.llm_model,
            "tokens_sent": agent_state.tokens_sent,
            "tokens_generated": agent_state.tokens_generated,
            "cost_llm": agent_state.cost_llm,
        })
    
        # Return the updated state as a dictionary
        return {
            "raw_data_files": updated_raw_data_files,
            "error_log": updated_error_log,
            "messages": messages,
            "status": status
        }

    except Exception as e:
        return {
            "error_log": agent_state.error_log + [f"Critical error in Data Collection Indian Node: {e}"],
            "status": NodeStatus.ERROR.value
        }

@record_node("Validate Collected Data")
def validate_collected_data_node(agent_state: AgentState) -> AgentState:
    """
    Verify each expected raw file exists and is valid JSON via our shared tool.
    Compute accuracy = 1.0 only if all pass; otherwise accuracy = 0.0, flag 'invalid'
    and set branch to 'end'.
    """
    # 1. Build the list of expected files
    tasks = agent_state.get_data_collection_tasks()

    # <-- record that function call -->
    rec = getattr(agent_state, "_current_node_record", None)
    if rec is not None:
        rec.setdefault("functions_used", []).append("get_data_collection_tasks")

    # 2. Delegate existence + JSON validity to helper
    result = validate_raw_data(tasks)
 
    # <-- record that function call -->
    rec = getattr(agent_state, "_current_node_record", None)
    if rec is not None:
        rec.setdefault("functions_used", []).append("validate_raw_data")
    
    # ----------------------------------
    # 3. Capture which files we actually read and replace the collected log
    for path in result["files_read"]:
        agent_state.raw_data_files.append(path)

    # 4. Compute pass/fail
    total      = result["total"]
    valid_cnt  = result["valid_count"]
    if valid_cnt == total:
        agent_state.llm_decision       = "continue"
        agent_state.accuracy_score        = 1.0
        agent_state.validation_result_key = "valid"
    else:
        agent_state.llm_decision       = "end"
        agent_state.accuracy_score        = 0.0
        agent_state.validation_result_key = "invalid"

        missing = result.get("missing_files", [])
        corrupt = result.get("corrupt_files", [])
        reasons = []
        if missing: 
            reasons.append(f"missing: {missing}")
            agent_state.termination_reason = "Missing files"
        if corrupt: 
            reasons.append(f"corrupt: {corrupt}")
            agent_state.termination_reason = "Corrupt files"

        agent_state.error_log.append(
            f"validate_collected_data failed ({'; '.join(reasons)})"
        )
        # tell the orchestrator to stop
        agent_state.memory["branch"] = "end"

        agent_state.llm_provider = "NA"
        agent_state.llm_model = "NA"

        # === LOG LLM/AGENT USAGE ===
        agent_state.memory.setdefault("models_used", []).append({
            "agent": "collection validation",
            "provider": agent_state.llm_provider,
            "model": agent_state.llm_model,
            "tokens_sent": agent_state.tokens_sent,
            "tokens_generated": agent_state.tokens_generated,
            "cost_llm": agent_state.cost_llm,
        })

    return agent_state


def synchronize_data_node(agent_state: AgentState) -> AgentState:
    agent_state.memory["normalized_data"] = agent_state.memory.get("raw_data", {})
    return agent_state

@record_node("Summarize Data")
def summarization_node(agent_state: AgentState) -> AgentState:
    """
    1) For each required summary spec:
       - load the two input files (JSON or TXT) from raw_data_dir / filing_dir
       - fill in the prompt_template
       - call the LLM
    2) Save each output to preliminaries/<summary_name>.txt
    3) Track the file names in agent_state.preliminary_files
    4) Also stash the dict of <summary_name>->text in agent_state.memory["summaries"]
    """
    executor = LangGraphLLMExecutor("summarizer")
    executor.agent_state = agent_state
   
    required = [
        "analyze_income_stmt",
        "analyze_balance_sheet",
        "analyze_cash_flow",
        "analyze_segment_stmt",
        "risk_assessment",
        "competitors_analysis",
        "analyze_company_description",
        "analyze_business_highlights"
    ]

    prelim_dir = Path(agent_state.preliminary_dir)
    prelim_dir.mkdir(parents=True, exist_ok=True)

    summaries: Dict[str, str] = {}
    files: List[str] = []

    # map just the basename -> full path from data collection
    filename_to_path = {
        os.path.basename(task["file"]): task["file"]
        for task in agent_state.get_data_collection_tasks()
    }
    
    for key in required:
        spec = summarization_prompt_library[key]

        # build the injection dict for template
        injection: Dict[str, str] = {}
        for fname, var in spec["input_file_map"].items():
            fpath = filename_to_path.get(os.path.basename(fname))
            if not fpath:
                raise FileNotFoundError(f"Expected {fname} under raw/filings but none found")
            if fname.endswith(".json"):
                with open(fpath, "r", encoding="utf-8") as f:
                    content = json.dumps(json.load(f))
            else:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
            injection[var] = content

        prompt = spec["prompt_template"].format(**injection)

        # --- call the LLM ---
        resp = executor.generate(
            [HumanMessage(content=prompt)],
            agent_state,
        )
        text = resp.content.strip()

        # --- write out to preliminary_dir ---
        out_path = prelim_dir / f"{key}.txt"
        out_path.write_text(text, encoding="utf-8")

        summaries[key] = text
        files.append(str(out_path))

    # persist into state
    agent_state.memory["summaries"] = summaries
    agent_state.preliminary_files = files
    print(f"[DEBUG] Summarizer executor → provider={executor.provider}, model={executor.model_name}")
    print("agent state llm ", agent_state.llm_provider)
    print("agent state model ", agent_state.llm_model)
    print("llm cost", agent_state.cost_llm)
    print("llm tokens sent", agent_state.tokens_sent)
    print("llm tokens generated", agent_state.tokens_generated)

    input("model check")

    return agent_state

@record_node("Validate Summarized Data")
def validate_summarized_data_node(agent_state: AgentState) -> AgentState:
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

    # <-- record that function call -->
    rec = getattr(agent_state, "_current_node_record", None)
    if rec is not None:
        rec.setdefault("functions_used", []).append("validate_summaries")

    # 3) Store the raw validation results
    agent_state.memory["summary_validation"] = result

    # 4) Update accuracy and decide pass/fail
    overall = result.get("overall_score", 0.0)
    agent_state.accuracy_score        = overall
    agent_state.validation_result_key = "valid" if overall >= 1.0 else "invalid"
    agent_state.llm_decision       = "continue"


    # 5) If any section failed, we end the pipeline
    if overall < 1.0:
        agent_state.memory["branch"] = "end"
        agent_state.termination_reason = "Summary validation failed"
        agent_state.error_log.append("Summary validation failed")
        agent_state.llm_decision       = "end"

    agent_state.llm_provider = "None"
    agent_state.llm_model = "None"

    # === LOG LLM/AGENT USAGE ===
    agent_state.memory.setdefault("models_used", []).append({
        "agent": "summarization validation",
        "provider": agent_state.llm_provider,
        "model": agent_state.llm_model,
        "tokens_sent": agent_state.tokens_sent,
        "tokens_generated": agent_state.tokens_generated,
        "cost_llm": agent_state.cost_llm,
    })

    return agent_state

@record_node("Concept Analysis")
def concept_analysis_node(agent_state: AgentState) -> AgentState:
    """
    Generate conceptual insights for each summary section using our shared helper.
    """
    executor = LangGraphLLMExecutor("concept")
    executor.agent_state = agent_state

    raw = agent_state.memory.get("summaries", {})
    insights, _ = generate_concept_insights(raw, executor, agent_state.company_details["company_name"])
    agent_state.memory["concepts"] = insights

    # <-- record that function call -->
    rec = getattr(agent_state, "_current_node_record", None)
    if rec is not None:
        rec.setdefault("functions_used", []).append("generate_concept_insights")

    # write out only the eight canonical files
    sum_dir = Path(agent_state.summaries_dir)
    sum_dir.mkdir(parents=True, exist_ok=True)

    for section_key in report_section_specs:
        key = section_key.replace(".txt","")
        if key in insights:
            p = sum_dir / f"{key}.txt"
            p.write_text(insights[key], encoding="utf-8")
            agent_state.summary_files.append(str(p))

    # IF NOde is LLM we don't need this here- testing
    """agent_state.llm_provider = str(executor.provider)
    agent_state.llm_model = str(executor.model_name)

    # === LOG LLM/AGENT USAGE ===
    agent_state.memory.setdefault("models_used", []).append({
        "agent": "analysis",
        "provider": agent_state.llm_provider,
        "model": agent_state.llm_model,
        "tokens_sent": agent_state.tokens_sent,
        "tokens_generated": agent_state.tokens_generated,
        "cost_llm": agent_state.cost_llm,
    })"""

    return agent_state

@record_node("Validate Conceptual Insights")
def validate_analyzed_data_node(agent_state: AgentState) -> AgentState:
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

    # <-- record that function call -->
    rec = getattr(agent_state, "_current_node_record", None)
    if rec is not None:
        rec.setdefault("functions_used", []).append("validate_insights")

    # 3) Persist raw validation results
    agent_state.memory["insight_validation"] = result

    # 4) Update accuracy & validation flag
    overall = result.get("overall_score", 0.0)
    agent_state.accuracy_score        = overall
    agent_state.validation_result_key = "valid" if overall >= 1.0 else "invalid"
    agent_state.llm_decision       = "continue"

    # 5) If any section failed, terminate the pipeline
    if overall < 1.0:
        agent_state.memory["branch"] = "end"
        agent_state.termination_reason = "Insight validation failed"
        agent_state.error_log.append("Insight validation failed")
        agent_state.llm_decision       = "end"


    agent_state.llm_provider = "None"
    agent_state.llm_model = "None"

    # === LOG LLM/AGENT USAGE ===
    agent_state.memory.setdefault("models_used", []).append({
        "agent": "analysis validation",
        "provider": agent_state.llm_provider,
        "model": agent_state.llm_model,
        "tokens_sent": agent_state.tokens_sent,
        "tokens_generated": agent_state.tokens_generated,
        "cost_llm": agent_state.cost_llm,
    })

    return agent_state

@record_node("Generate Annual Report")
def generate_report_node(agent_state: AgentState) -> AgentState:
    """
    Build out individual report‐section text files from the in-memory summaries,
    using report_section_specs for prompts & templates, then record all file paths
    and a simple metric into agent_state.memory.
    """
    # 1) Prepare the output directory & PDF filename
    report_dir = Path(agent_state.work_dir) / "report_sections"
    report_dir.mkdir(parents=True, exist_ok=True)

    pdf_filename = f"{agent_state.company}_{agent_state.year}_annual_report.pdf"
    pdf_path = report_dir / pdf_filename

    # 2) Load the artifacts your report builder needs
    key_data_file    = Path(agent_state.summaries_dir) / "key_data.json"
    fin_metrics_file = Path(agent_state.summaries_dir) / "financial_metrics.json"
    key_data         = json.loads(key_data_file.read_text(encoding="utf-8"))
    financial_metrics= json.loads(fin_metrics_file.read_text(encoding="utf-8"))

    # 3) Point to the charts (these should already have been generated)
    chart_paths = {
        "share_performance":    str(Path(agent_state.summaries_dir) / "share_performance_chart.png"),
        "pe_eps_performance":   str(Path(agent_state.summaries_dir) / "pe_eps_chart.png"),
    }

    # 4) Pull in your LLM‐generated section texts & appendix summaries
    sections_map = agent_state.memory.get("concepts", {})

    # Compose the text for LLM evaluation (flatten all section texts)
    full_report_text = "\n\n".join(
        [f"## {k}\n{v}" for k, v in sections_map.items()]
    )
    agent_state.memory["generated_report_text"] = full_report_text
    
    # ←—— NEW: read every file in summaries_dir into a dict of filename→text
    appendix: Dict[str,str] = {}
    for f in Path(agent_state.summaries_dir).iterdir():
        if f.is_file() and f.suffix in (".txt", ".json"):
            appendix[f.name] = f.read_text(encoding="utf-8")

    # 5) Call your ReportLab builder with the exact signature
    result = ReportLabUtils.build_annual_report(
        ticker_symbol     = agent_state.company_details["sec_ticker"],
        filing_date       = agent_state.filing_date,
        output_pdf_path   = str(pdf_path),
        sections          = sections_map,
        key_data          = key_data,
        financial_metrics = financial_metrics,
        chart_paths       = chart_paths,
        summaries         = appendix,   # now a proper filename→content map
    )

    rec = getattr(agent_state, "_current_node_record", None)
    if rec is not None:
        rec.setdefault("functions_used", []).append("build_annual_report")

    # 6) Record the PDF on state so you can inspect it
    agent_state.memory["final_report_path"]   = str(pdf_path)
    agent_state.memory["final_report_status"] = result

    agent_state.llm_provider = "None"
    agent_state.llm_model = "None"

    # === LOG LLM/AGENT USAGE ===
    agent_state.memory.setdefault("models_used", []).append({
        "agent": "report generation",
        "provider": agent_state.llm_provider,
        "model": agent_state.llm_model,
        "tokens_sent": agent_state.tokens_sent,
        "tokens_generated": agent_state.tokens_generated,
        "cost_llm": agent_state.cost_llm,
    })
    return agent_state

def parse_icaif_scores(response: str) -> dict:
    """
    Extracts ICAIF scores from LLM output, supporting both section and table formats.
    Returns a dict with keys: 'accuracy', 'logicality', 'storytelling'.
    """
    scores = {}

    # 1. Try to find headers like: ## 1. Accuracy: **9/10**
    patterns = {
        "accuracy": r"Accuracy\W{0,10}(\d{1,2})\s*/\s*10",
        "logicality": r"Logicality\W{0,10}(\d{1,2})\s*/\s*10",
        "storytelling": r"Storytelling\W{0,10}(\d{1,2})\s*/\s*10"
    }

    # 2. Table format: | Accuracy | 9 | Justification |
    table_patterns = {
        "accuracy": r"\|\s*Accuracy\s*\|\s*(\d{1,2})\s*\|",
        "logicality": r"\|\s*Logicality\s*\|\s*(\d{1,2})\s*\|",
        "storytelling": r"\|\s*Storytelling\s*\|\s*(\d{1,2})\s*\|"
    }

    # First, try section header pattern
    for key, pat in patterns.items():
        m = re.search(pat, response, re.IGNORECASE)
        if m:
            scores[key] = int(m.group(1))
        else:
            # Try table format if section not found
            m2 = re.search(table_patterns[key], response, re.IGNORECASE)
            scores[key] = int(m2.group(1)) if m2 else None

    return scores
    
@record_node("Run Evaluation")
def run_evaluation_node(agent_state: AgentState) -> AgentState:
    """
    Perform final pipeline evaluation: quantitative KPIs + qualitative LLM audit.
    Enhanced to include ICAIF scoring and a pipeline stats matrix.
    """

    # === 1) Instantiate the audit LLM executor ===
    audit_executor = LangGraphLLMExecutor("audit")
    audit_executor.agent_state = agent_state

    # === 2) ICAIF LLM-based report scoring ===
    report_text = agent_state.memory.get("generated_report_text", "")
    icaif_prompt = (
        "You are a financial analyst. Evaluate the following report on ICAIF criteria.\n"
        "[Accuracy], [Logicality], [Storytelling] — provide scores 0-10:\n\n"
        f"{report_text}"
    )
    messages = [HumanMessage(content=icaif_prompt)]

    # Save prompt & response for debug...
    work_dir = getattr(agent_state, "work_dir", ".")
    os.makedirs(work_dir, exist_ok=True)
    (Path(work_dir) / "icaif_prompt_debug.txt").write_text(icaif_prompt, encoding="utf-8")

    icaif_response_msg = audit_executor.generate(messages, agent_state)
    icaif_response = icaif_response_msg.content
    (Path(work_dir) / "icaif_response_debug.txt").write_text(icaif_response, encoding="utf-8")

    # Parse ICAIF scores
    icaif_scores = parse_icaif_scores(icaif_response)
    (Path(work_dir) / "icaif_scores_debug.json").write_text(
        json.dumps(icaif_scores, indent=2), encoding="utf-8"
    )

    # === 3) Quantitative & Qualitative evaluation ===
    result          = evaluate_pipeline(agent_state, icaif_scores=icaif_scores)
    pipeline_steps  = result.get("pipeline_steps", [])
    global_provider = result["kpis"].get("main_llm_provider", "unknown")

    # --- record that we called evaluate_pipeline() ---
    rec = getattr(agent_state, "_current_node_record", None)
    if rec is not None:
        rec.setdefault("functions_used", []).append("evaluate_pipeline")

    # === 4) Build the Pipeline Matrix including Functions Used ===
    pipeline_matrix = []
    for step in pipeline_steps:
        node       = step.get("current_node", "unknown")
        latency    = round(step.get("duration", 0.0), 1)
        errors     = len(step.get("errors", []))

        # grab provider/model from tools_used
        tools = step.get("tools_used", [])
        if tools:
            provider, model = tools[0].split(":", 1)
        else:
            provider, model = global_provider, "unknown"

        # join the functions_used list into a string
        funcs = step.get("functions_used", [])
        funcs_str = ", ".join(funcs) if funcs else "[none]"

        pipeline_matrix.append([
            node,  # Agent
            1,     # Requests
            latency,
            errors,
            provider,
            model,
            funcs_str
        ])

    # === 5) Save back to state ===
    result["icaif_scores"]    = icaif_scores
    result["pipeline_matrix"] = pipeline_matrix
    agent_state.memory["final_evaluation"] = result
    agent_state.accuracy_score = result["kpis"].get("total_pipeline_cost_usd", 0.0)

    # === 6) Pretty-print to console ===
    if icaif_scores:
        print("### ICAIF Ratings ###")
        print(tabulate(
            icaif_scores.items(),
            headers=["Criterion", "Score"],
            tablefmt="github"
        ))
    else:
        print("### No ICAIF scores available ###")

    print("\n### Pipeline Matrix ###")
    print(tabulate(
        pipeline_matrix,
        headers=["Agent", "Requests", "Avg Latency (sec)", "Errors",
                 "Provider", "Model", "Functions Used"],
        tablefmt="github"
    ))

    # === 7) Persist to disk ===
    metrics_to_save = {
        "final_evaluation":   result,
        "icaif_scores":       icaif_scores,
        "pipeline_matrix":    pipeline_matrix,
        "pipeline_data":      pipeline_steps,
        "error_log":          agent_state.error_log,
    }
    out_path = Path(work_dir) / "pipeline_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics_to_save, f, indent=2, default=str)
    print(f"[✓] Pipeline metrics saved to {out_path}")

    return agent_state
