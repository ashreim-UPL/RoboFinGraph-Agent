import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from functools import wraps
from datetime import datetime
from tabulate import tabulate
import sys
from utils.logger import get_logger, log_event
from langchain.schema import HumanMessage
import uuid
from agents.agent_utils import normalize_date

from utils.config_utils import LangGraphLLMExecutor
from agents.state_types import AgentState, NodeState, TOOL_MAP, NodeStatus
from tools.company_search import process_company_data
from tools.graph_tools import generate_concept_insights, validate_summaries, validate_insights, evaluate_pipeline, collect_us_financial_data, validate_raw_data
from pathlib import Path
from prompts.summarization_intsruction import summarization_prompt_library
from prompts.report_summaries import report_section_specs
from tools.report_writer import ReportLabUtils


# === Pipeline Metrics Recorder ===
def _record_metrics(agent_state: AgentState, node_key: str) -> None:
    """
    Append metrics from agent_state directly to memory['pipeline_data'].
    """
    metrics = {
        "node": node_key,
        "status": getattr(agent_state, "status", "unknown"),
        "duration": getattr(agent_state, "duration", None),
        "cost_llm": getattr(agent_state, "cost_llm", 0.0),
        "tokens_sent": getattr(agent_state, "tokens", {}).sent if hasattr(agent_state, "tokens") else 0,
        "tokens_generated": getattr(agent_state, "tokens", {}).generated if hasattr(agent_state, "tokens") else 0,
        "tools_used": list(getattr(agent_state, "tools_used", set())),
        "errors": list(getattr(agent_state, "errors", [])),
        "files_created": list(getattr(agent_state, "files_created", [])),
        "files_created_count": len(getattr(agent_state, "files_created", []))
    }
    # Ensure the pipeline_data list exists
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
        def wrapper(agent_state: AgentState, *args, **kwargs):
            start_time = datetime.now()

            # Emit start event for front-end
            start_event = {
                "event_type": "node_start",
                "data": {"node": node_key, "timestamp": start_time.isoformat()}
            }
            sys.stdout.write(json.dumps(start_event) + "\n")
            sys.stdout.flush()

            # Log start internally
            log_event("node_start", {"node": node_key, "timestamp": start_time.isoformat()})

            try:
                # Execute node function
                result = fn(agent_state, *args, **kwargs)
            except Exception as e:
                # Optional: capture to error_log if needed
                agent_state.error_log.append(str(e))
                raise

            end_time = datetime.now()
            agent_state.end_time = end_time
            agent_state.duration = (end_time - start_time).total_seconds()

            # Record metrics in memory
            _record_metrics(agent_state, node_key)
            # Emit end event
            metrics = agent_state.memory.get("pipeline_data", [])[-1]
            pipeline = agent_state.memory.setdefault("pipeline_data", [])
            pipeline.append(metrics)

            # Emit end event
            end_event = {"event_type": "node_end", "data": metrics}
            sys.stdout.write(json.dumps(end_event) + "\n")
            sys.stdout.flush()

            # Log end internally
            log_event("node_end", metrics)
            return result 
        return wrapper
    return decorator

# === Graph Node Implementations ===

@record_node("resolve_company")
def resolve_company_node(agent_state: AgentState) -> AgentState: 
    """
    Resolves company details using the specialized 'process_company_data' tool (LLM-only).
    This node updates agent_state with comprehensive company_details directly from the LLM.
    """
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
    except Exception as e:
        error_msg = f"resolve_company_node error: {e}"
        agent_state.error_log.append(error_msg) 
        agent_state.company_details = {"error": error_msg} 
        agent_state.validation_result_key = "Node_Error" 
        agent_state.accuracy_score = 0.0
    return agent_state


def llm_decision_node(agent_state: AgentState) -> AgentState:
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
            branch = "end"

    # 3) Persist both the “decision” and the explicit branch flag
    agent_state.llm_decision           = branch
    agent_state.memory["branch"]      = branch

    return agent_state

# not need probbaly.....
def data_collection_switch_node(agent_state: AgentState) -> AgentState:
    """Route pipeline based on LLM decision: 'us', 'india', or 'end'."""
    decision = (agent_state.llm_decision or "").strip().lower()
    if decision in ("us", "continue"):  branch = "us"
    elif decision == "india":             branch = "india"
    elif decision == "end":               branch = "end"
    else:                                  branch = "us"
    agent_state.memory["branch"] = branch
    return agent_state


def data_collection_us_node(state: AgentState) -> Dict[str, Any]:
    """
    High-level node to orchestrate US-specific data collection using a dedicated tool.
    """

    try:
        # Prepare the high-level arguments needed by the comprehensive tool
        # Ensure these match the parameters expected by collect_us_financial_data
        
        # Determine the primary ticker to pass to the collection tool
        primary_ticker = state.company_details.get("sec_ticker") or \
                         state.company_details.get("fmp_ticker") or \
                         state.company_details.get("yfinance_ticker") or \
                         state.company # Fallback to generic company name

        if not primary_ticker:
            raise ValueError("Could not determine a valid ticker for data collection.")
            
        # Get filing_date, with a robust fallback
        filing_date_val = getattr(state, "filing_date", None)
        if not filing_date_val:
            filing_date_val = f"{state.year}-12-31" # Default to end of year if not set
    
        # Call the comprehensive data collection tool
        collection_results = collect_us_financial_data(state)
  
        # Update the AgentState based on the results from the tool
        # Ensure raw_data_files and error_log are lists in AgentState
        updated_raw_data_files = list(state.raw_data_files) + collection_results["collected_files"]
        updated_error_log = list(state.error_log) + collection_results["errors"]

        messages = list(state.messages) # Start with existing messages
        for f in collection_results["collected_files"]:
            messages.append(HumanMessage(content=f"Successfully collected: {os.path.basename(f)}"))
        for err in collection_results["errors"]:
            messages.append(HumanMessage(content=f"Error during data collection: {err}"))
            
        errors_list = collection_results.get("errors", []) # Safely get 'errors', default to empty list if not present
        status = NodeStatus.COMPLETED.value if not errors_list else NodeStatus.PARTIAL_FAILURE.value

        # Return the updated state as a dictionary
        return {
            "raw_data_files": updated_raw_data_files,
            "error_log": updated_error_log,
            "messages": messages,
            "status": status
        }

    except Exception as e:
        return {
            "error_log": state.error_log + [f"Critical error in Data Collection US Node: {e}"],
            "status": NodeStatus.ERROR.value
        }

def data_collection_indian_node(agent_state: AgentState) -> AgentState:
    """
    Invoke all data‐collection tools (India); currently identical to US.
    """
    data = {}
    for tool_name, fn in TOOL_MAP.items():
        result = fn(agent_state.company)
        data[tool_name] = result
        #node_state.tools_used.add(tool_name)
    agent_state.memory["raw_data"] = data
    return agent_state


def validate_collected_data_node(agent_state: AgentState) -> AgentState:
    """
    Verify each expected raw file exists and is valid JSON via our shared tool.
    Compute accuracy = 1.0 only if all pass; otherwise accuracy = 0.0, flag 'invalid'
    and set branch to 'end'.
    """
    # 1. Build the list of expected files
    tasks = agent_state.get_data_collection_tasks()

    # 2. Delegate existence + JSON validity to helper
    result = validate_raw_data(tasks)
    
    # ----------------------------------

    # 3. Capture which files we actually read
    #for path in result["files_read"]:
    #    node_state.files_read.append(path)

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


def synchronize_data_node(agent_state: AgentState) -> AgentState:
    agent_state.memory["normalized_data"] = agent_state.memory.get("raw_data", {})
    return agent_state

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

    return agent_state

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
    #node_state.custom_metrics["section_scores"]     = result.get("section_scores", {})
    #node_state.custom_metrics["availability_score"] = result.get("availability_score", 0.0)

    return agent_state

def concept_analysis_node(agent_state: AgentState) -> AgentState:
    """
    Generate conceptual insights for each summary section using our shared helper.
    """
    executor = LangGraphLLMExecutor("concept")
    executor.agent_state = agent_state

    raw = agent_state.memory.get("summaries", {})
    insights, _ = generate_concept_insights(raw, executor, agent_state.company_details["company_name"])
    agent_state.memory["concepts"] = insights

    # write out only the eight canonical files
    sum_dir = Path(agent_state.summaries_dir)
    sum_dir.mkdir(parents=True, exist_ok=True)

    for section_key in report_section_specs:
        key = section_key.replace(".txt","")
        if key in insights:
            p = sum_dir / f"{key}.txt"
            p.write_text(insights[key], encoding="utf-8")
            agent_state.summary_files.append(str(p))

    return agent_state


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

    # 3) Persist raw validation results
    agent_state.memory["insight_validation"] = result

    # 4) Update accuracy & validation flag
    overall = result.get("overall_score", 0.0)
    agent_state.accuracy_score        = overall
    agent_state.validation_result_key = "valid" if overall >= 1.0 else "invalid"

    # 5) If any section failed, terminate the pipeline
    if overall < 1.0:
        agent_state.memory["branch"] = "end"

    print(result)
    input("inside validate_analyzed_data_node, enter to continue")
    # 6) Attach detailed metrics for observability
    #node_state.custom_metrics["insight_section_scores"] = result.get("section_scores", {})
    #node_state.custom_metrics["insight_availability"]   = result.get("availability_score", 0.0)

    return agent_state


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

    # 6) Record the PDF on state so you can inspect it
    agent_state.memory["final_report_path"]   = str(pdf_path)
    agent_state.memory["final_report_status"] = result

    return agent_state

def parse_scores(response_text):
    """
    Parse LLM response into a dictionary of ICAIF scores.
    Expected format in response_text:
    [Accuracy] Score: X
    [Logicality] Score: Y
    [Storytelling] Score: Z
    """
    scores = {}
    for line in response_text.splitlines():
        if "Accuracy" in line:
            try:
                scores["Accuracy"] = float(line.split(":")[1].strip())
            except:
                continue
        elif "Logicality" in line:
            try:
                scores["Logicality"] = float(line.split(":")[1].strip())
            except:
                continue
        elif "Storytelling" in line:
            try:
                scores["Storytelling"] = float(line.split(":")[1].strip())
            except:
                continue
    return scores

def run_evaluation_node(agent_state: AgentState) -> AgentState:
    """
    Perform final pipeline evaluation: quantitative KPIs + qualitative LLM audit.
    Enhanced to include ICAIF scoring and a pipeline stats matrix.
    """
    # 1) instantiate and attach the audit-model executor
    audit_executor = LangGraphLLMExecutor("audit")
    audit_executor.agent_state = agent_state

    # 2) delegate to utility, passing in the executor
    result = evaluate_pipeline(agent_state, audit_executor)

    # 3) report evaluation (ICAIF)
    report_text = agent_state.memory.get("generated_report_text", "")
    icaif_prompt = (
        "You are a financial analyst. Evaluate the following report on ICAIF criteria.\n"
        "[Accuracy], [Logicality], [Storytelling] — provide scores 0-10:\n\n"
        f"{report_text}"
    )

    messages = [HumanMessage(content=icaif_prompt)]
    response_msg = audit_executor.llm(messages)
    icaif_response = response_msg.content

    icaif_scores = parse_scores(icaif_response)

    # 4) pipeline matrix from agent_stats
    stats = getattr(agent_state, "agent_stats", {})
    pipeline_matrix = []
    for agent_name, stat in stats.items():
        pipeline_matrix.append([
            agent_name,
            stat.get("requests", 0),
            stat.get("avg_latency_ms", 0),
            stat.get("errors", 0)
        ])

    # 5) store results back on the state
    result["icaif_scores"] = icaif_scores
    result["pipeline_matrix"] = pipeline_matrix
    agent_state.memory["final_evaluation"] = result
    agent_state.accuracy_score = result["kpis"].get("total_pipeline_cost_usd", 0)

    # 6) print tables in a nice format
    print("### ICAIF Ratings ###")
    print(tabulate(icaif_scores.items(), headers=["Criterion", "Score"], tablefmt="github"))
    print("\n### Pipeline Matrix ###")
    print(tabulate(
        pipeline_matrix,
        headers=["Agent", "Requests", "Avg Latency (ms)", "Errors"],
        tablefmt="github"
    ))

    return agent_state