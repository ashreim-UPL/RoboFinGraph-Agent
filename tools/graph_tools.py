import os
import json
from pathlib import Path
import logging
from PIL import Image
from datetime import datetime, timedelta
import inspect
from typing import Dict, Any, List, Callable, Tuple, Optional
from collections import Counter, defaultdict
from langchain.schema import HumanMessage
from tabulate import tabulate

from utils.logger import get_logger, log_event, log_agent_step
from utils.config_utils import LangGraphLLMExecutor
from prompts.summarization_intsruction import summarization_prompt_library
from prompts.report_summaries import report_section_specs
from agents.state_types import AgentState
from agents.state_types import TOOL_MAP, TOOL_IN_MAP
from tools.global_API_toolkit import get_10k_metadata

logger = get_logger()
logger = logging.getLogger(__name__)


# --- Node Functions ---
def get_sec_metadata_node(state: AgentState) -> Dict[str, str]:
    """
    Fetches the most recent 10-K filing date for the specified fiscal year,
    pulling its own params out of `state`.
    """
    print("---Fetching SEC Metadata---")
    try:
        sec_ticker = state.company_details['sec_ticker']
        report_year = int(state.year)
        filing_year = report_year
        start_date = f"{filing_year}-01-01"
        end_date   = f"{filing_year+1}-06-30"

        # call your real helper directly:
        metadata = get_10k_metadata(
            sec_ticker=sec_ticker,
            start_date=start_date,
            end_date=end_date
        )

        if not metadata or "error" in metadata:
            print(f"[!] SEC Metadata Error: {metadata.get('error','No filings found')}")
            return {}
        filing_date = metadata["filedAt"].split("T")[0]
        state.filing_date = filing_date
        return {"filing_date": filing_date}

    except Exception as e:
        print(f"[!] CRITICAL Error in get_sec_metadata_node: {e}")
        get_logger().error("get_sec_metadata_node failed", exc_info=True)
        return {}

# ------------------     Indian DATA COLLECTION NODE
def collect_indian_financial_data(state: AgentState) -> Dict[str, List[str]]:
    """
    Comprehensive tool to collect all required US financial data.
    Iterates through a predefined list of US-specific data collection tasks,
    prepares arguments, and calls the respective tools.
    Returns a dictionary of collected files and any errors encountered.
    """

    work_dir = state.work_dir
    company_name = state.company_details["company_name"]  
    company_ticker = state.company_details["fmp_ticker"]
    filing_date = state.filing_date
    fyear = state.year
    company_details = state.company_details
   
    collected_files = []
    errors = []

    base_raw_data_dir = os.path.join(work_dir, "raw_data")
    os.makedirs(base_raw_data_dir, exist_ok=True)

    for task_def in state.get_data_collection_tasks():
    
        tool_name = task_def["task"]
        output_filename = task_def["file"]
        out_file_path = os.path.join(output_filename)
        
        if tool_name not in TOOL_IN_MAP:     
            msg = f"Tool '{tool_name}' not found in TOOL_IN_MAP. Skipping."
            logger.warning(msg)
            errors.append(msg)
            continue

        tool_info = TOOL_IN_MAP[tool_name]
        tool_fn = tool_info["function"]
        try:
            os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
            sig = inspect.signature(tool_fn)
            kwargs = {}
            # Prepare arguments based on the tool's signature
            for p_name, param in sig.parameters.items():
                if p_name == "company_ticker":
                    kwargs[p_name] = company_ticker
                elif p_name == "ticker": # Some FMP tools might use 'ticker'
                    kwargs[p_name] = company_details.get("fmp_ticker", company_ticker)
                elif p_name == "sec_ticker":
                    kwargs[p_name] = company_details.get("sec_ticker", company_ticker)
                elif p_name == "sec_report_address":
                    kwargs[p_name] = state.sec_report_address
                elif p_name == "fyear":
                    kwargs[p_name] = int(fyear)
                elif p_name == "filing_date":
                    kwargs[p_name] = filing_date
                elif p_name == "save_path":
                    kwargs[p_name] = out_file_path

                elif p_name in ["work_dir", "company_details", "region"]: # Pass directly if needed
                    kwargs[p_name] = locals().get(p_name) # Access local variables
                elif param.default is inspect.Parameter.empty and p_name not in kwargs:
                    # If a required parameter is missing and no default, raise error
                    raise ValueError(f"Missing required argument for tool '{tool_name}': '{p_name}'")

            #logger.info(f"Calling tool: {tool_name} with kwargs: {kwargs}")
            tool_fn(**kwargs) # Execute the individual tool
            collected_files.append(out_file_path)

        except Exception as e:
            msg = f"Error running tool '{tool_name}' for {company_name} ({fyear}): {e}"
            logger.error(msg, exc_info=True)
            errors.append(msg)

    return {"collected_files": collected_files, "errors": errors}
# ------------------     US DATA COLLECTION NODE
def collect_us_financial_data(state: AgentState) -> Dict[str, List[str]]:
    """
    Comprehensive tool to collect all required US financial data.
    Iterates through a predefined list of US-specific data collection tasks,
    prepares arguments, and calls the respective tools.
    Returns a dictionary of collected files and any errors encountered.
    """

    work_dir = state.work_dir
    company_name = state.company_details["company_name"]  
    company_ticker = state.company_details["fmp_ticker"]
    filing_date = state.filing_date
    fyear = state.year
    company_details = state.company_details
   
    collected_files = []
    errors = []

    # if fillign date is not availabe get it from sec_meta_data
    if  not filing_date:
        get_sec_metadata_node(state)
        filing_date = state.filing_date


    base_raw_data_dir = os.path.join(work_dir, "raw_data")
    os.makedirs(base_raw_data_dir, exist_ok=True)

    for task_def in state.get_data_collection_tasks():
    
        tool_name = task_def["task"]
        output_filename = task_def["file"]
        out_file_path = os.path.join(output_filename)
           
        if tool_name not in TOOL_MAP:          
            msg = f"Tool '{tool_name}' not found in TOOL_MAP. Skipping."
            logger.warning(msg)
            errors.append(msg)
            continue

        tool_info = TOOL_MAP[tool_name]
        tool_fn = tool_info["function"]
        try:
            os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
            sig = inspect.signature(tool_fn)
            kwargs = {}

            # Prepare arguments based on the tool's signature
            for p_name, param in sig.parameters.items():
                if p_name == "company_ticker":
                    kwargs[p_name] = company_ticker
                elif p_name == "ticker": # Some FMP tools might use 'ticker'
                    kwargs[p_name] = company_details.get("fmp_ticker", company_ticker)
                elif p_name == "sec_ticker":
                    kwargs[p_name] = company_details.get("sec_ticker", company_ticker)
                elif p_name == "sec_report_address":
                    kwargs[p_name] = state.sec_report_address
                elif p_name == "fyear":
                    kwargs[p_name] = fyear
                elif p_name == "filing_date":
                    kwargs[p_name] = filing_date
                elif p_name == "save_path":
                    kwargs[p_name] = out_file_path

                elif p_name in ["work_dir", "company_details", "region"]: # Pass directly if needed
                    kwargs[p_name] = locals().get(p_name) # Access local variables
                elif param.default is inspect.Parameter.empty and p_name not in kwargs:
                    # If a required parameter is missing and no default, raise error
                    raise ValueError(f"Missing required argument for tool '{tool_name}': '{p_name}'")

            #logger.info(f"Calling tool: {tool_name} with kwargs: {kwargs}")
            tool_fn(**kwargs) # Execute the individual tool
            collected_files.append(out_file_path)

        except Exception as e:
            msg = f"Error running tool '{tool_name}' for {company_name} ({fyear}): {e}"
            logger.error(msg, exc_info=True)
            errors.append(msg)

    return {"collected_files": collected_files, "errors": errors}

# ------------------------- File  Validation Function Begin ------------------
def validate_raw_data(tasks: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Checks each file path in `tasks` for existence and basic validity:
      - .json: valid, non-empty JSON
      - .txt: non-empty text file
      - .png: valid image (can be opened by PIL)
    Returns a dict with:
      - total: int
      - valid_count: int
      - missing_files: List[str]
      - corrupt_files: List[str]
      - files_read: List[str]
      - file_results: Dict[str, str]  # optional: {path: status}
    """
    files_read: List[str] = []
    total = len(tasks)
    valid_count = 0
    missing_files: List[str] = []
    corrupt_files: List[str] = []
    file_results: Dict[str, str] = {}

    for t in tasks:
        path = t["file"]
        files_read.append(path)
        ext = os.path.splitext(path)[1].lower()
        if not os.path.exists(path):
            missing_files.append(path)
            file_results[path] = "missing"
            continue

        try:
            if ext == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data:
                    valid_count += 1
                    file_results[path] = "valid"
                else:
                    corrupt_files.append(path)
                    file_results[path] = "corrupt (empty JSON)"
            elif ext == ".txt":
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                if text.strip():  # non-empty after stripping whitespace
                    valid_count += 1
                    file_results[path] = "valid"
                else:
                    corrupt_files.append(path)
                    file_results[path] = "corrupt (empty text)"
            elif ext == ".png":
                try:
                    with Image.open(path) as img:
                        img.verify() 
                    valid_count += 1
                    file_results[path] = "valid"
                except Exception as e:
                    corrupt_files.append(path)
                    file_results[path] = f"corrupt (bad image: {e})"
            else:
                # Unknown type, just check non-empty file
                if os.path.getsize(path) > 0:
                    valid_count += 1
                    file_results[path] = "valid (unknown type)"
                else:
                    corrupt_files.append(path)
                    file_results[path] = "corrupt (empty, unknown type)"
        except Exception as e:
            corrupt_files.append(path)
            file_results[path] = f"corrupt (exception: {e})"

    return {
        "total": total,
        "valid_count": valid_count,
        "missing_files": missing_files,
        "corrupt_files": corrupt_files,
        "files_read": files_read,
        "file_results": file_results,
    }
# ------------------------- File  Validation Function END ------------------

# tools/summarizer.py

def get_summary_task_details(summary_name: str, prompt_library: dict) -> dict:
    """
    Given a summary_name (e.g. 'analyze_income_stmt'), return:
        - output_file: (suggested filename or key)
        - input_files: list of required filenames (from prompt_library)
        - prompt_template: instruction template
    """
    if summary_name not in prompt_library:
        raise ValueError(f"Unknown summary task: {summary_name}")

    spec = prompt_library[summary_name]
    # Suggest output file name as <summary_name>.txt if not defined
    output_file = spec.get("output_file", f"{summary_name}.txt")
    input_files = spec["input_files"]
    prompt_template = spec["prompt_template"]
    return {
        "output_file": output_file,
        "input_files": input_files,
        "prompt_template": prompt_template,
    }

def validate_summaries(
    summaries: Dict[str, str],
    specs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validates summaries against the prompt specs by checking availability and
    basic non-emptiness. Computes an availability score and per-section flag.

    Returns a dict with:
      - section_scores: Dict[str, float] (1.0 if summary present, else 0.0)
      - availability_score: float (ratio of valid sections)
      - overall_score: float (same as availability_score, placeholder for extension)
    """
    expected = len(specs)
    section_scores: Dict[str, float] = {}
    valid_count = 0

    for key in specs:
        text = summaries.get(key, "").strip()
        score = 1.0 if text else 0.0
        section_scores[key] = score
        if score > 0:
            valid_count += 1

    availability_score = valid_count / expected if expected > 0 else 0.0
    overall_score = availability_score

    return {
        "section_scores": section_scores,
        "availability_score": availability_score,
        "overall_score": overall_score
    }

def generate_report_sections(
    company_name: str,
    summaries: Dict[str, str],
    specs: Dict[str, Any],
    llm_executor: Any,
    output_dir: Path
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    For each section spec, formats the prompt, calls the LLM via llm_executor,
    writes the output to a file under output_dir, and returns:
      - sections: mapping section key -> text
      - files: mapping section key -> file path (as str)
    """
    sections: Dict[str, str] = {}
    files: Dict[str, str] = {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, spec in specs.items():
        # 1) Gather the relevant summary snippets
        parts = [summaries.get(src, "") for src in spec["sources"]]
        combined = "\n\n".join(parts)

        # 2) Build prompt
        prompt = spec["prompt_template"].format(
            company_name=company_name,
            sections=combined
        )

        # 3) Call LLM
        resp = llm_executor.generate(
            [HumanMessage(content=prompt)],
            llm_executor.agent_state,
        )
        text = resp.content
        sections[filename] = text

        # 4) Write to file
        path = output_dir / filename
        path.write_text(text, encoding="utf-8")
        files[filename] = str(path)

    return sections, files

def generate_concept_insights(
    raw_summaries: Dict[str, str],
    llm_executor: Any,
    company_name: str
) -> Tuple[Dict[str, str], int]:
    insights: Dict[str, str] = {}
    count = 0

    for filename, spec in report_section_specs.items():
        # strip “.txt” to get our canonical key
        section_key = filename.replace(".txt", "")

        # gather the raw summaries this section depends on
        parts = []
        for src in spec["sources"]:
            text = raw_summaries.get(src)
            if text:
                parts.append(text)
        if not parts:
            continue

        # join them into one block
        joined = "\n\n".join(parts)

        # fill the template
        prompt = spec["prompt_template"].format(
            company_name=company_name,
            sections=joined
        )

        # call the LLM
        response = llm_executor.generate(
            [HumanMessage(content=prompt)],
            llm_executor.agent_state
        )

        insights[section_key] = response.content
        count += 1

    return insights, count


def validate_insights(insights: Dict[str, str]) -> Dict[str, Any]:
    """
    Validates conceptual insights by checking presence and non-emptiness.
    Returns a dict with:
      - section_scores: mapping each key to 1.0 or 0.0
      - availability_score: ratio of valid insights
      - overall_score: same as availability_score
    """
    total = len(insights)
    section_scores: Dict[str, float] = {}
    valid_count = 0

    for key, text in insights.items():
        score = 1.0 if text and text.strip() else 0.0
        section_scores[key] = score
        if score > 0:
            valid_count += 1

    availability_score = valid_count / total if total > 0 else 0.0
    overall_score = availability_score

    return {
        "section_scores": section_scores,
        "availability_score": availability_score,
        "overall_score": overall_score
    }

def print_pipeline_kpis(pipeline_data):
    """
    Given a list of per-node dicts (pipeline_data), prints concise tables:
    Node Status, Duration & Cost, Tools, Functions, Errors/File I/O.
    """
    if not pipeline_data:
        print("\n[No pipeline data available to display]\n", flush=True)
        return

    def short_ts(ts):
        try:
            return datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return ts

    # 1) Node Status
    status_cols = ["current_node", "status"]
    status_table = [
        {col: row.get(col, "-") for col in status_cols}
        for row in pipeline_data
    ]
    print("\n=== Node Status ===", flush=True)
    print(tabulate(status_table, headers="keys", tablefmt="plain"), flush=True)

    # 2) Duration & Cost Metrics
    time_cost_table = [
        {
            "current_node":     row.get("current_node", "-"),
            "start_time":       short_ts(row.get("start_time", "")),
            "end_time":         short_ts(row.get("end_time", "")),
            "duration":         row.get("duration", "-"),
            "cost_llm":         row.get("cost_llm", "-"),
            "tokens_sent":      row.get("tokens_sent", "-"),
            "tokens_generated": row.get("tokens_generated", "-"),
        }
        for row in pipeline_data
    ]

    print("\n=== Duration & Cost Metrics ===", flush=True)
    print(tabulate(time_cost_table, headers="keys", tablefmt="plain"), flush=True)

    # 3) Tools Used
    tool_table = [
        {
            "current_node": row.get("current_node", "-"),
            "LLMs_used":   ", ".join(list(dict.fromkeys(row.get("tools_used", [])))) or "-",
        }
        for row in pipeline_data
    ]
    print("\n=== LLMs ===")
    print(tabulate(tool_table, headers="keys", tablefmt="plain"))

    # 4) Functions Used
    function_table = [
        {
            "current_node": row.get("current_node", "-"),
            "functions_used": ", ".join(row.get("functions_used", [])) or "-",
        }
        for row in pipeline_data
    ]
    print("\n=== Functions ===")
    print(tabulate(function_table, headers="keys", tablefmt="plain"))

    # 5) Errors & File I/O
    io_table = [
        {
            "current_node":  row.get("current_node", "-"),
            "errors":        ", ".join(row.get("errors", [])) or "-",
            "files_read":    ", ".join(row.get("files_read", [])) or "-",
            "files_created": ", ".join(row.get("files_created", [])) or "-",
        }
        for row in pipeline_data
    ]
    print("\n=== Errors & File I/O ===")
    print(tabulate(io_table, headers="keys", tablefmt="plain"))

def evaluate_pipeline(agent_state: AgentState, icaif_scores: Optional[dict] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluates the pipeline execution and generates quantitative and qualitative metrics,
    including LLM KPIs, per-step analysis, and ICAIF scores (if provided).
    Returns:
      - results: dictionary of all evaluation metrics and pipeline data
      - metrics_to_save: streamlined dict suitable for persistence to disk
    """

    # --- Extract pipeline state ---
    raw_steps      = agent_state.memory.get("pipeline_data", []) or []
    models_used    = agent_state.memory.get("models_used", []) or []
    nodes_sequence = agent_state.memory.get("pipeline_nodes", []) or []
    ps = agent_state.memory.get("pipeline_start_time")
    pe = agent_state.memory.get("pipeline_end_time")

    # --- LLM KPIs ---
    valid_calls = [m for m in models_used if m.get("provider") and m.get("model")]
    provider_counts = Counter(m["provider"] for m in valid_calls)
    main_provider   = provider_counts.most_common(1)[0][0] if provider_counts else None
    unique_models   = sorted({m["model"] for m in valid_calls})
    total_tokens_in  = sum(m.get("tokens_sent", 0) for m in valid_calls)
    total_tokens_out = sum(m.get("tokens_generated", 0) for m in valid_calls)
    total_cost       = sum(m.get("cost_llm", 0.0) for m in valid_calls)

    # --- Pipeline-level metrics ---
    total_latency = sum(s.get("duration", 0.0) for s in raw_steps if isinstance(s, dict))
    total_retries = sum(s.get("retry_count", 0) for s in raw_steps if isinstance(s, dict))
    try:
        pipeline_duration = (datetime.fromisoformat(pe) - datetime.fromisoformat(ps)).total_seconds()
    except Exception:
        pipeline_duration = None

    # --- Per-node tool aggregation (for summary use only) ---
    grouped_tools = defaultdict(list)
    for s in raw_steps:
        if not isinstance(s, dict):
            continue
        node = s.get("current_node") or s.get("node")
        for t in s.get("tools_used", []):
            grouped_tools[node].append(t)
    model_steps = {node: sorted(set(grouped_tools[node])) for node in grouped_tools}

    # --- KPIs dictionary ---
    kpis = {
        "total_pipeline_cost_usd":     round(total_cost, 4),
        "total_pipeline_latency_sec":  round(total_latency, 2),
        "total_retries":               total_retries,
        "pipeline_nodes_sequence":     nodes_sequence,
        "pipeline_start_time":         ps,
        "pipeline_end_time":           pe,
        "pipeline_total_duration_sec": round(pipeline_duration, 2) if pipeline_duration is not None else None,
        "model_steps":                 model_steps,
        "main_llm_provider":           main_provider,
        "unique_llm_models":           unique_models,
        "total_tokens_sent":           total_tokens_in,
        "total_tokens_generated":      total_tokens_out,
    }
    if icaif_scores:
        kpis.update({
            "icaif_accuracy":     icaif_scores.get("accuracy"),
            "icaif_logicality":   icaif_scores.get("logicality"),
            "icaif_storytelling": icaif_scores.get("storytelling"),
            "icaif_scores":       icaif_scores,
        })
        print("### ICAIF Ratings ###")
        print(tabulate(
            icaif_scores.items(),
            headers=["Criterion", "Score"],
            tablefmt="github"
        ))
    else:
        print("### No ICAIF scores available ###")

    # --- Build the pipeline_matrix (per step) ---
    pipeline_matrix = []
    cumulative_cost = 0.0
    for s in raw_steps:
        if not isinstance(s, dict):
            continue
        node    = s.get("current_node") or s.get("node")
        requests= 1
        latency = round(s.get("duration", 0.0), 2)
        errors  = len(s.get("errors", []))
        tools   = s.get("tools_used", [])
        if tools:
            first = tools[0].split(":", 1)
            provider, model = first[0], first[1] if len(first) > 1 else ""
        else:
            provider, model = "unknown", "unknown"
        # Get the step LLM cost
        step_cost = s.get("cost_llm", 0.0) or 0.0
        cumulative_cost += step_cost
        # Use dict instead of list!
        pipeline_matrix.append({
            "Agent": node,
            "Requests": requests,
            "Avg Latency (sec)": latency,
            "Errors": errors,
            "Provider": provider,
            "Model": model,
            "Step LLM Cost (USD)": round(step_cost, 6),
            "Cumulative LLM Cost (USD)": round(cumulative_cost, 6),
            # Optionally add more fields as needed
        })
        s.setdefault("retry_count", 0)


    #print_pipeline_kpis(pipeline_matrix)

    # --- Register function usage for orchestration traceability ---
    rec = getattr(agent_state, "_current_node_record", None)
    if rec is not None:
        rec.setdefault("functions_used", []).append("evaluate_pipeline")

    # --- Collect result and persist structure ---
    results = {
        "kpis": kpis,
        "pipeline_steps": raw_steps,
        "pipeline_matrix": pipeline_matrix,
        "icaif_scores": icaif_scores,
    }
    metrics_to_save = {
        "final_evaluation":   results,
        "icaif_scores":       icaif_scores,
        "pipeline_matrix":    pipeline_matrix,
        "pipeline_data":      raw_steps,
        "error_log":          agent_state.error_log,
    }
    # (Persist to disk logic should be outside this function for separation of concerns)

    # --- Set state outputs (for chaining/side effects) ---
    agent_state.memory["final_evaluation"] = results
    agent_state.accuracy_score = kpis.get("total_pipeline_cost_usd", 0.0)

    log_event(
        event_type="pipeline_end",
        payload={
            "latency": pipeline_duration,
            "tokens": total_tokens_in + total_tokens_out,
            "cost": round(cumulative_cost, 6),
            "accuracy": icaif_scores.get("accuracy"),
            "logicality": icaif_scores.get("logicality"),
            "storytelling": icaif_scores.get("storytelling"),
        }
    )


    return results, metrics_to_save
