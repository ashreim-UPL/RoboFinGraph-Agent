import os
import json
import asyncio
import time
from pathlib import Path
import logging
from PIL import Image
import sys
from datetime import datetime, timedelta
import inspect
from typing import Dict, Any, List, Callable, Tuple
from langchain.schema import HumanMessage

from tools.file_utils import check_file_stats
from tools.global_API_toolkit import get_10k_section, get_10k_metadata
from utils.logger import get_logger, log_event, log_agent_step, log_cost_estimate
from prompts.summarization_intsruction import summarization_prompt_library
from prompts.report_summaries import report_section_specs
from functools import partial
from agents.state_types import NodeState, AgentState
from agents.state_types import TOOL_MAP



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
        ticker = state.company_details['identifiers']['ticker']
        report_year = int(state.year)
        filing_year = report_year + 1
        start_date = f"{filing_year}-01-01"
        end_date   = f"{filing_year}-05-31"

        # call your real helper directly:
        from tools.global_API_toolkit import get_10k_metadata
        metadata = get_10k_metadata(
            ticker_symbol=ticker,
            fyear=filing_year,
            save_path =state.filing_files,
            start_date=start_date,
            end_date=end_date
        )

        if not metadata or "error" in metadata:
            print(f"[!] SEC Metadata Error: {metadata.get('error','No filings found')}")
            return {}
        filing_date = metadata["filedAt"].split("T")[0]
        return {"filing_date": filing_date}

    except Exception as e:
        print(f"[!] CRITICAL Error in get_sec_metadata_node: {e}")
        get_logger().error("get_sec_metadata_node failed", exc_info=True)
        return {}

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

    base_raw_data_dir = os.path.join(work_dir, "raw_data")
    os.makedirs(base_raw_data_dir, exist_ok=True)

    for task_def in state.get_data_collection_tasks():
        print(task_def)
    
        tool_name = task_def["task"]
        output_filename = task_def["file"]
        out_file_path = os.path.join(output_filename)
        
        
        if tool_name not in TOOL_MAP:          
            msg = f"Tool '{tool_name}' not found in TOOL_MAP. Skipping."
            logger.warning(msg)
            errors.append(msg)
            continue

        tool_fn = TOOL_MAP[tool_name]
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

            logger.info(f"Calling tool: {tool_name} with kwargs: {kwargs}")
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
            llm_executor.node_state
        )
        text = resp.content
        sections[filename] = text

        # 4) Write to file
        path = output_dir / filename
        path.write_text(text, encoding="utf-8")
        files[filename] = str(path)

    return sections, files

def generate_concept_insights(
    summaries: Dict[str, str],
    llm_executor: Any
) -> Tuple[Dict[str, str], int]:
    """
    For each summary section, generate a conceptual analysis insight.
    Returns a dict mapping section_key -> insight, and the count of sections.
    """
    insights: Dict[str, str] = {}
    count = 0

    for section_key, summary_text in summaries.items():
        if not summary_text:
            continue
        # Build a generic prompt for concept analysis
        prompt = (
            f"Perform a conceptual analysis of the following summary for '{section_key}'. "
            "Extract key themes, strategic implications, and potential risks or opportunities. "
            "Provide your answer in one concise paragraph.\n\n" + summary_text
        )
        # Call the LLM
        response = llm_executor.generate(
            [HumanMessage(content=prompt)],
            llm_executor.agent_state,
            llm_executor.node_state
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

def generate_report_node(
    state: AgentState,
    builder: Any
) -> Dict[str, Any]:
    pdf = os.path.join(state.work_dir, 'final_annual_report.pdf')
    result = builder.build_annual_report(
        ticker_symbol=state.company_details['identifiers']['ticker'],
        filing_date=state.year,
        output_pdf_path=pdf,
        sections=state.conceptual_sections,
        key_data=json.loads(state.summary_outputs.get('key_data','{}')),
        financial_metrics=json.loads(state.summary_outputs.get('financial_metrics','{}')),
        chart_paths=state.raw_data_files,
        summaries=state.summary_outputs
    )
    log_event('report_generated', {'pdf': pdf})
    return {'final_report_text': result, 'output_pdf': pdf}

def evaluate_pipeline(
    agent_state: AgentState,
    node_state: NodeState
) -> Dict[str, Any]:
    """
    Performs both quantitative and qualitative evaluation of the pipeline.
    - Quantitative KPIs: total cost, total latency, per-step models
    - Qualitative assessment: uses the 'audit' LLM via LangGraphLLMExecutor

    Returns a dict with 'kpis' and 'quality_assessment'.
    """
    # Quantitative
    steps = agent_state.memory.get("pipeline_data", [])
    total_cost = sum(s.get("cost_llm", 0.0) for s in steps)
    total_latency = sum(s.get("duration", 0.0) for s in steps)
    model_steps = {s["node"]: s.get("tools_used", []) for s in steps}

    kpis = {
        "total_pipeline_cost_usd": round(total_cost, 4),
        "total_pipeline_latency_sec": round(total_latency, 2),
        "model_steps": model_steps
    }

    # Qualitative: assemble context
    payload = {
        "initial_query": agent_state.user_input,
        "summaries": agent_state.memory.get("summaries"),
        "final_report": agent_state.memory.get("final_report")
    }
    # Use existing executor attached to node_state
    executor = getattr(node_state, "llm_executor", None)
    if not executor:
        raise ValueError("No LLM executor attached for evaluation")

    from langchain.schema import HumanMessage
    prompt = f"Evaluate the final report quality given context: {json.dumps(payload)}"
    resp = executor.generate([HumanMessage(content=prompt)], agent_state, node_state)
    qualitative = resp.content

    return {"kpis": kpis, "quality_assessment": qualitative}
