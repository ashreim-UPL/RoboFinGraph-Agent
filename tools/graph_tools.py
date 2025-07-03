import os
import json
import asyncio
import time
from pathlib import Path
import logging
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

def get_data_collection_tasks(state: AgentState) -> List[Dict[str, str]]:
    raw_dir = state.raw_data_dir
    filing_dir= state.filing_dir
    summaries_dir = state.summaries_dir
  
    # Assign the list to a variable first
    tasks_list = [
        # Individual entries for each SEC section (as per our last refactoring)
        {'task': 'get_sec_10k_section_1', 'file': os.path.join(filing_dir, 'sec_10k_section_1.txt')},
        {'task': 'get_sec_10k_section_1a', 'file': os.path.join(filing_dir, 'sec_10k_section_1a.txt')},
        {'task': 'get_sec_10k_section_7', 'file': os.path.join(filing_dir, 'sec_10k_section_7.txt')},
        
        # The rest of your tasks, which produce a single file
        {'task': 'get_key_data', 'file': os.path.join(raw_dir, 'key_data.json')},
        {'task': 'get_company_profile',  'file': os.path.join(raw_dir, 'company_profile.json')},
        {'task': 'get_competitors', 'file': os.path.join(raw_dir, 'competitors.json')},
        {'task': 'get_income_statement', 'file': os.path.join(raw_dir, 'income_statement.json')},
        {'task': 'get_balance_sheet', 'file': os.path.join(raw_dir, 'balance_sheet.json')},
        {'task': 'get_cash_flow',  'file': os.path.join(raw_dir, 'cash_flow.json')},
        {'task': 'get_pe_eps_chart', 'file': os.path.join(summaries_dir, 'pe_eps_performance.png')},
        {'task': 'get_share_performance_chart',  'file': os.path.join(summaries_dir, 'share_performance.png')},
        {'task': 'financial_metrics', 'file': os.path.join(summaries_dir, 'financial_metrics.json')}
    ]

    print(f"--- DEBUG: get_data_collection_tasks generated tasks: {tasks_list} ---\n")
    return tasks_list

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


logger = logging.getLogger(__name__)

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

    for task_def in get_data_collection_tasks(state):
        print(task_def)
    
        tool_name = task_def["task"]
        output_filename = task_def["file"]
        out_file_path = os.path.join(output_filename)
        
        
        print("Tool map", TOOL_MAP)
        if tool_name not in TOOL_MAP:          
            msg = f"Tool '{tool_name}' not found in TOOL_MAP. Skipping."
            logger.warning(msg)
            errors.append(msg)
            continue

        tool_fn = TOOL_MAP[tool_name]
        print("\n",tool_name, tool_fn,"\n")
        try:
            os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
            sig = inspect.signature(tool_fn)
            print(sig)
            input("check function signature")
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

def validate_raw_data(tasks: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Checks each file path in `tasks` for existence and valid non-empty JSON.
    Returns a dict with:
      - total: int
      - valid_count: int
      - missing_files: List[str]
      - corrupt_files: List[str]
      - files_read: List[str]
    """
    files_read: List[str] = []
    total = len(tasks)
    valid_count = 0
    missing_files: List[str] = []
    corrupt_files: List[str] = []

    for t in tasks:
        path = t["file"]
        files_read.append(path)
        if not os.path.exists(path):
            missing_files.append(path)
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data:
                valid_count += 1
            else:
                corrupt_files.append(path)
        except Exception:
            corrupt_files.append(path)

    return {
        "total": total,
        "valid_count": valid_count,
        "missing_files": missing_files,
        "corrupt_files": corrupt_files,
        "files_read": files_read
    }


# tools/summarizer.py

def summarize_sections(
    raw_data_dir: Path,
    filing_dir: Path,
    prompts: Dict[str, Any],
    file_key_map: Dict[str, str],
    llm_executor: Any
) -> Tuple[Dict[str, str], int]:
    """
    For each section in `prompts`:
      1. Load the required files (JSON from raw_data_dir, text from filing_dir)
      2. Map each filename into its template key via file_key_map
      3. Fill prompt_template with those pieces
      4. Call llm_executor.generate(...)
      5. Collect the response and increment a counter
    Returns:
      - summaries: Dict[section_key, LLM output]
      - sections_processed: int
    """
    summaries: Dict[str, str] = {}
    sections_processed = 0

    for section_key, spec in prompts.items():
        # 1) Gather inputs
        inputs: Dict[str, str] = {}
        for fname in spec["input_files"]:
            # Decide whether it's in raw_data_dir or filing_dir
            candidate = raw_data_dir / fname
            if not candidate.exists():
                candidate = filing_dir / Path(fname).name

            if candidate.suffix == ".json":
                data = json.load(candidate.open("r", encoding="utf-8"))
                # Insert as a JSON‐string
                inputs[file_key_map[fname]] = json.dumps(data)
            else:
                inputs[file_key_map[fname]] = candidate.read_text(encoding="utf-8")

        # 2) Build the prompt
        prompt = spec["prompt_template"].format(**inputs)

        # 3) Call the LLM
        response = llm_executor.generate(
            [HumanMessage(content=prompt)],
            llm_executor.agent_state,
            llm_executor.node_state
        )
        summaries[section_key] = response.content
        sections_processed += 1

    return summaries, sections_processed

def validate_raw_data(tasks: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Checks each file path in `tasks` for existence and valid non-empty JSON.
    Returns a dict with:
      - total: int
      - valid_count: int
      - missing_files: List[str]
      - corrupt_files: List[str]
      - files_read: List[str]
    """
    files_read: List[str] = []
    total = len(tasks)
    valid_count = 0
    missing_files: List[str] = []
    corrupt_files: List[str] = []

    for t in tasks:
        path = t["file"]
        files_read.append(path)
        if not os.path.exists(path):
            missing_files.append(path)
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data:
                valid_count += 1
            else:
                corrupt_files.append(path)
        except Exception:
            corrupt_files.append(path)

    return {
        "total": total,
        "valid_count": valid_count,
        "missing_files": missing_files,
        "corrupt_files": corrupt_files,
        "files_read": files_read
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
