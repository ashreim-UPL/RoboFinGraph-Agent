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
from tools import report_utils
from functools import partial
from agents.state_types import NodeState, AgentState
from agents.state_types import TOOL_MAP
from tools.graph_tools import get_data_collection_tasks
from utils.config_utils import LangGraphLLMExecutor

logger = get_logger()

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

def data_collection_node(state: AgentState) -> Dict[str, Any]:
    print("--- Executing Node: Data Collection ---")
    ticker = state.company_details["identifiers"]["ticker"]
    year   = state.year
    filing_date = getattr(state, "filing_date", None) or f"{year}-12-31"
    raw_data_files = []
    error_log     = []

    available_args = {
        "ticker":        ticker,
        "ticker_symbol": ticker,
        "region":        state.region,
        "fyear":         year,
        "filing_date":   filing_date,
        "work_dir":      state.work_dir
    }

    for task in get_data_collection_tasks(state):
        name = task["task"]
        out  = task["file"]
        fn   = TOOL_MAP[name]
        try:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            sig = inspect.signature(fn)
            kwargs = {p: available_args[p] for p in sig.parameters if p in available_args}
            if "save_path" in sig.parameters:
                kwargs["save_path"] = out
            print(f"[DataCollection] {name}({kwargs}) -> {out}")
            fn(**kwargs)
            raw_data_files.append(out)
        except Exception as e:
            msg = f"{name}: {e}"
            logging.error(msg, exc_info=True)
            error_log.append(msg)

    return {"raw_data_files": raw_data_files, "error_log": error_log}

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
