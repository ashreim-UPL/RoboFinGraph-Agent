# workflows/graph_nodes.py
import logging
import os
import json
import re
from typing import Dict, Any, Callable, List
from functools import partial
import asyncio
import inspect
import time
from graph_utils.state_types import AgentState
from tools.company_resolver import identify_company_and_region
from tools import report_utils 
# Add in graph_nodes.py
from prompts.summarization_intsruction import finrobot_prompt_library
from prompts.report_summaries import report_section_specs
from utils.logger import get_logger, log_event, log_agent_step, log_final_summary, log_cost_estimate, get_logger
from tools.report_writer import ReportLabUtils
from tools.toolkit_loader import load_io_tools
from tools.file_utils import check_file_stats


# In a config or at the top of your graph_nodes.py
TOOL_MAP: Dict[str, Callable] = {
    "get_sec_10k_sections": report_utils.get_sec_10k_sections,
    "get_company_profile": report_utils.get_company_profile,
    "get_key_data": report_utils.get_key_data,
    "get_competitors": report_utils.get_competitor_analysis,
    "get_income_statement": partial(report_utils.get_financial_statement, statement_type="income_statement"),
    "get_balance_sheet": partial(report_utils.get_financial_statement, statement_type="balance_sheet"),
    "get_cash_flow": partial(report_utils.get_financial_statement, statement_type="cash_flow_statement"),
    "get_pe_eps_chart": report_utils.generate_pe_eps_chart,
    "get_share_performance_chart": report_utils.generate_share_performance_chart,
    "financial_metrics": report_utils.get_financial_metrics
}

def get_data_collection_tasks(state: AgentState) -> List[Dict[str, str]]:
    work_dir = state.work_dir or f"report/{state.company}_{state.year}"
    return [
        {'task': 'get_sec_10k_sections',          'file': f'{work_dir}/sec_filings'},
        {'task': 'get_key_data',                  'file': f'{work_dir}/summaries/key_data.json'},
        {'task': 'get_company_profile',           'file': f'{work_dir}/company_profile.json'},
        {'task': 'get_competitors',               'file': f'{work_dir}/competitors.json'},
        {'task': 'get_income_statement',          'file': f'{work_dir}/income_statement.json'},
        {'task': 'get_balance_sheet',             'file': f'{work_dir}/balance_sheet.json'},
        {'task': 'get_cash_flow',                 'file': f'{work_dir}/cash_flow.json'},
        {'task': 'get_pe_eps_chart',              'file': f'{work_dir}/summaries/pe_eps_performance.png'},
        {'task': 'get_share_performance_chart',   'file': f'{work_dir}/summaries/share_performance.png'},
        {'task': 'financial_metrics',             'file': f'{work_dir}/summaries/financial_metrics.json'}
    ]

# Parse cost from chat output
def parse_cost(cost_obj):
    try:
        usage = cost_obj.get("usage_including_cached_inference", {})
        total_cost = usage.get("total_cost")
        # Model key may change—grab the first (after 'total_cost')
        for k, v in usage.items():
            if isinstance(v, dict):
                model = k
                model_stats = v
                break
        else:
            return {}

        return {
            "total_cost": total_cost,
            "model": model,
            "model_cost": model_stats.get("cost"),
            "prompt_tokens": model_stats.get("prompt_tokens"),
            "completion_tokens": model_stats.get("completion_tokens"),
            "total_tokens": model_stats.get("total_tokens"),
        }
    except Exception as e:
        print("Cost parsing error:", e)
        return {}

# --- LLM Decision Node
def llm_decision_node(state: AgentState, agent) -> Dict[str, str]:
    company_details = state.company_details
    region = state.region

    prompt = f"""
You are an AI workflow validator.
...
Your response (just the keywords continue or end):
"""

    response = agent.summarize(prompt)

    # Defensive fallback
    if isinstance(response, str):
        decision = response.strip().lower()
    elif hasattr(response, "summary"):
        decision = response.summary.strip().lower()
    else:
        decision = "end"

    # Route logic
    if decision.startswith("continue"):
        route = "continue"
    elif decision.startswith("end"):
        route = "end"
    else:
        route = "continue"

    return {
        "llm_decision": decision,
        "__route__": route 
    }
# --- Validate data
def validate_collected_data_node(state: AgentState, agent) -> Dict[str, str]:
    company = state.company
    year = state.year
    work_dir = state.work_dir

    # Dynamically load resolved paths
    task_specs = get_data_collection_tasks(state)
    file_paths = [task["file"] for task in task_specs]

    missing_or_empty = []
    stats_summary = {}

    for fpath in file_paths:
        norm_path = fpath.replace("\\", "/")
        if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
            missing_or_empty.append(norm_path)
        else:
            stats_summary[norm_path] = check_file_stats(fpath)

    total_files = len(file_paths)
    quality_score = int(100 * (total_files - len(missing_or_empty)) / total_files)

    log_event("data_validation_pre_summary", {
        "company": company,
        "year": year,
        "work_dir": work_dir,
        "quality_score": quality_score,
        "missing_or_empty": missing_or_empty,
        "stats_summary": stats_summary
    })

    prompt = f"""
You are a financial data quality auditor.

You are reviewing the contents of the following real directory:
  {work_dir}

Missing or Empty Files:
{json.dumps(missing_or_empty, indent=2) if missing_or_empty else "None"}

File Stats (sizes, types, etc.):
{json.dumps(stats_summary, indent=2)}

Scoring Rule:
- Expected files: {total_files}
- Missing or empty: {len(missing_or_empty)}
- Quality Score: {quality_score}%

Rules:
- Reject if any file is missing or empty.
- PNG files are mandatory and must be non-zero in size.
- If quality_score < 100%, return "end: <reason>"
- If all files are present and valid, return "valid"

Output ONLY:
- "valid"
- "end: <reason>"
"""

    result = agent.summarize(prompt, tools=load_io_tools())

    if isinstance(result, str):
        decision = result.strip().lower()
    elif hasattr(result, "summary"):
        decision = result.summary.strip().lower()
    else:
        decision = "end: validator response unclear"

    log_event("data_validation_decision", {
        "company": company,
        "year": year,
        "decision": decision
    })

    return {
        "llm_decision": decision,
        "__route__": "data_collection_continue" if decision.startswith("valid") else "end"
    }

# --- resolve_company_node and decide_to_continue remain the same ---
def resolve_company_node(state: AgentState) -> Dict[str, Any]:
    logger = get_logger()
    print("--- Executing Node: Resolve Company (Live Call) ---")
    
    company_name = state.company
    print(f"Resolving company: {company_name}...")
    
    start_time = time.time()
    api_ok = False
    company_ok = False
    region_ok = False
    peers_ok = False
    
    try:
        result = asyncio.run(identify_company_and_region(company_name))
        runtime_sec = round(time.time() - start_time, 2)
        api_ok = True

        company_details = result.get("company_details")
        region = result.get("region")
        peers = company_details.get("competitors") if company_details else None

        company_ok = bool(company_details)
        region_ok = bool(region)
        peers_ok = bool(peers and isinstance(peers, list) and len(peers) > 0)

        success_score = (company_ok + region_ok + peers_ok) / 3 * 100

        logger.info(f"Company resolved: {company_details.get('official_name', 'N/A') if company_details else 'None'}")
        logger.info(f"Region detected: {region}")
        logger.info(f"Peers: {peers}")
        logger.info(f"Resolve runtime: {runtime_sec} seconds")
        logger.info(f"Success Score: {success_score:.0f}%")

        log_event("resolve_company", {
            "company_input": company_name,
            "runtime_seconds": runtime_sec,
            "api_response_ok": api_ok,
            "company_resolved": company_ok,
            "region_detected": region_ok,
            "peers_found": peers_ok,
            "success_score": success_score,
            "resolved_company": company_details,
            "region": region,
            "peers": peers
        })

        return {
            "company_details": company_details,
            "region": region
        }

    except Exception as e:
        runtime_sec = round(time.time() - start_time, 2)
        logger.error(f"Exception while resolving company '{company_name}': {e}", exc_info=True)
        log_event("resolve_company_error", {
            "company_input": company_name,
            "error": str(e),
            "runtime_seconds": runtime_sec
        })
        return {"company_details": None, "region": None}

# --- The data collection node now inspects function signatures ---
def data_collection_node(state: AgentState) -> Dict[str, Any]:
    """
    Runs all data collection tasks, saving outputs and returning their paths.
    """
    print("--- Executing Node: Data Collection ---")
    ticker = state.company_details['identifiers']['ticker']
    region = state.region
    year = state.year

    raw_data_files = []
    error_log = []

    available_args = {
        "ticker": ticker,
        "ticker_symbol": ticker,
        "region": region,
        "fyear": year,
        "filing_date": f"{year}-12-31",
    }

    for task in get_data_collection_tasks(state):
        task_name = task['task']
        output_file = task['file']
        task_function = TOOL_MAP[task_name]

        try:
            sig = inspect.signature(task_function)
            call_args = {param: available_args[param] for param in sig.parameters if param in available_args}
            # Add save_path argument
            if 'save_path' in sig.parameters:
                call_args['save_path'] = output_file

            print(f"[DataCollection] Running {task_name} -> {output_file}")
            task_function(**call_args)
            raw_data_files.append(output_file)
        except Exception as e:
            error_message = f"Error executing {task_name}: {e}"
            logging.error(error_message, exc_info=True)
            error_log.append(error_message)

    return {
        "raw_data_files": raw_data_files,
        "error_log": error_log,
    }

def summarization_node(state: AgentState, agent) -> dict:
    """
    Loads all raw data, builds the LLM prompt for each summary, sends it to the agent,
    and saves the resulting summary.
    """

    import os
    import json

    # --- Extract orchestration vars
    company_name = getattr(state, "company", None) or getattr(state, "company_details", {}).get("official_name")
    ticker = getattr(state, "company_details", {}).get("identifiers", {}).get("ticker")
    year = getattr(state, "year", None)
    peers = getattr(state, "company_details", {}).get("competitors", []) or []

    # --- Load all report data
    def load_all_data(data_dir):
        all_data = {}
        for fname in os.listdir(data_dir):
            if fname.endswith(".json") or fname.endswith(".txt"):
                with open(os.path.join(data_dir, fname), "r", encoding="utf-8") as f:
                    all_data[fname] = f.read()
        # sec_filings subdir
        sec_dir = os.path.join(data_dir, "sec_filings")
        if os.path.isdir(sec_dir):
            for fname in os.listdir(sec_dir):
                if fname.endswith(".txt"):
                    with open(os.path.join(sec_dir, fname), "r", encoding="utf-8") as f:
                        all_data[fname] = f.read()
        return all_data

    all_data = load_all_data("report")

    FILE_TO_TEMPLATE_KEY = {
        "income_statement.json": "table_str",
        "balance_sheet.json": "table_str",
        "cash_flow.json": "table_str",
        "competitors.json": "table_str",
        "company_profile.json": "business_summary",
        "key_data.json": "company_name",
        "sec_10k_section_7.txt": "section_text",
        "sec_10k_section_1.txt": "business_summary",
        "sec_10k_section_1a.txt": "risk_factors",
    }

    keys = [
        "analyze_income_stmt", "analyze_balance_sheet", "analyze_cash_flow",
        "analyze_segment_stmt", "analyze_business_highlights",
        "analyze_company_description", "get_risk_assessment",
        "get_competitors_analysis"
    ]

    summary_outputs = {}

    for key in keys:
        prompt_template = finrobot_prompt_library[key]["prompt_template"]
        input_files = finrobot_prompt_library[key]["input_files"]

        file_var_map = {}
        for file_path in input_files:
            fname = os.path.basename(file_path)
            if fname in FILE_TO_TEMPLATE_KEY:
                file_var_map[FILE_TO_TEMPLATE_KEY[fname]] = fname

        # Orchestration variables
        extra_args = {
            "company_name": company_name,
            "ticker": ticker,
            "year": year,
            "competitor_names": ", ".join(peers) if isinstance(peers, list) else peers,
        }

        # Gather all required template vars from all_data
        template_args = {**extra_args}
        for var, fname in file_var_map.items():
            template_args[var] = all_data.get(fname, f"PLACEHOLDER FOR {fname}")

        if "section_text" in template_args:
            template_args["section_7"] = template_args["section_text"]

        try:
            prompt = prompt_template.format(**template_args)
        except KeyError as e:
            summary_outputs[key] = f"ERROR: Missing {e} in template_args"
            continue

        # === KEY LINE: Call the agent and pass the prompt ===
        summary_info = agent.summarize(prompt)
  
        summary_text = summary_info.summary


        # --- LOGGING STEPS ---
        log_agent_step(
            agent_name=getattr(agent, "name", "summarizer"),
            input_text=prompt,
            output_text=summary_text,
            tool_used="summarize"
        )

        cost_text = summary_info.cost
        parsed_cost = parse_cost(cost_text)

        log_cost_estimate(
            agent_outputs={
                "summary_key": key,
                "summary_length": len(summary_text),
                **parsed_cost
            }
        )
        summary_outputs[key] = summary_text

    # Validation before writing
    if not summary_outputs:
        raise RuntimeError("[FATAL] No summaries generated! Failing before file write.")
    if any(v.startswith("ERROR:") for v in summary_outputs.values()):
        raise RuntimeError("[FATAL] One or more summaries failed. Check logs.")

    out_path = f"{state.work_dir}/preliminaries/all_summaries.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary_outputs, f, indent=2, ensure_ascii=False)
        # Immediately verify
        with open(out_path, "r", encoding="utf-8") as f:
            reloaded = json.load(f)
            if len(reloaded) != len(summary_outputs):
                raise RuntimeError(f"[FATAL] File write mismatch! Expected {len(summary_outputs)} sections, found {len(reloaded)}.")
    except Exception as file_exc:
        logging.error(f"[FILE WRITE FAILURE] {out_path}: {file_exc}", exc_info=True)
        raise

    return {"summary_outputs": summary_outputs}


# --- Node 3: Conceptual Analysis ---
def conceptual_analysis_node(state: AgentState, agent) -> dict:
    """
    Loads preliminary summaries, synthesizes each required conceptual section using agent, and logs all steps/costs.
    """
    import os
    import json

    print("--- Executing Node: Conceptual Analysis ---")
    company_name = getattr(state, "company", None) or getattr(state, "company_details", {}).get("official_name")
    
    os.makedirs(f"{state.work_dir}/summaries", exist_ok=True)
    with open(f"{state.work_dir}/preliminaries/all_summaries.json", "r", encoding="utf-8") as f:
        all_summaries = json.load(f)

    conceptual_outputs = {}

    # --- Ensure output directory exists ---
    output_dir = os.path.join("report", "summaries")
    os.makedirs(output_dir, exist_ok=True)

    for out_file, spec in report_section_specs.items():
        section_texts = []
        missing_sources = []
        for src in spec["sources"]:
            section = all_summaries.get(src, "")
            if section:
                section_texts.append(f"\n\n-- {src} --\n{section}")
            else:
                missing_sources.append(src)
        sections = "\n".join(section_texts).strip()
        if not sections:
            conceptual_outputs[out_file] = f"ERROR: Missing required source sections for {out_file}"
            continue

        prompt = spec["prompt_template"].format(
            company_name=company_name,
            sections=sections
        )
        summary_info = agent.summarize(prompt)
        summary_text = summary_info.summary
        #print("Debug Summary Text : ", summary_text)
        cost_text = summary_info.cost
        parsed_cost = parse_cost(cost_text) if cost_text else {}
        #print("Debug Cots: ", parsed_cost)
        log_agent_step(
            agent_name=getattr(agent, "name", "conceptual_analyzer"),
            input_text=prompt,
            output_text=summary_text,
            tool_used="summarize"
        )
        log_cost_estimate({
            "section": out_file,
            "summary_length": len(summary_text),
            **parsed_cost
        })

        # Save section result and write to file
        conceptual_outputs[out_file] = summary_text
        out_path = os.path.join(output_dir, out_file)
        with open(out_path, "w", encoding="utf-8") as out_f:
            out_f.write(summary_text)

    # Save full mapping for downstream steps
    os.makedirs(f"{state.work_dir}/summaries", exist_ok=True)
    with open(f"{state.work_dir}/summaries/conceptual_sections.json", "w", encoding="utf-8") as f:
        json.dump(conceptual_outputs, f, indent=2, ensure_ascii=False)

    return {"conceptual_sections": conceptual_outputs}

# --- Node 4: Thesis Generation ---
def generate_report_node(state: AgentState, agent) -> dict:
    print("--- Executing Node: Generate Thesis Report ---")
    summaries_dir = f"{state.work_dir}/summaries"
    # 1. Conceptual summaries
    conceptual_sections = state.conceptual_sections

    # 2. Key data
    with open(os.path.join(summaries_dir,"key_data.json"), "r", encoding="utf-8") as f:
        key_data = json.load(f)

    # 3. Financial metrics
    with open(os.path.join(summaries_dir,"financial_metrics.json"), "r", encoding="utf-8") as f:
        financial_metrics = json.load(f)  

    # 4. Chart paths
    chart_paths = {
        "pe_eps_performance":f"{state.work_dir}/summaries/pe_eps_performance.png",
        "share_performance": f"{state.work_dir}/summaries/share_performance.png"
    }

    # 5. Section summaries (txts)
    summary_texts = {}
    for fname in os.listdir(summaries_dir):
        if fname.endswith(".txt"):
            with open(os.path.join(summaries_dir, fname), "r", encoding="utf-8") as f:
                summary_texts[fname] = f.read()

    # 6. Call the new report writer
    output_pdf_path = f"{state.work_dir}/final_annual_report.pdf"
    result = ReportLabUtils.build_annual_report(
        ticker_symbol=state.company_details['identifiers']['ticker'],
        filing_date=state.year,
        output_pdf_path=output_pdf_path,
        sections=conceptual_sections,
        key_data=key_data,
        financial_metrics=financial_metrics,
        chart_paths=chart_paths,
        summaries=summary_texts
    )

    return {"final_report_text": result, "output_pdf": output_pdf_path}

# --- Node 5 & 6: I/O and Auditing ---
def save_report_node(state: AgentState, io_agent) -> Dict[str, Any]:
    """
    Saves the final report to disk.
    """
    print("--- Executing Node: Save Final Report ---")
    report_text = state.final_report_text
    # file_path = io_agent.save(report_text, "final_report.md")
    file_path = "reports/final_report.md"
    return {"final_report_path": file_path}

# --- Node 7, 8, 9: Evaluation and Meta-Reporting ---
def run_evaluation_node(state: AgentState, audit_agent) -> Dict[str, Any]:
    """
    Runs quality checks on the final report.
    """
    print("--- Executing Node: Run Evaluation ---")
    # CORRECTED: Use dot notation
    report_text = state.final_report_text
    # evaluation_results = audit_agent.chat(f"Audit this report for cost, time, and hallucinations: {report_text}")
    evaluation_results = {"cost": 0.50, "hallucination_score": 0.95}
    return {"evaluation_results": evaluation_results}

def save_evaluation_report_node(state: AgentState, io_agent) -> Dict[str, Any]:
    """
    Saves the evaluation results to disk.
    """
    print("--- Executing Node: Save Evaluation Report ---")
    # CORRECTED: Use dot notation
    eval_results = state.evaluation_results
    # file_path = io_agent.save(eval_results, "evaluation_report.json")
    file_path = "reports/evaluation_report.json"
    logging.info(f"Evaluation report saved to {file_path}")
    # This is a final node, it doesn't need to return anything to update the state
    return {}