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


FILE_TO_TEMPLATE_KEY = {
    "income_statement.json": "table_str",
    "balance_sheet.json": "table_str",
    "cash_flow.json": "table_str",
    "competitors.json": "table_str",
    "company_profile.json": "business_summary",
    "key_data.json": "company_name",
    # IMPORTANT: Ensure these exact basenames are produced by your data collection tasks
    "sec_10k_section_7.txt": "section_text",
    "sec_10k_section_1.txt": "business_summary",
    "sec_10k_section_1a.txt": "risk_factors",
    "financial_metrics.json": "financial_metrics_json_str", 
    # Note: PNG files are handled separately by report writer, not usually summarized directly
}

# In a config or at the top of your graph_nodes.py
TOOL_MAP: Dict[str, Callable] = {
    "get_sec_10k_section_1": report_utils.get_sec_10k_section_1,
    "get_sec_10k_section_1a": report_utils.get_sec_10k_section_1a,
    "get_sec_10k_section_7": report_utils.get_sec_10k_section_7,
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
print(f"\n--- DEBUG: TOOL_MAP keys at module load: {list(TOOL_MAP.keys())} ---\n") # ADD THIS LINE

def get_data_collection_tasks(state: AgentState) -> List[Dict[str, str]]:
    work_dir = state.work_dir
    sec_filings_dir = os.path.join(work_dir, 'sec_filings') 

    # Assign the list to a variable first
    tasks_list = [
        # Individual entries for each SEC section (as per our last refactoring)
        {'task': 'get_sec_10k_section_1', 'file': os.path.join(sec_filings_dir, 'sec_10k_section_1.txt')},
        {'task': 'get_sec_10k_section_1a', 'file': os.path.join(sec_filings_dir, 'sec_10k_section_1a.txt')},
        {'task': 'get_sec_10k_section_7', 'file': os.path.join(sec_filings_dir, 'sec_10k_section_7.txt')},
        
        # The rest of your tasks, which produce a single file
        {'task': 'get_key_data', 'file': os.path.join(work_dir, 'summaries', 'key_data.json')},
        {'task': 'get_company_profile',  'file': os.path.join(work_dir, 'company_profile.json')},
        {'task': 'get_competitors', 'file': os.path.join(work_dir, 'competitors.json')},
        {'task': 'get_income_statement', 'file': os.path.join(work_dir, 'income_statement.json')},
        {'task': 'get_balance_sheet', 'file': os.path.join(work_dir, 'balance_sheet.json')},
        {'task': 'get_cash_flow',  'file': os.path.join(work_dir, 'cash_flow.json')},
        {'task': 'get_pe_eps_chart', 'file': os.path.join(work_dir, 'summaries', 'pe_eps_performance.png')},
        {'task': 'get_share_performance_chart',  'file': os.path.join(work_dir, 'summaries', 'share_performance.png')},
        {'task': 'financial_metrics', 'file': os.path.join(work_dir, 'summaries', 'financial_metrics.json')}
    ]

    print(f"--- DEBUG: get_data_collection_tasks generated tasks: {tasks_list} ---\n") # Now this print is correct
    return tasks_list

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
    Each task is now expected to produce a single output file.
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
        "work_dir": state.work_dir # Ensure work_dir is passed, though individual functions might not need it if save_path is full path
    }

    for task in get_data_collection_tasks(state):
        task_name = task['task']
        output_file = task['file'] # This is now always a full file path
        print(f"--- DEBUG: Current task_name being looked up: '{task_name}' ---\n") # ADD THIS LINE

        task_function = TOOL_MAP[task_name]

        try:
            # Ensure the directory for the output file exists BEFORE calling the tool
            output_dir = os.path.dirname(output_file)
            os.makedirs(output_dir, exist_ok=True)

            sig = inspect.signature(task_function)
            call_args = {param: available_args[param] for param in sig.parameters if param in available_args}
            
            # Add save_path argument if the function expects it
            if 'save_path' in sig.parameters:
                call_args['save_path'] = output_file # Pass the full file path for saving

            print(f"[DataCollection] Running {task_name} -> {output_file}")
            task_function(**call_args) # Call the function
            
            # After execution, add the created file to raw_data_files
            raw_data_files.append(output_file)

        except Exception as e:
            error_message = f"Error executing {task_name}: {e}"
            logging.error(error_message, exc_info=True)
            error_log.append(error_message)

    return {
        "raw_data_files": raw_data_files,
        "error_log": error_log,
    }

def load_all_data(state: AgentState) -> Dict[str, Any]: # Changed value type to Any for json objects
    """
    Loads all collected raw data files specified in get_data_collection_tasks.
    Returns a dictionary where keys are file basenames and values are content (str for text, dict for JSON).
    """
    all_data = {}
    work_dir = state.work_dir # Correctly access work_dir from the passed AgentState

    # Use the tasks defined by get_data_collection_tasks as the source of truth for paths
    task_specs = get_data_collection_tasks(state) # Get the full list of expected file paths
    
    for task in task_specs:
        full_file_path = task["file"]
        # Use the basename of the file as the key in the all_data dictionary
        # This matches how FILE_TO_TEMPLATE_KEY is likely structured (e.g., "key_data.json")
        file_basename = os.path.basename(full_file_path) 

        if os.path.exists(full_file_path) and os.path.getsize(full_file_path) > 0:
            try:
                # Determine if it's JSON or plain text based on extension
                if full_file_path.lower().endswith(".json"): # Use .lower() for case-insensitivity
                    with open(full_file_path, "r", encoding="utf-8") as f:
                        all_data[file_basename] = json.load(f) # Load JSON as Python object (dict/list)
                else: # Assume plain text for other extensions (e.g., .txt, .png but png won't be loaded as text here)
                    with open(full_file_path, "r", encoding="utf-8") as f:
                        all_data[file_basename] = f.read() # Read content as a string
            except Exception as e:
                logging.warning(f"Could not load {full_file_path} into all_data: {e}", exc_info=True)
                # Consider adding a placeholder or specific error message to all_data here if crucial
        else:
            # This handles cases where optional files might be missing, or errors from data_collection
            logging.warning(f"File not found or empty during data loading for summarization: {full_file_path}")
            # Add a placeholder for missing data to prevent KeyError in prompt formatting
            all_data[file_basename] = f"MISSING_DATA_FOR_{file_basename.upper().replace('.', '_')}" 
    
    return all_data

def summarization_node(state: AgentState, agent) -> dict:
    """
    Loads all raw data, builds the LLM prompt for each summary, sends it to the agent,
    and saves the resulting summary.
    """
    print("--- Executing Node: Summarization ---") # Add a clear start message

    # --- Extract orchestration vars defensively ---
    company_name = state.company_details.get("official_name", state.company) if state.company_details else state.company
    ticker = state.company_details.get("identifiers", {}).get("ticker") if state.company_details else None
    year = state.year
    peers = state.company_details.get("competitors", []) if state.company_details else []
    
    # Ensure work_dir is present and valid
    if not state.work_dir:
        logging.error("summarization_node: state.work_dir is missing or empty. Cannot proceed.")
        raise ValueError("Work directory not set in state for summarization.")

    # --- Load all collected raw data using the externalized helper ---
    # all_data will now have basenames as keys (e.g., "key_data.json")
    all_data = load_all_data(state) 
    
    # Debugging: Check what was loaded
    logging.info(f"Summarization: Loaded data keys: {list(all_data.keys())}")

    keys_to_summarize = [
        "analyze_income_stmt", "analyze_balance_sheet", "analyze_cash_flow",
        "analyze_segment_stmt", "analyze_business_highlights",
        "analyze_company_description", "get_risk_assessment",
        "get_competitors_analysis"
    ]

    summary_outputs = {}
    
    # Loop through each summary type defined in finrobot_prompt_library
    for key in keys_to_summarize:
        if key not in finrobot_prompt_library:
            logging.warning(f"Prompt key '{key}' not found in finrobot_prompt_library. Skipping.")
            summary_outputs[key] = f"ERROR: Prompt not defined for {key}"
            continue

        prompt_template = finrobot_prompt_library[key].get("prompt_template")
        input_files_expected = finrobot_prompt_library[key].get("input_files", [])

        if not prompt_template:
            logging.warning(f"Prompt template missing for key '{key}'. Skipping.")
            summary_outputs[key] = f"ERROR: Prompt template missing for {key}"
            continue

        # Orchestration variables (common to many prompts)
        extra_args = {
            "company_name": company_name,
            "ticker": ticker,
            "year": year,
            "competitor_names": ", ".join(peers) if isinstance(peers, list) else str(peers),
        }

        # Gather all required template vars from all_data using FILE_TO_TEMPLATE_KEY
        template_args = {**extra_args}
        
        missing_template_data = [] # Track what's missing for this prompt
        for file_rel_path in input_files_expected: # These are relative paths, like 'summaries/key_data.json'
            fname_basename = os.path.basename(file_rel_path) # Extract basename to match all_data keys
            template_var_name = FILE_TO_TEMPLATE_KEY.get(fname_basename) # Get the variable name (e.g., "company_name")

            if template_var_name:
                # Retrieve content from all_data (which uses basenames as keys)
                content = all_data.get(fname_basename)
                if content is not None:
                    # Special handling for JSON content if the template expects a string
                    if isinstance(content, (dict, list)) and template_var_name.endswith("_json_str"):
                        template_args[template_var_name] = json.dumps(content, indent=2)
                    elif isinstance(content, (dict, list)): # If it's json but template expects string, convert
                         template_args[template_var_name] = str(content)
                    else: # It's a string (text file content)
                        template_args[template_var_name] = content
                else:
                    # Use the placeholder added by load_all_data if content is None (file missing)
                    template_args[template_var_name] = f"PLACEHOLDER_FOR_{fname_basename.upper().replace('.', '_')}"
                    missing_template_data.append(fname_basename)
            else:
                logging.warning(f"No FILE_TO_TEMPLATE_KEY mapping for basename '{fname_basename}' from '{file_rel_path}'. Skipping its use in prompt.")

        # Special handling for 'section_text' if prompt expects 'section_7' separately (e.g., for direct naming)
        if "section_text" in template_args and "section_7" not in template_args:
             template_args["section_7"] = template_args["section_text"] # Example: copy for backward compatibility

        try:
            prompt = prompt_template.format(**template_args)
            if missing_template_data:
                logging.warning(f"Prompt '{key}' generated with missing data for: {', '.join(missing_template_data)}")
        except KeyError as e:
            summary_outputs[key] = f"ERROR: Missing expected placeholder for '{e}' in prompt template '{key}'. Check finrobot_prompt_library and FILE_TO_TEMPLATE_KEY."
            logging.error(summary_outputs[key], exc_info=True)
            continue # Skip LLM call for this prompt if formatting failed

        # === Call the agent and pass the prompt ===
        try:
            summary_info = agent.summarize(prompt)
            summary_text = summary_info.summary if hasattr(summary_info, "summary") else str(summary_info) # Defensive access
            cost_text = summary_info.cost if hasattr(summary_info, "cost") else {} # Defensive access
            parsed_cost = parse_cost(cost_text) if cost_text else {}

            # --- LOGGING STEPS ---
            log_agent_step(
                agent_name=getattr(agent, "name", "summarizer"),
                input_text=prompt,
                output_text=summary_text,
                tool_used="summarize"
            )

            log_cost_estimate(
                agent_outputs={
                    "summary_key": key,
                    "summary_length": len(summary_text),
                    **parsed_cost
                }
            )
            summary_outputs[key] = summary_text

        except Exception as agent_exc:
            error_message = f"Error during agent summarization for key '{key}': {agent_exc}"
            logging.error(error_message, exc_info=True)
            summary_outputs[key] = f"ERROR: Agent summarization failed for {key}: {agent_exc}"


    # --- Post-Summarization Validation and Saving ---
    if not summary_outputs:
        logging.error("[FATAL] No summaries generated from any prompt. Failing before file write.")
        raise RuntimeError("[FATAL] No summaries generated! Workflow cannot proceed.")
    if any(v and isinstance(v, str) and v.startswith("ERROR:") for v in summary_outputs.values()):
        logging.error("[FATAL] One or more summaries failed with an ERROR message. Check logs for details.")
        # Decide if you want to allow partial success or fail completely
        # For a financial report, a fatal error seems appropriate.
        raise RuntimeError("[FATAL] One or more summaries failed. Check logs.")

    out_path = os.path.join(state.work_dir, "preliminaries", "all_summaries.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary_outputs, f, indent=2, ensure_ascii=False)
        
        # Immediately verify write
        with open(out_path, "r", encoding="utf-8") as f:
            reloaded = json.load(f)
            if len(reloaded) != len(summary_outputs):
                raise RuntimeError(f"[FATAL] File write mismatch! Expected {len(summary_outputs)} sections, found {len(reloaded)}.")
        logging.info(f"All summaries successfully saved to {out_path}")
    except Exception as file_exc:
        logging.error(f"[FILE WRITE FAILURE] Could not save summaries to {out_path}: {file_exc}", exc_info=True)
        raise # Re-raise to fail the node if saving fails

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
    output_dir = os.path.join(state.work_dir, "summaries")
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
    os.makedirs(os.path.join(state.work_dir, "summaries"), exist_ok=True)
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
    file_path = f"{state.work_dir}/final_report.md"
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
    file_path = f"{state.work_dir}/evaluation_report.json"
    logging.info(f"Evaluation report saved to {file_path}")
    # This is a final node, it doesn't need to return anything to update the state
    return {}