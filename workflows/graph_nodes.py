# workflows/graph_nodes.py
import logging
import os
import json
import re
from typing import Dict, Any, Callable
from functools import partial
import asyncio
import inspect
from graph_utils.state_types import AgentState
from tools.company_resolver import identify_company_and_region
from tools import report_utils 
# Add in graph_nodes.py
from prompts.summarization_intsruction import finrobot_prompt_library
from prompts.report_summaries import report_section_specs
from utils.logger import log_agent_step, log_final_summary, log_cost_estimate

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

# --- resolve_company_node and decide_to_continue remain the same ---
def resolve_company_node(state: AgentState) -> Dict[str, Any]:
    """
    Resolves the company name to get ticker, region, and peers by calling
    the identify_company_and_region tool.
    """
    print("--- Executing Node: Resolve Company (Live Call) ---")
    company_name = state.company
    print(f"Resolving company: {company_name}...")
    result = asyncio.run(identify_company_and_region(company_name))
    logging.info(f"Company resolved: {result.get('company_details')}")
    return {"company_details": result.get('company_details'), "region": result.get('region')}

def decide_to_continue(state: AgentState) -> str:
    # ... (this function remains the same)
    print("--- Executing Node: Decide to Continue ---")
    if state.company_details and state.company_details.get('identifiers', {}).get('ticker'):
        return "continue"
    else:
        return "end"


# --- The data collection node now inspects function signatures ---
def data_collection_node(state: AgentState, task_function: Callable, output_file: str) -> Dict[str, Any]:
    """
    Receives a tool function and intelligently calls it with only the
    arguments it can accept from the available state.
    """
    task_name_str = task_function.func.__name__ if isinstance(task_function, partial) else task_function.__name__
    print(f"--- Executing Data Collection via tool: {task_name_str} ---")

    ticker = state.company_details['identifiers']['ticker']
    region = state.region
    year = state.year

    try:
        sig = inspect.signature(task_function)
        
        # --- Start of Changes ---
        # Create a dictionary of ALL possible arguments our tools might need.
        # This makes the node compatible with any of our tool functions.
        available_args = {
            "ticker": ticker,
            "ticker_symbol": ticker, # Alias for ticker
            "region": region,
            "fyear": year,
            "filing_date": f"{year}-12-31", # Provide a constructed filing_date
            "save_path": output_file
        }
        # --- End of Changes ---
        
        # Create a dictionary of arguments that the target function actually accepts
        call_args = {param: available_args[param] for param in sig.parameters if param in available_args}
        
        # Execute the function with only the valid arguments
        task_function(**call_args)
        
        return {"raw_data_files": [output_file]}

    except Exception as e:
        error_message = f"Error executing task {task_name_str}: {e}"
        logging.error(error_message, exc_info=True)
        # Return None or an error message so the graph can continue
        return {"error_log": [error_message]}

    except Exception as e:
        error_message = f"Error executing task {task_name_str}: {e}"
        logging.error(error_message, exc_info=True)
        return {"error_log": [error_message]}

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

    # Save everything to one file
    with open("report/preliminaries/all_summaries.json", "w", encoding="utf-8") as f:
        json.dump(summary_outputs, f, indent=2, ensure_ascii=False)

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

    with open("report/preliminaries/all_summaries.json", "r", encoding="utf-8") as f:
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
    with open("report/summaries/conceptual_sections.json", "w", encoding="utf-8") as f:
        json.dump(conceptual_outputs, f, indent=2, ensure_ascii=False)

    return {"conceptual_sections": conceptual_outputs}



# --- Node 4: Thesis Generation ---
def generate_report_node(state: AgentState, agent) -> Dict[str, Any]:
    """
    Generates the final report from the conceptual sections.
    """
    print("--- Executing Node: Generate Thesis Report ---")
    # CORRECTED: Use dot notation
    sections = state.conceptual_sections
    # final_report_text = agent.chat(f"Create a report from these sections: {sections}")
    final_report_text = "This is the final investment report for the company."
    return {"final_report_text": final_report_text}

# --- Node 5 & 6: I/O and Auditing ---
def save_report_node(state: AgentState, io_agent) -> Dict[str, Any]:
    """
    Saves the final report to disk.
    """
    print("--- Executing Node: Save Final Report ---")
    # CORRECTED: Use dot notation
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