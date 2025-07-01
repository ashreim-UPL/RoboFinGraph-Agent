import os
import json
import asyncio
import time
from datetime import datetime, timedelta
import inspect
from typing import Dict, Any, List, Callable

from graph_utils.state_types import AgentState
from tools.company_resolver import identify_company_and_region
from tools.file_utils import check_file_stats
from tools.report_writer import ReportLabUtils
from tools.global_API_toolkit import get_10k_section, get_10k_metadata
from utils.logger import get_logger, log_event, log_agent_step, log_cost_estimate
from prompts.summarization_intsruction import finrobot_prompt_library
from prompts.report_summaries import report_section_specs
from tools import report_utils
from functools import partial

logger = get_logger()

# --- Helper Constants & Functions ---
FILE_TO_TEMPLATE_KEY = {
    "income_statement.json": "table_str",
    "balance_sheet.json": "table_str",
    "cash_flow.json": "table_str",
    "competitors.json": "table_str",
    "company_profile.json": "business_summary",
    "key_data.json": "company_name",
    "sec_10k_section_1.txt": "business_summary",
    "sec_10k_section_1a.txt": "risk_factors",
    "sec_10k_section_7.txt": "section_text",
    "financial_metrics.json": "financial_metrics_json_str",
}

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

def get_data_collection_tasks(state: AgentState) -> List[Dict[str, str]]:
    work_dir = state.work_dir
    sec_dir = os.path.join(work_dir, 'sec_filings')
    tasks = [
        {'task': 'get_sec_10k_section_1', 'file': os.path.join(sec_dir, 'sec_10k_section_1.txt')},
        {'task': 'get_sec_10k_section_1a', 'file': os.path.join(sec_dir, 'sec_10k_section_1a.txt')},
        {'task': 'get_sec_10k_section_7', 'file': os.path.join(sec_dir, 'sec_10k_section_7.txt')},
        {'task': 'get_key_data', 'file': os.path.join(work_dir, 'summaries', 'key_data.json')},
        {'task': 'get_company_profile', 'file': os.path.join(work_dir, 'company_profile.json')},
        {'task': 'get_competitors', 'file': os.path.join(work_dir, 'competitors.json')},
        {'task': 'get_income_statement', 'file': os.path.join(work_dir, 'income_statement.json')},
        {'task': 'get_balance_sheet', 'file': os.path.join(work_dir, 'balance_sheet.json')},
        {'task': 'get_cash_flow', 'file': os.path.join(work_dir, 'cash_flow.json')},
        {'task': 'get_pe_eps_chart', 'file': os.path.join(work_dir, 'summaries', 'pe_eps_performance.png')},
        {'task': 'get_share_performance_chart', 'file': os.path.join(work_dir, 'summaries', 'share_performance.png')},
        {'task': 'financial_metrics', 'file': os.path.join(work_dir, 'summaries', 'financial_metrics.json')}
    ]
    print(f"--- DEBUG: get_data_collection_tasks => {tasks} ---")
    return tasks


def parse_cost(cost_obj: Dict[str, Any]) -> Dict[str, Any]:
    try:
        usage = cost_obj.get("usage_including_cached_inference", {})
        total = usage.get("total_cost")
        model = next((k for k,v in usage.items() if isinstance(v, dict)), None)
        stats = usage.get(model, {}) if model else {}
        return {
            "total_cost": total,
            "model": model,
            "model_cost": stats.get("cost"),
            "prompt_tokens": stats.get("prompt_tokens"),
            "completion_tokens": stats.get("completion_tokens"),
            "total_tokens": stats.get("total_tokens"),
        }
    except Exception as e:
        logger.error(f"Cost parse error: {e}")
        return {}


def generate_validation_prompt(work_dir: str, missing: List[str], stats: Dict[str, Any], score: int) -> str:
    return f"""
You are a financial data quality auditor.
Directory: {work_dir}
Missing: {json.dumps(missing)}
Stats: {json.dumps(stats)}
Score: {score}%
Rules:
- If any missing => end: <reason>
- Else valid
Output EXACTLY:
valid
end: <reason>
"""


def extract_route(result: Any) -> str:
    raw = (getattr(result, 'summary', str(result))).strip().lower()
    if raw.startswith("valid"): return "valid"
    if raw.startswith("end"):   return "end"
    return "end"

def load_all_data(state: AgentState) -> Dict[str, Any]:
    data = {}
    for spec in get_data_collection_tasks(state):
        path = spec['file']
        key = os.path.basename(path)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            if path.lower().endswith('.json'):
                with open(path, 'r', encoding='utf-8') as f: data[key] = json.load(f)
            else:
                with open(path, 'r', encoding='utf-8') as f: data[key] = f.read()
        else:
            data[key] = None
    return data

# --- Node Functions ---

def resolve_company_node(state: AgentState, **kwargs) -> Dict[str, Any]:
    start = time.time()
    result = asyncio.run(identify_company_and_region(state.company))
    duration = round(time.time() - start, 2)
    details = result.get('company_details')
    region  = result.get('region')
    log_event('resolve_company', {'company': state.company, 'duration': duration, 'details': details, 'region': region})
    return {'company_details': details, 'region': region}

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

    result = {
        "__route__": route,    # LangGraph routing key
        "llm_decision": route  # your own field
    }
    # DEBUG output
    print(f"[DEBUG llm_decision_node] returning: {result}")
    return result

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


def validate_collected_data_node(
    state: AgentState,
    validator_agent: Any,
    tasks_fn: Callable,
    gen_prompt: Callable,
    extract: Callable
) -> Dict[str, str]:
    specs = tasks_fn(state)
    missing = [s['file'] for s in specs if not os.path.exists(s['file']) or os.path.getsize(s['file'])==0]
    stats   = {s['file']: check_file_stats(s['file']) for s in specs if s['file'] not in missing}
    score   = int(100*(len(specs)-len(missing))/len(specs))
    prompt  = gen_prompt(state.work_dir, missing, stats, score)
    res     = validator_agent.summarize(prompt)
    route   = extract(res)
    log_event('validate_collected', {'score': score, 'missing': missing, 'route': route})
    return {'llm_decision_route_key': route, '__route__': route}


def summarization_node(
    state: AgentState,
    summarizer_agent: Any,
    load_fn: Callable,
    prompts: Dict[str, Any]
) -> Dict[str, Any]:
    data = load_fn(state)
    outputs = {}
    for key, spec in prompts.items():
        tpl = spec.get('prompt_template', '')
        inputs = spec.get('input_files', [])
        args = {os.path.basename(p): data.get(os.path.basename(p), '') for p in inputs}
        prompt = tpl.format(**args)
        resp   = summarizer_agent.summarize(prompt)
        text   = getattr(resp, 'summary', str(resp))
        cost   = getattr(resp, 'cost', {})
        parsed = parse_cost(cost)
        log_agent_step(getattr(summarizer_agent, 'name','summarizer'), prompt, text, 'summarize')
        log_cost_estimate(parsed)
        outputs[key] = text
    log_event('summarization_complete', {'sections': list(outputs.keys())})
    return {'summary_outputs': outputs}


def conceptual_analysis_node(
    state: AgentState,
    concept_agent: Any,
    specs: Dict[str, Any]
) -> Dict[str, Any]:
    outs = {}
    for out_file, spec in specs.items():
        sections = '\n\n'.join(f"--{s}--\n{state.summary_outputs.get(s,'')}" for s in spec['sources'])
        if not sections: outs[out_file] = ''
        else:
            prompt = spec['prompt_template'].format(company_name=state.company, sections=sections)
            resp   = concept_agent.summarize(prompt)
            text   = getattr(resp,'summary',str(resp))
            log_agent_step(getattr(concept_agent,'name','conceptual'), prompt, text, 'summarize')
            outs[out_file] = text
    log_event('conceptual_complete', {'sections': list(outs.keys())})
    return {'conceptual_sections': outs}


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


def save_report_node(state: AgentState, io_agent: Any) -> Dict[str, Any]:
    path = os.path.join(state.work_dir, 'final_report.md')
    io_agent.save(state.final_report_text, path)
    log_event('save_report', {'path': path})
    return {'final_report_path': path}


def run_evaluation_node(state: AgentState, audit_agent: Any) -> Dict[str, Any]:
    prompt = f"Audit report: {state.final_report_text}"  
    resp   = audit_agent.summarize(prompt)
    results= getattr(resp,'summary',str(resp))
    log_event('evaluation', {'results': results})
    return {'evaluation_results': results}


def save_evaluation_report_node(state: AgentState, io_agent: Any) -> Dict[str, Any]:
    path = os.path.join(state.work_dir, 'evaluation_report.json')
    io_agent.save(state.evaluation_results, path)
    log_event('save_evaluation_report', {'path': path})
    return {}
