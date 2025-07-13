# utils/logger.py
# Centralized logging module for RoboFinGraph

import os
import sys
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

# === Logger Setup ===
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

_log_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
_logger = logging.getLogger("RoboFinGraphLogger")
_logger.setLevel(logging.INFO)

# File handler - logs everything
_file_handler = logging.FileHandler(os.path.join(LOG_DIR, "robofingraph_events.log"), encoding="utf-8")
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(_log_formatter)
_logger.addHandler(_file_handler)

# Stream handler - only show WARNING or above in CLI
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setLevel(logging.WARNING)  # Only WARNING/ERROR shown on CLI
_stream_handler.setFormatter(_log_formatter)
_logger.addHandler(_stream_handler)

def _safe_json(obj):
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "value"):
        return obj.value
    return str(obj)

def setup_logging():
    pass

def get_logger(name: str = "RoboFinGraphLogger") -> logging.Logger:
    return _logger

def _truncate(val, n=100):
    if isinstance(val, str) and len(val) > n:
        return val[:n-3] + "..."
    if isinstance(val, list) and len(val) > 5:
        return val[:2] + ["...(truncated, {} items)".format(len(val))]
    return val

def cli_human_summary(event_type, payload):
    """
    Print a human-friendly message for the CLI. Only show important keys, and truncate any long content.
    """
    summary_keys = [
        "company", "year", "status", "termination_reason", "error_log", "accuracy_score",
        "work_dir", "raw_data_files"
    ]
    msg = [f"{event_type.upper()}:"]
    if isinstance(payload, dict):
        for k in summary_keys:
            if k in payload:
                val = payload[k]
                if k == "raw_data_files":
                    msg.append(f"- {k}: {len(val)} files")
                else:
                    msg.append(f"- {k}: {_truncate(val)}")
    print("\n[RoboFinGraph] " + " | ".join(msg))

def log_event(event_type: str, payload: Dict[str, Any], session_id: Optional[str] = None) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "session_id": session_id or "global",
        "payload": payload
    }
    # print(json.dumps({"event_type": "test_event", "payload": {"hello": "frontend"} }), flush=True)

    _logger.info(json.dumps(entry))

    print(json.dumps(entry, default=_safe_json), flush=True)

    # Show CLI human summary for warnings/errors/key events (customize if needed)
    if event_type.lower() in {"error", "tool_failure", "terminate", "pipeline_end"}:
        cli_human_summary(event_type, payload)


def log_agent_step(agent_name: str, input_text: str, output_text: str, tool_used: Optional[str] = None, session_id: Optional[str] = None) -> None:
    log_event("agent_step", {
        "agent": agent_name,
        "input": input_text[:100],
        "output": output_text[:100],
        "tool": tool_used
    }, session_id=session_id)

def log_final_summary(summary: str, session_id: Optional[str] = None) -> None:
    log_event("final_summary", {"summary": summary[:100]}, session_id=session_id)

# === Tool & Audit Logs ===
def classify_hallucination_type(filename: str, tool_name: str, failure_type: str) -> str:
    ftype = failure_type.lower()
    if "missing" in ftype:
        return "missing_output"
    if "timeout" in ftype:
        return "api_timeout"
    if "short" in ftype:
        return "undergenerated"
    if "tool_crash" in ftype:
        return "tool_exception"
    return "unknown"

def llm_should_retry(tool_name: str, failure_reason: str) -> str:
    reason = failure_reason.lower()
    if any(term in reason for term in ["timeout", "rate limit", "empty"]):
        return "retry"
    if "irrelevant" in reason:
        return "switch"
    return "no_action"

def log_tool_failure(tool_name: str, filename: str, reason: str, agent: Optional[str] = None, session_id: Optional[str] = None) -> None:
    failure_entry = {
        "tool": tool_name,
        "file": filename,
        "reason": reason,
        "classification": classify_hallucination_type(filename, tool_name, reason),
        "recommendation": llm_should_retry(tool_name, reason)
    }
    log_event("tool_failure", failure_entry, session_id=session_id or agent)

def log_audit_result(agent: str, hallucinations: List[str], context: str = "", session_id: Optional[str] = None) -> None:
    explanation = "[Manual audit required. No LLM call used.]"
    entry = {
        "agent": agent,
        "hallucinations": hallucinations,
        "explanation": explanation,
        "timestamp": datetime.now().isoformat()
    }
    log_event("audit", entry, session_id=session_id or agent)

# Optional: Table/summary printer for pipeline runs
def print_pipeline_status(memory):
    steps = memory.get('pipeline_data', [])
    print("\n[Pipeline Status]")
    print(f"{'Node':25} | {'Status':10} | {'Duration':8}")
    print("-" * 48)
    for s in steps:
        print(f"{s.get('current_node','')[:25]:25} | {str(s.get('status','-'))[:10]:10} | {str(s.get('duration','-'))[:8]:8}")

def match_file_to_concept(section_name: str, file_list: List[str]) -> Optional[str]:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"""
You are given a section of a financial report titled: '{section_name}'.
From the following available files, pick the most appropriate one for that section:

{file_list}

Return only the exact filename best suited.
"""
    try:
        response = client.chat.completions.create(
            model="go3-mini-2025-01-31",
            messages=[
                {"role": "system", "content": "You are a financial data assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=50
        )
        filename = response.choices[0].message.content.strip()
        return filename if filename in file_list else None
    except Exception:
        return None

def generate_placeholder_summary(filename: str, company: str) -> str:
    return f"[AUTO] Placeholder summary for {company} - Source: {filename}. This was auto-generated due to missing data or tool failure."
