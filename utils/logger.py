# utils/logger.py
# Centralized logging module for FinRobot

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

# === Logger Setup ===
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

_log_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
_logger = logging.getLogger("FinRobotLogger")
_logger.setLevel(logging.INFO)

_file_handler = logging.FileHandler(os.path.join(LOG_DIR, "finrobot_events.log"), encoding="utf-8")
_file_handler.setLevel(logging.INFO )
_file_handler.setFormatter(_log_formatter)
_logger.addHandler(_file_handler)

import sys
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(_log_formatter)
_stream_handler.setStream(sys.stdout)
_stream_handler.encoding = 'utf-8' 
_logger.addHandler(_stream_handler)

# === Logger Access API ===
def setup_logging():
    # Already initialized above, this function exists for symmetry.
    pass

def get_logger(name: str = "FinRobotLogger") -> logging.Logger:
    return _logger

# === Core Logging ===
def log_event(event_type: str, payload: Dict[str, Any], session_id: Optional[str] = None) -> None:
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "session_id": session_id or "global",
        "payload": payload
    }
    _logger.info(json.dumps(entry))

def log_agent_step(agent_name: str, input_text: str, output_text: str, tool_used: Optional[str] = None, session_id: Optional[str] = None) -> None:
    log_event("agent_step", {
        "agent": agent_name,
        "input": input_text,
        "output": output_text,
        "tool": tool_used
    }, session_id=session_id)

def log_final_summary(summary: str, session_id: Optional[str] = None) -> None:
    log_event("final_summary", {"summary": summary}, session_id=session_id)

def log_cost_estimate(agent_outputs: List[str], session_id: Optional[str] = None) -> None:
    cost = calculate_token_cost(agent_outputs)
    log_event("cost_estimate", {"total_cost": cost}, session_id=session_id)

def log_evaluation(agent_outputs: List[str], session_id: Optional[str] = None) -> None:
    accuracy = validate_pipeline_accuracy([{"status": "ok" if a else "fail"} for a in agent_outputs])
    log_event("evaluation", accuracy, session_id=session_id)

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
        "timestamp": datetime.now(datetime.timezone.utc).isoformat()
    }
    log_event("audit", entry, session_id=session_id or agent)

# === Concept Matching Helper ===
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
