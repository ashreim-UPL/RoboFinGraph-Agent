# tools/audit_utils.py

from typing import List, Dict, Any

def calculate_token_cost(texts: list[str]) -> float:
    total_tokens = sum(len(t.split()) for t in texts)  # rough estimate
    return round(total_tokens / 1000 * 0.03, 4)  # Assuming $0.03/1K tokens

def validate_pipeline_accuracy(results: list[dict]) -> dict:
    """Dummy placeholder for audit logic"""
    passed = sum(1 for r in results if r.get("status") == "ok")
    failed = len(results) - passed
    return {"passed": passed, "failed": failed, "score": passed / max(1, len(results))}
