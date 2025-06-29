# agents/triggers.py

from typing import Dict, Any

def trigger_on_file_save_confirmation(message: Dict[str, Any]) -> bool:
    """
    (Placeholder) A trigger function that returns True if a file save is confirmed.
    """
    content = message.get("content", "")
    return "file has been saved" in content.lower()