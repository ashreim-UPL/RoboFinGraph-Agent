# graph_utils/state_types.py
import operator
from typing import Callable, Dict, Any, List, Optional, Union, Annotated
from pydantic import BaseModel, Field

# --- LangGraph State Type ---- #
class AgentState(BaseModel):
    """
    Defines the state passed between nodes in LangGraph.
    """
    company: str
    year: str
    user_input: Optional[str] = None
    
    company_details: Optional[Dict[str, Any]] = None
    region: Optional[str] = None
    
    # --- THIS IS THE CRITICAL CHANGE ---
    # We are "annotating" the raw_data_files field.
    # This tells LangGraph that when multiple nodes write to this field
    # in the same step, it should use the `operator.add` function
    # (which concatenates lists) to combine the results.
    raw_data_files: Annotated[List[str], operator.add] = []
    
    # --- The rest of your state fields remain the same ---
    conceptual_sections: Optional[Dict[str, Any]] = None
    final_report_text: Optional[str] = None
    final_report_path: Optional[str] = None
    evaluation_results: Optional[Dict[str, Any]] = None
    
    agent_name: Optional[str] = None
    tool_call_result: Optional[Dict[str, Any]] = None
    memory: Dict[str, Any] = Field(default_factory=dict)
    audit_notes: List[str] = Field(default_factory=list)