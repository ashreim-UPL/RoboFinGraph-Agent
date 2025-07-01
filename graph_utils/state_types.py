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
    year: int
    filing_date: Optional[str]

    raw_data_files: Annotated[List[str], operator.add] = Field(default_factory=list) # Use Field for default with Annotated
    
    conceptual_sections: Optional[Dict[str, Any]] = None
    final_report_text: Optional[str] = None
    final_report_path: Optional[str] = None
    evaluation_results: Optional[Dict[str, Any]] = None
    
    agent_name: Optional[str] = None
    tool_call_result: Optional[Dict[str, Any]] = None
    memory: Dict[str, Any] = Field(default_factory=dict)
    audit_notes: List[str] = Field(default_factory=list)
    
    llm_decision: Optional[str] = None # Stores the raw LLM decision (e.g., "continue", "end")
    validation_result_key: Optional[str] = None # Stores the programmatic decision from validate_collected_data_node for routing
    
    work_dir: str 
    error_log: List[str]
