import os
from functools import partial
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field
import tools.report_utils as report_utils
import tools.report_utils_india as report_utils_indian


TOOL_MAP: Dict[str, Dict[str, Any]] = {
    "get_sec_10k_section_1": {
        "function": report_utils.get_sec_10k_section_1,
        "file_type": "txt",
        "location": "filing"
    },
    "get_sec_10k_section_1a": {
        "function": report_utils.get_sec_10k_section_1a,
        "file_type": "txt",
        "location": "filing"
    },
    "get_sec_10k_section_7": {
        "function": report_utils.get_sec_10k_section_7,
        "file_type": "txt",
        "location": "filing"
    },
    "get_company_profile": {
        "function": report_utils.get_company_profile,
        "file_type": "json",
        "location": "raw_data"
    },
    "get_key_data": {
        "function": report_utils.get_key_data,
        "file_type": "json",
        "location": "summaries" # Assuming key_data is a summary output
    },
    "get_competitors": {
        "function": report_utils.get_competitor_analysis,
        "file_type": "json",
        "location": "raw_data"
    },
    "get_income_statement": {
        "function": partial(report_utils.get_financial_statement, statement_type="income_statement"),
        "file_type": "json",
        "location": "raw_data"
    },
    "get_balance_sheet": {
        "function": partial(report_utils.get_financial_statement, statement_type="balance_sheet"),
        "file_type": "json",
        "location": "raw_data"
    },
    "get_cash_flow": {
        "function": partial(report_utils.get_financial_statement, statement_type="cash_flow_statement"),
        "file_type": "json",
        "location": "raw_data"
    },
    "get_pe_eps_chart": {
        "function": report_utils.generate_pe_eps_chart,
        "file_type": "png",
        "location": "summaries"
    },
    "get_share_performance_chart": {
        "function": report_utils.generate_share_performance_chart,
        "file_type": "png",
        "location": "summaries"
    },
    "get_financial_metrics": {
        "function": report_utils.get_financial_metrics,
        "file_type": "json",
        "location": "summaries"
    },
}

TOOL_IN_MAP: Dict[str, Dict[str, Any]] = {
    "get_sec_10k_section_1": {
        "function": report_utils_indian.get_annual_report_sections_1,
        "file_type": "txt",
        "location": "filing"
    },
    "get_sec_10k_section_1a": {
        "function": report_utils_indian.get_annual_report_sections_1a,
        "file_type": "txt",
        "location": "filing"
    },
    "get_sec_10k_section_7": {
        "function": report_utils_indian.get_annual_report_sections_7,
        "file_type": "txt",
        "location": "filing"
    },
    "get_company_profile": {
        "function": report_utils_indian.get_company_profile,
        "file_type": "json",
        "location": "raw_data"
    },
    "get_key_data": { # Renamed to 'get_key_data' for consistency with main TOOL_MAP,
                      # even if the function name itself implies India-specific data.
        "function": report_utils_indian.get_key_data_india,
        "file_type": "json",
        "location": "summaries" # Assuming this is a summarized output
    },
    "get_competitors": {
        "function": report_utils_indian.get_competitor_analysis,
        "file_type": "json",
        "location": "raw_data"
    },
    "get_income_statement": {
        "function": partial(report_utils_indian.get_financial_statement, statement_type="income_statement"),
        "file_type": "json",
        "location": "raw_data"
    },
    "get_balance_sheet": {
        "function": partial(report_utils_indian.get_financial_statement, statement_type="balance_sheet"),
        "file_type": "json",
        "location": "raw_data"
    },
    "get_cash_flow": {
        "function": partial(report_utils_indian.get_financial_statement, statement_type="cash_flow_statement"),
        "file_type": "json",
        "location": "raw_data"
    },
    "get_pe_eps_chart": {
        "function": report_utils_indian.generate_pe_eps_chart_indian_market,
        "file_type": "png",
        "location": "summaries"
    },
    "get_share_performance_chart": {
        "function": report_utils_indian.generate_share_performance_chart_indian_market,
        "file_type": "png",
        "location": "summaries"
    },
    "get_financial_metrics": {
        "function": report_utils_indian.get_financial_metrics_indian_market,
        "file_type": "json",
        "location": "summaries"
    },
}

# --- LangGraph Agent State Type ---- #
class AgentState(BaseModel):
    """
    Carries business and domain data through each node in LangGraph.
    """
    work_dir: Optional[str] = None
    company: str
    year: str
    user_input: Optional[str] = None

    company_details: Optional[Dict[str, Any]] = None
    region: Optional[str] = None
    filing_date: Optional[str] = None
    sec_report_address: Optional[str] = None
    
    raw_data_files:       List[str] = Field(default_factory=list)
    summary_files:        List[str] = Field(default_factory=list)
    preliminary_files:    List[str] = Field(default_factory=list)
    filing_files:         List[str] = Field(default_factory=list)

    messages:             List[Dict[str, Any]] = Field(default_factory=list)
    evaluation_results:   Optional[Dict[str, Any]] = None
    success_score:        Optional[float]    = None
    accuracy_score:       Optional[float] = None

    agent_name:           Optional[str] = None
    llm_provider:         Optional[str] = None
    llm_model:         Optional[str] = None
    tool_call_result:     Optional[Dict[str, Any]] = None
    memory:               Dict[str, Any] = Field(default_factory=dict)
    audit_notes:          List[str] = Field(default_factory=list)

    llm_decision:         Optional[str] = None
    termination_reason:   Optional[str] = None
    validation_result_key:Optional[str] = None

    start_time:        Optional[datetime] = None
    end_time:          Optional[datetime] = None
    duration:          Optional[float]   = None  # seconds
    
    error_log:            List[str] = Field(default_factory=list)
    
    tokens_sent:      int   = 0
    tokens_generated: int   = 0
    cost_llm:         float = 0.0

    def __init__(self, **data: Any):
        super().__init__(**data)
        if not self.work_dir:
            self.work_dir = f"/report/{self.company}_{self.year}"
        # ensure base directories exist
        Path(self.raw_data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.summaries_dir).mkdir(parents=True, exist_ok=True)
        Path(self.preliminary_dir).mkdir(parents=True, exist_ok=True)
        Path(self.filing_dir).mkdir(parents=True, exist_ok=True)

    @property
    def raw_data_dir(self) -> str:
        return str(Path(self.work_dir) / "raw_data")

    @property
    def summaries_dir(self) -> str:
        return str(Path(self.work_dir) / "summaries")

    @property
    def preliminary_dir(self) -> str:
        return str(Path(self.work_dir) / "preliminaries")

    @property
    def filing_dir(self) -> str:
        return str(Path(self.work_dir) / "filings")

    def get_data_collection_tasks(self) -> List[Dict[str, str]]:
        """
        Build a list of tasks with target filenames for data collection
        based on the instance's region (self.region) and the appropriate TOOL_MAP.
        """
        tasks: List[Dict[str, str]] = []

        # Map location string to actual directory path
        location_map = {
            "filing": self.filing_dir,
            "raw_data": self.raw_data_dir,
            "summaries": self.summaries_dir,
        }

        # Select the appropriate TOOL_MAP based on self.region
        selected_tool_map: Dict[str, Dict[str, Any]]
        region_lower = self.region.lower() # Ensure comparison is case-insensitive

        if region_lower in ("india" or "in"):
            selected_tool_map = TOOL_IN_MAP # For India specific tools
            print(f"INFO: Using TOOL_IN_MAP for region: {self.region}")
        else:
            selected_tool_map = TOOL_MAP # For US specific tools
            print(f"INFO: Using TOOL_MAP for region: {self.region}")

        for task_name, metadata in selected_tool_map.items(): # Iterate over the selected map
            file_type = metadata.get("file_type")
            location_key = metadata.get("location")

            if not file_type or not location_key:
                print(f"WARNING: {task_name} entry in selected TOOL_MAP is missing 'file_type' or 'location'. Skipping.")
                continue

            base_filename = task_name.replace("get_", "")
            filename = f"{base_filename}.{file_type}"

            target_dir = location_map.get(location_key)
            if not target_dir:
                print(f"WARNING: Invalid 'location' specified for '{task_name}': '{location_key}' in selected TOOL_MAP. Skipping.")
                continue

            full_file_path = os.path.join(target_dir, filename)

            tasks.append({
                "task": task_name,
                "file": full_file_path
            })
        return tasks

# -------------------   Indian Specific eventually coudl be integrated
# Usign same general function now get_data_collection_tasks
# --- Instrumentation & Observability Types ---- #
class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR   = "error"

class TokenUsage(BaseModel):
    sent:      int = 0
    received:  int = 0
    generated: int = 0

class NodeState(BaseModel):
    id:                str
    name:              Optional[str] = None
    status:            NodeStatus         = NodeStatus.PENDING
    errors:            List[str]          = Field(default_factory=list)
    files_read:        List[str]          = Field(default_factory=list)
    files_created:     List[str]          = Field(default_factory=list)
    cost_llm:          float              = 0.0
    tokens:            TokenUsage         = Field(default_factory=TokenUsage)
    tools_used:        Set[str]           = Field(default_factory=set)
    success_score:     Optional[float]    = None
    accuracy_score:    Optional[float] = None
    custom_metrics:    Dict[str, Any]     = Field(default_factory=dict)
    parents:           List[str]          = Field(default_factory=list)
    children:          List[str]          = Field(default_factory=list)
    model_version:     Optional[str]      = None
    retries:           int                = 0

class AggregateMetrics(BaseModel):
    total_cost_llm:         float           = 0.0
    total_tokens_sent:      int             = 0
    total_tokens_generated: int             = 0
    overall_success_score:  Optional[float] = None
    overall_accuracy_score: Optional[float] = None

class PipelineState(BaseModel):
    graph_start_time: datetime
    graph_end_time:   Optional[datetime] = None
    total_duration:   Optional[float]    = None
    aggregate:        AggregateMetrics   = Field(default_factory=AggregateMetrics)
    nodes:            Dict[str, NodeState] = Field(default_factory=dict)
