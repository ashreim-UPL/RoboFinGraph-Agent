import os
from functools import partial
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field
import tools.report_utils as report_utils

TOOL_MAP: Dict[str, Any] = {
    "get_sec_10k_section_1":    report_utils.get_sec_10k_section_1,
    "get_sec_10k_section_1a":   report_utils.get_sec_10k_section_1a,
    "get_sec_10k_section_7":    report_utils.get_sec_10k_section_7,
    "get_company_profile":      report_utils.get_company_profile,
    "get_key_data":             report_utils.get_key_data,
    "get_competitors":          report_utils.get_competitor_analysis,
    "get_income_statement":     partial(report_utils.get_financial_statement, statement_type="income_statement"),
    "get_balance_sheet":        partial(report_utils.get_financial_statement, statement_type="balance_sheet"),
    "get_cash_flow":            partial(report_utils.get_financial_statement, statement_type="cash_flow_statement"),
    "get_pe_eps_chart":         report_utils.generate_pe_eps_chart,
    "get_share_performance_chart": report_utils.generate_share_performance_chart,
    "get_financial_metrics":        report_utils.get_financial_metrics,
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
    tool_call_result:     Optional[Dict[str, Any]] = None
    memory:               Dict[str, Any] = Field(default_factory=dict)
    audit_notes:          List[str] = Field(default_factory=list)

    llm_decision:         Optional[str] = None
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
        Build a list of tasks with target filenames for data collection based on state directories.
        """
        tasks: List[Dict[str, str]] = []
        # SEC sections
        for section in ["1", "1a", "7"]:
            key = f"get_sec_10k_section_{section}"
            filename = f"sec_10k_section_{section}.txt"
            tasks.append({
                "task": key,
                "file": os.path.join(self.filing_dir, filename)
            })
        # JSON data in raw
        for task_name in ["get_company_profile", "get_competitors",
                           "get_income_statement", "get_balance_sheet", "get_cash_flow"]:
            json_file = task_name.replace("get_", "") + ".json"
            tasks.append({
                "task": task_name,
                "file": os.path.join(self.raw_data_dir, json_file)
            })
        # charts and metrics in summaries
        for task_name, ext in [("get_pe_eps_chart", "png"), ("get_share_performance_chart", "png"), ("get_financial_metrics", "json"), ("get_key_data", "json")]:
            filename = task_name.replace("get_", "") + (".png" if ext=="png" else ".json")
            tasks.append({
                "task": task_name,
                "file": os.path.join(self.summaries_dir, filename)
            })
        return tasks

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
