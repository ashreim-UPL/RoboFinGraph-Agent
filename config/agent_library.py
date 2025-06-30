# Central config registry (e.g., agent definitions)
"""
This module contains all named agent instances used across the LangGraph pipeline.
Each agent is initialized using FinRobot and can be a leader, worker, auditor, or tool caller.
"""

from typing import Dict, Any
from agents.finrobot_base import FinRobot
from autogen import UserProxyAgent
from agents.single_agent import SingleAssistant

# All functions that create a FinRobot agent have been updated to accept a 'config' dictionary.
def get_validation_agent(config: Dict[str, Any], **kwargs) -> SingleAssistant:
    return SingleAssistant(
        agent_config={
            "name": "LLMValidator",
            "description": "Validates if company resolution is sufficient to proceed. Strictly Respond with Continue or End",
            "toolkits": [],
        },
        llm_config={"model": config["llm_models"]["LLMValidator"]},
        max_consecutive_auto_reply=1,
        **kwargs
    )

def get_tool_agent(name: str, endpoint: str, task: str, config: Dict[str, Any], **kwargs) -> FinRobot: #
    """Creates a generic agent for calling a specific tool or API."""
    return FinRobot(
        agent_config={
            "name": name,
            "description": f"Calls external API for {task}",
            "api_endpoint": endpoint
        },
        # Use the 'leader' model for generic tools, or another appropriate default.
        llm_config={"model": config["llm_models"]["leader"]},
        **kwargs,
    )

def get_company_resolver_agent(config: Dict[str, Any], **kwargs) -> FinRobot: #
    """Creates the agent responsible for resolving company details."""
    # The 'description' should be part of the agent_config, not a direct kwarg to FinRobot.
    return FinRobot(
        agent_config={
            "name": "company_resolver",
            "description": "Resolves company names and fetches metadata like ticker, country, and peers"
        },
        llm_config={"model": config["llm_models"]["leader"]}, # Uses the leader model
        **kwargs,
    )

def get_summarizer_agent(config: Dict[str, Any], **kwargs) -> SingleAssistant:
    return SingleAssistant(
        agent_config={"name": "Summarizer"},
        llm_config={"model": config["llm_models"]["summarizer"]},
        code_execution_config={"work_dir": "report", "use_docker": False},
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        **kwargs,
    )

def get_concept_summarizer_global(config: Dict[str, Any], **kwargs) -> SingleAssistant:
    return SingleAssistant(
        agent_config={"name": "Concept_Summarizer"},
        llm_config={"model": config["llm_models"]["concept_cot"]},
        code_execution_config={"work_dir": "report", "use_docker": False},
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        **kwargs,
    )

# === Leaders ===
def get_expert_investor_us(config: Dict[str, Any], **kwargs) -> FinRobot: #
    return FinRobot(
        agent_config="expert_us",
        llm_config={"model": config["llm_models"]["leader"]},
        **kwargs)

def get_expert_investor_india(config: Dict[str, Any], **kwargs) -> FinRobot: #
    return FinRobot(
        agent_config="expert_india",
        llm_config={"model": config["llm_models"]["leader"]},
        **kwargs)

# === Shadows (Auditors) ===
def get_shadow_auditor(config: Dict[str, Any], **kwargs) -> FinRobot: #
    return FinRobot(
        agent_config="shadow_auditor",
        llm_config={"model": config["llm_models"]["auditor"]},
        **kwargs)

def get_shadow_pipeline_checker(config: Dict[str, Any], **kwargs) -> FinRobot: #
    return FinRobot(
        agent_config="shadow_pipeline",
        llm_config={"model": config["llm_models"]["auditor"]},
        **kwargs)

# === Data Collectors (Data_CoT) ===
def get_data_collector_us(config: Dict[str, Any], **kwargs) -> FinRobot: #
    return FinRobot(
        agent_config="data_cot_us",
        llm_config={"model": config["llm_models"]["data_cot"]},
        **kwargs,
    )

def get_data_collector_india(config: Dict[str, Any], **kwargs) -> FinRobot: #
    return FinRobot(
        agent_config="data_cot_india",
        llm_config={"model": config["llm_models"]["data_cot"]},
        **kwargs)

# === Concept Summarizers (Concept_CoT) ===
def get_concept_summarizer_us(config: Dict[str, Any], **kwargs) -> FinRobot: #
    return FinRobot(
        agent_config="concept_cot_us",
        llm_config={"model": config["llm_models"]["concept_cot"]},
        **kwargs)

# === Thesis Creator (Thesis_CoT) ===
def get_thesis_creator(config: Dict[str, Any], **kwargs) -> FinRobot: #
    return FinRobot(
        agent_config="thesis_cot",
        llm_config={"model": config["llm_models"]["thesis_cot"]},
        **kwargs)

# === I/O Agent ===
def get_io_agent(config: Dict[str, Any], **kwargs) -> FinRobot: #
    # IO agent might not need an LLM, but accepting config makes it consistent.
    return FinRobot(agent_config="io_agent", **kwargs)

# === User Proxy ===
# This function does not create a FinRobot, so it doesn't need the config for LLM selection.
def get_user_proxy(**kwargs) -> UserProxyAgent: #
    return UserProxyAgent(
        name="User_Proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config={"work_dir": kwargs.get("work_dir", "coding"), "use_docker": False},
    )

# Note on Global Agents:
# The following lines will cause an error because the 'config' object is not available
# when this module is first imported. It is better to create these agents inside your
# orchestrator function where 'config' is available.

# Example:
# In orchestrator.py:
# rag_agent = get_tool_agent("RAG_Query", "...", "doc retrieval", config=config)

# rag_agent = get_tool_agent("RAG_Embedding_Query", "https://yourrag.com/query", "document retrieval")
# stock_agent = get_tool_agent("Stock_Data_Fetcher", "https://api.fmp.com/...", "stock retrieval")