# Central config registry (e.g., agent definitions)
"""
This module contains all named agent instances used across the LangGraph pipeline.
Each agent is initialized using FinRobot and can be a leader, worker, auditor, or tool caller.
"""

import os # Ensure os is imported for environment variables
from typing import Dict, Any
from agents.finrobot_base import FinRobot
from autogen import UserProxyAgent
from agents.single_agent import SingleAssistant

# Import from main.py. Ensure your sys.path in orchestrator.py is correctly set up
# to allow this import, or consider moving these utility functions to a separate
# shared module (e.g., `llm_utils.py`) that both main.py and agent_library.py can import.
# Corrected import: Import the 'app' object and access its 'logger' attribute.
from main import resolve_model_config, inject_model_env, app as main_app_instance


def _get_agent_llm_config(agent_type_key: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper function to resolve and inject LLM configuration for a given agent type.
    This centralizes the logic for fetching model name, resolving config, and setting env vars.
    It now includes a fallback to a global default model if an agent-specific model is not found.
    """
    llm_models_config = config.get("llm_models", {})
    model_providers = config.get("model_providers", [])
    
    # Attempt to get the agent-specific model name first
    model_name = llm_models_config.get(agent_type_key)

    # If no agent-specific model is found, try to get a global default model
    if not model_name:
        default_model_name = config.get("default_llm_model") # Check for a global default in the main config
        if default_model_name:
            model_name = default_model_name
            # Corrected logger call
            main_app_instance.logger.info(f"No specific model for '{agent_type_key}' found. Falling back to global default: {model_name}")
        else:
            # Corrected logger call
            main_app_instance.logger.error(f"Error: Model not specified for '{agent_type_key}' in llm_models config, and no 'default_llm_model' is defined in the main config.")
            raise ValueError(f"No LLM model configured for agent: {agent_type_key} (and no global default fallback).")

    try:
        model_config = resolve_model_config(model_name, model_providers)
        # Inject environment variables for this model.
        # This assumes SingleAssistant/FinRobot will pick up the env vars when initialized.
        inject_model_env(model_config)
        # Corrected logger call
        main_app_instance.logger.info(f"Configured '{agent_type_key}' with model: {model_config.get('model')} (API Type: {model_config.get('api_type')})")
        
        # Return only the model name in llm_config, as the API key is now in env vars.
        return {"model": model_config['model']}
        
    except ValueError as e:
        # Corrected logger call
        main_app_instance.logger.error(f"Could not resolve model config for '{agent_type_key}' (model: {model_name}): {e}")
        raise # Re-raise to propagate the configuration error
    except Exception as e:
        # Corrected logger call
        main_app_instance.logger.error(f"Unexpected error during LLM config for '{agent_type_key}': {e}")
        raise

# All functions that create a FinRobot agent have been updated to accept a 'config' dictionary.
def get_validation_agent(config: Dict[str, Any], **kwargs) -> SingleAssistant:
    return SingleAssistant(
        agent_config={
            "name": "LLMValidator",
            "description": "Validates if company resolution is sufficient to proceed. Strictly Respond with Continue or End",
            "toolkits": [],
        },
        llm_config=_get_agent_llm_config("LLMValidator", config), # Use helper
        max_consecutive_auto_reply=1,
        **kwargs
    )

def get_tool_agent(name: str, endpoint: str, task: str, config: Dict[str, Any], **kwargs) -> FinRobot:
    """Creates a generic agent for calling a specific tool or API."""
    # For generic tools, you might want a dedicated model or default to 'leader'
    return FinRobot(
        agent_config={
            "name": name,
            "description": f"Calls external API for {task}",
            "api_endpoint": endpoint
        },
        llm_config=_get_agent_llm_config("leader", config), # Assuming tools use the 'leader' model
        **kwargs,
    )

def get_company_resolver_agent(config: Dict[str, Any], **kwargs) -> FinRobot:
    """Creates the agent responsible for resolving company details."""
    return FinRobot(
        agent_config={
            "name": "company_resolver",
            "description": "Resolves company names and fetches metadata like ticker, country, and peers"
        },
        llm_config=_get_agent_llm_config("leader", config), # Uses the leader model
        **kwargs,
    )

def get_summarizer_agent(config: Dict[str, Any], **kwargs) -> SingleAssistant:
    return SingleAssistant(
        agent_config={"name": "Summarizer"},
        llm_config=_get_agent_llm_config("summarizer", config), # Use helper
        code_execution_config={"work_dir": "report", "use_docker": False},
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        **kwargs,
    )

def get_concept_summarizer_global(config: Dict[str, Any], **kwargs) -> SingleAssistant:
    return SingleAssistant(
        agent_config={"name": "Concept_Summarizer"},
        llm_config=_get_agent_llm_config("concept_cot", config), # Use helper
        code_execution_config={"work_dir": "report", "use_docker": False},
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        **kwargs,
    )

# === Leaders ===
def get_expert_investor_us(config: Dict[str, Any], **kwargs) -> FinRobot:
    return FinRobot(
        agent_config="expert_us",
        llm_config=_get_agent_llm_config("leader", config), # Use helper
        **kwargs)

def get_expert_investor_india(config: Dict[str, Any], **kwargs) -> FinRobot:
    return FinRobot(
        agent_config="expert_india",
        llm_config=_get_agent_llm_config("leader", config), # Use helper
        **kwargs)

# === Shadows (Auditors) ===
def get_shadow_auditor(config: Dict[str, Any], **kwargs) -> FinRobot:
    return FinRobot(
        agent_config="shadow_auditor",
        llm_config=_get_agent_llm_config("auditor", config), # Use helper
        **kwargs)

def get_shadow_pipeline_checker(config: Dict[str, Any], **kwargs) -> FinRobot:
    return FinRobot(
        agent_config="shadow_pipeline",
        llm_config=_get_agent_llm_config("auditor", config), # Assuming this also uses the 'auditor' model
        **kwargs)

# === Data Collectors (Data_CoT) ===
def get_data_collector_us(config: Dict[str, Any], **kwargs) -> FinRobot:
    return FinRobot(
        agent_config="data_cot_us",
        llm_config=_get_agent_llm_config("data_cot", config), # Use helper
        **kwargs,
    )

def get_data_collector_india(config: Dict[str, Any], **kwargs) -> FinRobot:
    return FinRobot(
        agent_config="data_cot_india",
        llm_config=_get_agent_llm_config("data_cot", config), # Use helper
        **kwargs)

# === Concept Summarizers (Concept_CoT) ===
def get_concept_summarizer_us(config: Dict[str, Any], **kwargs) -> FinRobot:
    return FinRobot(
        agent_config="concept_cot_us",
        llm_config=_get_agent_llm_config("concept_cot", config), # Use helper
        **kwargs)

# === Thesis Creator (Thesis_CoT) ===
def get_thesis_creator(config: Dict[str, Any], **kwargs) -> FinRobot:
    return FinRobot(
        agent_config="thesis_cot",
        llm_config=_get_agent_llm_config("thesis_cot", config), # Use helper
        **kwargs)

# === I/O Agent ===
def get_io_agent(config: Dict[str, Any], **kwargs) -> FinRobot:
    # IO agent might not need an LLM. If it does, specify its key.
    # For now, assuming it doesn't need an LLM or uses a default not tied to llm_models config.
    # If it needs an LLM, you'd add: llm_config=_get_agent_llm_config("io_agent_key", config)
    return FinRobot(agent_config="io_agent", **kwargs)

# === User Proxy ===
# This function does not create a FinRobot, so it doesn't need the config for LLM selection.
def get_user_proxy(**kwargs) -> UserProxyAgent:
    return UserProxyAgent(
        name="User_Proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config={"work_dir": kwargs.get("work_dir", "coding"), "use_docker": False},
    )
