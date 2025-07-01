# config/agent_library.py
"""
Central config registry: defines all named agent instances for the LangGraph pipeline.
Each agent gets its LLM resolved and environment injected here, with structured logging.
"""
import json
from typing import Dict, Any

from agents.finrobot_base import FinRobot
from autogen import UserProxyAgent
from agents.single_agent import SingleAssistant

from utils.logger import get_logger, log_event
from utils.config_utils import resolve_model_config, inject_model_env

logger = get_logger()


def _get_agent_llm_config(agent_key: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve and inject LLM configuration for a given agent.
    Falls back to a global default if agent-specific model is not set.
    Emits fallback events when using default model.
    """
    llm_map = config.get("llm_models", {})
    providers = config.get("model_providers", [])

    # Try agent-specific model
    model_name = llm_map.get(agent_key)
    if not model_name:
        # Fallback to global default
        default_llm = config.get("default_llm_model")
        if default_llm:
            model_name = default_llm
            msg = f"[{agent_key}] No specific model configured; falling back to default: {model_name}"
            logger.info(msg)
            log_event("agent_llm_fallback", {"agent": agent_key, "model": model_name})
        else:
            err = f"No LLM model for '{agent_key}' and no 'default_llm_model' provided."
            logger.error(err)
            log_event("agent_llm_error", {"agent": agent_key, "error": err})
            raise ValueError(err)

    try:
        # Resolve and inject environment variables
        mc = resolve_model_config(model_name, providers)
        inject_model_env(mc)
        info_msg = (
            f"[{agent_key}] LLM env injected -> model='{mc['model']}', api_type='{mc.get('api_type')}'"
        )
        logger.info(info_msg)
        log_event("agent_llm_configured", {"agent": agent_key, "model": mc['model']})
        return {"model": mc['model']}

    except ValueError as e:
        err_msg = f"Failed to resolve LLM config for '{agent_key}' (model: {model_name}): {e}"
        logger.error(err_msg)
        log_event("agent_llm_error", {"agent": agent_key, "error": err_msg})
        raise
    except Exception as e:
        err_msg = f"Unexpected error during LLM config for '{agent_key}': {e}"
        logger.error(err_msg)
        log_event("agent_llm_error", {"agent": agent_key, "error": err_msg})
        raise


# === Agent factories ===

def get_validation_agent(config: Dict[str, Any], **kwargs) -> SingleAssistant:
    return SingleAssistant(
        agent_config={
            "name": "LLMValidator",
            "description": "Validates company resolution; replies 'Continue' or 'End'",
            "toolkits": []
        },
        llm_config=_get_agent_llm_config("LLMValidator", config),
        max_consecutive_auto_reply=1,
        **kwargs
    )


def get_tool_agent(name: str, endpoint: str, task: str, config: Dict[str, Any], **kwargs) -> FinRobot:
    return FinRobot(
        agent_config={
            "name": name,
            "description": f"Calls external API for {task}"
        },
        llm_config=_get_agent_llm_config("leader", config),
        **kwargs
    )


def get_company_resolver_agent(config: Dict[str, Any], **kwargs) -> FinRobot:
    return FinRobot(
        agent_config={
            "name": "company_resolver",
            "description": "Fetches metadata for a given company"
        },
        llm_config=_get_agent_llm_config("leader", config),
        **kwargs
    )


def get_summarizer_agent(config: Dict[str, Any], **kwargs) -> SingleAssistant:
    return SingleAssistant(
        agent_config={"name": "Summarizer"},
        llm_config=_get_agent_llm_config("summarizer", config),
        code_execution_config={"work_dir": "report", "use_docker": False},
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        **kwargs
    )


def get_concept_summarizer_global(config: Dict[str, Any], **kwargs) -> SingleAssistant:
    return SingleAssistant(
        agent_config={"name": "Concept_Summarizer"},
        llm_config=_get_agent_llm_config("concept_cot", config),
        code_execution_config={"work_dir": "report", "use_docker": False},
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        **kwargs
    )


def get_expert_investor_us(config: Dict[str, Any], **kwargs) -> FinRobot:
    return FinRobot(
        agent_config="expert_us",
        llm_config=_get_agent_llm_config("leader", config),
        **kwargs
    )


def get_expert_investor_india(config: Dict[str, Any], **kwargs) -> FinRobot:
    return FinRobot(
        agent_config="expert_india",
        llm_config=_get_agent_llm_config("leader", config),
        **kwargs
    )


def get_shadow_auditor(config: Dict[str, Any], **kwargs) -> FinRobot:
    return FinRobot(
        agent_config="shadow_auditor",
        llm_config=_get_agent_llm_config("auditor", config),
        **kwargs
    )


def get_shadow_pipeline_checker(config: Dict[str, Any], **kwargs) -> FinRobot:
    return FinRobot(
        agent_config="shadow_pipeline",
        llm_config=_get_agent_llm_config("auditor", config),
        **kwargs
    )


def get_data_collector_us(config: Dict[str, Any], **kwargs) -> FinRobot:
    return FinRobot(
        agent_config="data_cot_us",
        llm_config=_get_agent_llm_config("data_cot", config),
        **kwargs
    )


def get_data_collector_india(config: Dict[str, Any], **kwargs) -> FinRobot:
    return FinRobot(
        agent_config="data_cot_india",
        llm_config=_get_agent_llm_config("data_cot", config),
        **kwargs
    )


def get_thesis_creator(config: Dict[str, Any], **kwargs) -> FinRobot:
    return FinRobot(
        agent_config="thesis_cot",
        llm_config=_get_agent_llm_config("thesis_cot", config),
        **kwargs
    )


def get_io_agent(config: Dict[str, Any], **kwargs) -> FinRobot:
    return FinRobot(
        agent_config="io_agent",
        **kwargs
    )


def get_user_proxy(**kwargs) -> UserProxyAgent:
    return UserProxyAgent(
        name="User_Proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config={"work_dir": kwargs.get("work_dir", "coding"), "use_docker": False},)