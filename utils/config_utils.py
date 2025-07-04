import os
import json
from pathlib import Path
from typing import Any, List, Dict, Tuple
from langchain_community.chat_models import ChatOpenAI
from langchain_together import ChatTogether
from langchain.schema import BaseMessage, AIMessage
from langchain.callbacks import get_openai_callback
from agents.state_types import AgentState, NodeState

from utils.logger import get_logger, log_event

logger = get_logger()

# --- Load Configuration --- #
env_path = os.getenv("CONFIG_PATH")
if env_path and Path(env_path).is_file():
    CONFIG_PATH = Path(env_path)
else:
    # Default location: two levels up from this file, langgraph_config.json
    CONFIG_PATH = (Path(__file__).resolve().parent.parent / "langgraph_config.json")

try:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        CONFIG: Dict[str, Any] = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"FinRobot config not found at {CONFIG_PATH}")

# Global settings
DEFAULT_PROVIDER = CONFIG.get("default_provider", "openai").lower()
TEMPERATURE      = float(CONFIG.get("temperature", 0.1))
MAX_TOKENS       = int(CONFIG.get("max_tokens", 4096))
TOOL_TIMEOUT     = int(CONFIG.get("tool_timeout", 30))

# Providers credentials & endpoints
PROVIDERS: Dict[str, Dict[str, str]] = CONFIG.get("providers", {})

# Per-node model mappings by provider
OPENAI_MODELS : Dict[str, str] = CONFIG.get("openai_llm_models", {})
MIXTRAL_MODELS: Dict[str, str] = CONFIG.get("mixtral_llm_models", {})
QWEN_MODELS   : Dict[str, str] = CONFIG.get("qwen_llm_models", {})


def _get_node_config(node_name: str) -> Tuple[str, str]:
    """
    Returns (provider, model_name) for a given node based on DEFAULT_PROVIDER.
    """
    key = node_name.lower().removesuffix("_agent")
    if DEFAULT_PROVIDER == "openai":
        model = OPENAI_MODELS.get(key)
        provider = "openai"
    elif DEFAULT_PROVIDER == "together":
        model = MIXTRAL_MODELS.get(key)
        provider = "together"
    elif DEFAULT_PROVIDER == "qwen":
        model = QWEN_MODELS.get(key)
        provider = "qwen"
    else:
        raise ValueError(f"Unsupported default provider: {DEFAULT_PROVIDER}")

    if not model:
        raise ValueError(f"No model configured for node '{key}' under provider '{provider}'")
    return provider, model


def _init_llm(provider: str, model_name: str) -> Any:
    """
    Initializes a LangChain chat model for the given provider and model.
    """
    # For Qwen, use Together creds, since that's how you're serving it.
    actual_provider = provider
    if provider == "qwen":
        actual_provider = "together"
    conf = PROVIDERS.get(actual_provider, {})
    api_key  = conf.get("api_key")
    base_url = conf.get("base_url")

    if actual_provider == "openai":
        return ChatOpenAI(
            model=model_name,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            openai_api_key=api_key,
            openai_api_base=base_url,
            timeout=TOOL_TIMEOUT
        )

    elif actual_provider == "together":
        return ChatTogether(
            model=model_name,
            temperature=TEMPERATURE,
            auth_token=api_key,
            base_url=base_url,
            request_timeout=TOOL_TIMEOUT
        )
    else:
        raise ValueError(f"Unsupported provider '{provider}'")
class LangGraphLLMExecutor:
    """
    Wraps a LangChain chat model for use in LangGraph nodes,
    auto-selecting provider/model per node and capturing metrics."""
    def __init__(self, node_name: str):
        self.provider, self.model_name = _get_node_config(node_name)
        self.llm = _init_llm(self.provider, self.model_name)

    def generate(
        self,
        messages: List[BaseMessage],
        agent_state: AgentState
    ) -> AIMessage:
        # audit input
        for msg in messages:
            agent_state.messages.append({"role": msg.type, "content": msg.content})

        # call LLM
        if self.provider == "openai":
            with get_openai_callback() as cb:
                response: AIMessage = self.llm(messages)
            # update agent totals
            agent_state.tokens_sent      += cb.prompt_tokens
            agent_state.tokens_generated += cb.completion_tokens
            agent_state.cost_llm         += cb.total_cost
        else:
            response: AIMessage = self.llm(messages)
            usage = getattr(response, "usage", {}) or {}
            agent_state.tokens_sent      += usage.get("prompt_tokens", 0)
            agent_state.tokens_generated += usage.get("completion_tokens", 0)
            # (if Together returns cost, add that here)

        # record which model you used
        agent_state.memory.setdefault("models_used", []).append(f"{self.provider}:{self.model_name}")

        # audit output
        agent_state.messages.append({"role": "assistant", "content": response.content})
        return response

def resolve_model_config(model_name: str, model_providers: List[Dict[str, Any]], provider_secrets: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find the provider entry for `model_name` and return its config,
    looking up credentials from provider_secrets.
    """
    for provider in model_providers:
        if "models" in provider and model_name in provider["models"]:
            provider_name = provider.get("provider", "")
            secrets = provider_secrets.get(provider_name, {})
            config = {
                "model": model_name,
                "api_key": secrets.get("api_key", ""),
                "api_type": provider_name,
            }
            if "base_url" in secrets:
                config["base_url"] = secrets["base_url"]

            logger.info(f"Resolved config for model '{model_name}' from provider '{provider_name}'")
            log_event("model_config_resolved", config)
            return config

    err_msg = f"Model '{model_name}' not found in model_providers."
    logger.error(err_msg)
    log_event("model_config_error", {"model": model_name, "error": err_msg})
    raise ValueError(err_msg)


def inject_model_env(model_config: Dict[str, Any]):
    """
    Injects the API key (and base_url if needed) into environment vars
    and logs the injection event.
    """
    api_type = model_config.get("api_type", "").lower()
    model_name = model_config.get("model", "<unknown>")

    if api_type == "openai":
        os.environ["OPENAI_API_KEY"] = model_config.get("api_key", "")
        msg = f"Injected OPENAI_API_KEY for model: {model_name}"
        logger.info(msg)
        log_event("env_injection", {"model": model_name, "api": "openai"})

    elif api_type == "together":
        os.environ["TOGETHER_API_KEY"] = model_config.get("api_key", "")
        os.environ["TOGETHER_BASE_URL"] = model_config.get("base_url", "")
        msg = f"Injected TOGETHER_API_KEY and BASE_URL for model: {model_name}"
        logger.info(msg)
        log_event("env_injection", {
            "model": model_name,
            "api": "together",
            "base_url": model_config.get("base_url", "")
        })

    else:
        msg = f"No env rules for API type: '{api_type}' (model: {model_name})"
        logger.info(msg)
        log_event("env_injection_skipped", {"model": model_name, "api_type": api_type})

def test_get_node_config_openai():
    global DEFAULT_PROVIDER, OPENAI_MODELS
    DEFAULT_PROVIDER = "openai"
    OPENAI_MODELS = {"resolve_company": "gpt-3.5-turbo"}
    assert _get_node_config("resolve_company") == ("openai", "gpt-3.5-turbo")

def test_get_node_config_missing():
    global DEFAULT_PROVIDER, OPENAI_MODELS
    DEFAULT_PROVIDER = "openai"
    OPENAI_MODELS = {}
    try:
        _get_node_config("resolve_company")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

if __name__ == "__main__":
    # Setup a minimal fake config for openai
    OPENAI_MODELS = {"resolve_company": "gpt-3.5-turbo"}
    PROVIDERS["openai"] = {"api_key": "testkey"}
    DEFAULT_PROVIDER = "openai"

    provider, model = _get_node_config("resolve_company")
    print("Provider/model:", provider, model)

    # Now test env injection
    inject_model_env({"api_type": "openai", "model": model, "api_key": "testkey"})
    print("Env var loaded?", os.environ.get("OPENAI_API_KEY"))