import os
import json
from pathlib import Path
from typing import Any, List, Dict, Tuple
from langchain_community.chat_models import ChatOpenAI
from langchain_together import ChatTogether
from langchain.schema import BaseMessage, AIMessage
from langchain.callbacks import get_openai_callback
from agents.state_types import AgentState, NodeState
from datetime import datetime, timezone

from utils.logger import get_logger, log_event

logger = get_logger()

def get_config_path() -> Path:
    """Resolve config path from env or fallback location."""
    env_path = os.getenv("CONFIG_PATH")
    if env_path and Path(env_path).is_file():
        return Path(env_path)
    # Default: two levels up from this file, langgraph_config.json
    return Path(__file__).resolve().parent.parent / "langgraph_config.json"

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Loads config from path (or default) and returns as dict."""
    path = Path(config_path) if config_path else get_config_path()
    if not path.is_file():
        raise FileNotFoundError(f"FinRobot config not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def inject_env_from_config(config: Dict[str, Any]) -> None:
    """Injects all API keys and provider URLs from config into os.environ."""
    # Inject top-level API keys
    for k, v in config.get("api_keys", {}).items():
        if k and v:
            os.environ[k] = v
            logger.info(f"Injected API key: {k}")
            log_event("api_key_injected", {"key": k})

    # Inject provider API keys and base URLs
    for provider_name, provider_data in config.get("providers", {}).items():
        env_var = f"{provider_name.upper()}_API_KEY"
        api_key = provider_data.get("api_key")
        if api_key:
            os.environ[env_var] = api_key
            logger.info(f"Injected Provider API key: {env_var}")
            log_event("provider_api_key_injected", {"env_var": env_var})
        if "base_url" in provider_data:
            url_env = f"{provider_name.upper()}_BASE_URL"
            os.environ[url_env] = provider_data["base_url"]
            logger.info(f"Injected {url_env}")
            log_event("provider_base_url_injected", {"env_var": url_env})

def available_llm_options(config: Dict[str, Any]) -> Dict[str, List[str]]:
    """Returns available models per provider for UI selection."""
    return {
        k: v.get("models", [])
        for k, v in config.get("providers", {}).items()
    }

def get_node_model(node_name: str, config: dict, provider: str = None) -> tuple:
    default_provider = (provider or config.get("default_provider", "openai")).lower()
    provider_map = {
        "openai": config.get("openai_llm_models", {}),
        "mixtral": config.get("mixtral_llm_models", {}),
        "qwen": config.get("qwen_llm_models", {}),
    }
    models = provider_map.get(default_provider)
    if not models:
        raise ValueError(f"Provider not found in config: {default_provider}")
    key = node_name.lower().removesuffix("_agent")
    model = models.get(key)
    if not model:
        raise ValueError(f"No model for node '{key}' under provider '{default_provider}'")
    return default_provider, model

def get_provider_creds(provider: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Return API keys/base_url for a given provider, with together-fallback for qwen."""
    p = provider
    if p == "qwen":
        p = "together"
    creds = config.get("providers", {}).get(p, {})
    if not creds:
        raise ValueError(f"No credentials for provider '{p}'")
    return creds

def inject_model_env(model_config: Dict[str, Any]) -> None:
    """Injects API keys for any provider."""
    api_type = model_config.get("api_type", "").lower()
    if api_type == "openai":
        os.environ["OPENAI_API_KEY"] = model_config.get("api_key", "")
        os.environ["OPENAI_BASE_URL"] = model_config.get("base_url", "")
    elif api_type in ("mixtral", "qwen"):
        os.environ["TOGETHER_API_KEY"] = model_config.get("api_key", "")
        os.environ["TOGETHER_BASE_URL"] = model_config.get("base_url", "")

def resolve_model_config(model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find the provider entry for `model_name` and return its config,
    looking up credentials from config['providers'].
    """
    for provider_name, provider_data in config.get("providers", {}).items():
        if "models" in provider_data and model_name in provider_data["models"]:
            return {
                "model": model_name,
                "api_key": provider_data.get("api_key", ""),
                "api_type": provider_name,
                "base_url": provider_data.get("base_url", "")
            }
    raise ValueError(f"Model '{model_name}' not found in config providers.")

def _init_llm(provider: str, model_name: str, config: Dict[str, Any]) -> Any:
    """
    Initializes a LangChain chat model for the given provider and model.
    """
    # For Qwen, use Together creds, since that's how you're serving it.
    creds = config.get("providers", {}).get(provider, {})
    api_key  = creds.get("api_key")
    base_url = creds.get("base_url")
    temperature = float(config.get("temperature", 0.3))
    max_tokens = int(config.get("max_tokens", 4096))
    tool_timeout = int(config.get("tool_timeout", 30))

    if provider == "openai":
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=api_key,
            openai_api_base=base_url,
            timeout=tool_timeout
        )
    elif provider == "mixtral":
        return ChatTogether(
            model=model_name,
            temperature=temperature,
            auth_token=api_key,
            base_url=base_url,
            request_timeout=tool_timeout
        )
    elif provider == "qwen":
        return ChatTogether(
            model=model_name,
            temperature=temperature,
            auth_token=api_key,
            base_url=base_url,
            request_timeout=tool_timeout
        )
    
    else:
        raise ValueError(f"Unsupported provider '{provider}'")

class LangGraphLLMExecutor:
    """
    Wraps a LangChain chat model for use in LangGraph nodes,
    auto-selecting provider/model per node and capturing metrics.
    """
    def __init__(self, node_name: str):
        self.node_name = node_name
        config = self._load_config()
        self.provider, self.model_name = get_node_model(node_name, config)
        self.llm = _init_llm(self.provider, self.model_name, config)

    @staticmethod
    def _load_config() -> dict:
        from utils.config_utils import load_config
        return load_config()

    def generate(self, messages: List[BaseMessage], agent_state: AgentState):
        # agent_state._current_node_record = step_record_dict                           check this
        # 1. Audit all inputs (for trace/debug)
        model_start = datetime.now(timezone.utc)
        for msg in messages:
            agent_state.messages.append({
                "role": msg.type,
                "content": msg.content,
                "ts": datetime.now(timezone.utc).isoformat()
            })

        # 2. Call LLM, record usage
        if self.provider == "openai":
            with get_openai_callback() as cb:
                response = self.llm(messages)
            prompt_tokens = cb.prompt_tokens
            completion_tokens = cb.completion_tokens
            cost = cb.total_cost
        else:
            response = self.llm(messages)
            usage = getattr(response, "usage", {}) or {}
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            # Support more providers here (e.g., "cost" or other fields)
            cost = usage.get("total_cost", 0.0)

        # 3. Update global agent state (totals)
        agent_state.tokens_sent += prompt_tokens
        agent_state.tokens_generated += completion_tokens
        agent_state.cost_llm += cost

        # 4. If using node/task-level records, update those as well
        rec = getattr(agent_state, "_current_node_record", None)
        if rec is not None:
            rec["tokens_sent"] += prompt_tokens
            rec["tokens_generated"] += completion_tokens
            rec["cost_llm"] += cost
            rec.setdefault("tools_used", []).append(f"{self.provider}:{self.model_name}")
        model_end = datetime.now(timezone.utc)
        model_duration = (model_end - model_start).total_seconds() 
        agent = self.node_name
        rec = {
            "agent": agent,
            "provider": self.provider,
            "model": self.model_name,
            "tokens_sent": prompt_tokens,
            "tokens_generated": completion_tokens,
            "cost_llm": cost,
            "model_start_time": model_start.isoformat(),
            "model_end_time": model_end.isoformat(),
            "model_duration": model_duration
        }
        agent_state.memory.setdefault("models_used", []).append(rec)



        # 5. Audit output
        agent_state.messages.append({
            "role": "assistant",
            "content": response.content,
            "ts": datetime.now(timezone.utc).isoformat()
        })
        # (Optionally, add LLM metadata in output if needed)

        return response

if __name__ == "__main__":
    # Test example: always pass config
    try:
        config = load_config()
        p, m = get_node_model("resolve_company", config)
        print("Resolved:", p, m)
        creds = get_provider_creds(p, config)
        print("Creds:", creds)
        inject_model_env({
            "api_type": p,
            "model": m,
            "api_key": creds.get("api_key"),
            "base_url": creds.get("base_url")
        })
        print("OPENAI_API_KEY:", os.environ.get("OPENAI_API_KEY"))
    except Exception as e:
        print("Error:", e)
