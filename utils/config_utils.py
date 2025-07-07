import os
import json
from pathlib import Path
import httpx
from httpx import HTTPStatusError
import time
from typing import Any, List, Dict, Tuple
from threading import Lock
from langchain_community.chat_models import ChatOpenAI
from langchain_together import ChatTogether
from langchain.schema import BaseMessage, AIMessage
from langchain_community.callbacks.manager import get_openai_callback
from agents.state_types import AgentState, NodeState
from datetime import datetime, timezone

from utils.logger import get_logger, log_event

logger = get_logger()
_APP_CONFIG: Dict[str, Any] = {}
_APP_CONFIG_LOCK = Lock()

def get_config_path() -> Path:
    """Resolve config path from env or fallback location."""
    env_path = os.getenv("CONFIG_PATH")
    if env_path and Path(env_path).is_file():
        return Path(env_path)
    return Path(__file__).resolve().parent.parent / "langgraph_config.json"

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Loads config from path (or default) and returns as dict."""
    path = Path(config_path) if config_path else get_config_path()
    if not path.is_file():
        raise FileNotFoundError(f"Config not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def inject_provider_pricing(config: Dict[str, Any]) -> None: # RENAMED
    # First, handle Together.ai pricing dynamically for all Together-hosted models
    together_api_key_found = False
    together_base_url = None
    together_api_key = None

    for prov, pdata in config.get("providers", {}).items():
        if prov != "openai" and prov != "google" and \
           ("together.ai" in pdata.get("base_url", "") or "together.xyz" in pdata.get("base_url", "")):
            if pdata.get("api_key"):
                together_api_key = pdata["api_key"]
                together_base_url = pdata["base_url"].rstrip("/")
                together_api_key_found = True
                break

    pricing_map: Dict[str, Dict[str, float]] = {}
    if together_api_key_found:
        pricing_map = fetch_together_pricing(together_base_url, together_api_key)
        config["_together_pricing_map"] = pricing_map # Store globally
    else:
        logger.warning("No valid Together.ai API key found for Together-hosted models. Cannot fetch Together.ai pricing dynamically.")

    # Now, iterate through all providers to inject pricing into environment variables
    for prov, pdata in config.get("providers", {}).items():
        price_in, price_out = 0.0, 0.0

        # Prioritize pricing from the config itself if explicitly defined
        if "pricing" in pdata:
            price_in = pdata["pricing"].get("input", 0.0)
            price_out = pdata["pricing"].get("output", 0.0)

        # For Together-hosted models, try to get dynamic pricing from the fetched map
        elif prov != "openai" and prov != "google" and config.get("_together_pricing_map"):
            model_name_for_pricing = pdata.get("models")[0] if pdata.get("models") else None
            if model_name_for_pricing:
                model_pricing = config["_together_pricing_map"].get(model_name_for_pricing, {})
                price_in = model_pricing.get("input", price_in) # Use fallback if not found
                price_out = model_pricing.get("output", price_out) # Use fallback if not found

        # Store in environment
        os.environ[f"{prov.upper()}_PRICE_INPUT"]  = str(price_in)
        os.environ[f"{prov.upper()}_PRICE_OUTPUT"] = str(price_out)

        logger.info(f"Injected pricing for {prov}: in={price_in} out={price_out}")
        log_event("provider_pricing_injected", {"provider": prov, "input": price_in, "output": price_out})

def prepare_config_and_env(
    config_path: str,
    llm_override: Dict[str, str],
    provider_override: str = None
) -> Dict[str, Any]:
    # 1) load base JSON
    app_config = load_config(config_path)

    # 2) override default_provider if requested
    if provider_override:
        app_config["default_provider"] = provider_override.lower()
        logger.info(f"Overrode default_provider → {provider_override}")

    # 2.1) inject Together-AI pricing right after we know default_provider
    inject_provider_pricing(app_config)

    # 3) pick the <provider>_llm_models block
    provider = app_config["default_provider"]
    section = f"{provider}_llm_models"
    final_llm = app_config.get(section, {}).copy()

    # 4) if you passed any per-agent overrides, use those instead
    if llm_override:
        final_llm = llm_override.copy()
        logger.info(f"Using CLI LLM override: {final_llm}")

    app_config["llm_models"] = final_llm

    # 5) inject every provider’s API key
    inject_env_from_config(app_config)

    # 6) inject each model’s env (so resolve_model_config can work)
    for agent, model in final_llm.items():
        if not model:
            continue
        mc = resolve_model_config(model, app_config)
        inject_model_env(mc)
        logger.info(f"Env preloaded for agent '{agent}' → model '{model}'")

    # finally, stash the fully‐mutated config
    with _APP_CONFIG_LOCK:
        _APP_CONFIG.clear()
        _APP_CONFIG.update(app_config)

    return app_config

def get_app_config() -> Dict[str, Any]:
    """Return the same config that was loaded & mutated by prepare_config_and_env."""
    with _APP_CONFIG_LOCK:
        if not _APP_CONFIG:
            raise RuntimeError("app_config has not been initialized")
        return _APP_CONFIG

def fetch_together_pricing(base_url: str, api_key: str) -> Dict[str, Dict[str, float]]:
    """
    Call Together’s GET /models endpoint to retrieve pricing per model.
    Returns a dict: { model_id: { "input": input_price_per_token,
                                  "output": output_price_per_token } }
    """
    url = base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = httpx.get(url, headers=headers, timeout=30.0)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to fetch Together models/pricing: {e}")
        return {}

    models = resp.json()
    pricing_map: Dict[str, Dict[str, float]] = {}
    for m in models:
        pid = m.get("id")
        pr = m.get("pricing", {})
        pricing_map[pid] = {
            "input":  pr.get("input", 0.0),
            "output": pr.get("output", 0.0),
        }      
    logger.info(f"Loaded pricing for {len(pricing_map)} Together models")
    return pricing_map
        
def inject_env_from_config(config: Dict[str, Any]) -> None:
    """Injects all API keys and provider URLs from config into os.environ."""
    # top-level API keys
    for k, v in config.get("api_keys", {}).items():
        if k and v:
            os.environ[k] = v
            logger.info(f"Injected API key: {k}")
            log_event("api_key_injected", {"key": k})

    # each provider’s creds
    for name, pdata in config.get("providers", {}).items():
        key = pdata.get("api_key")
        if key:
            env_key = f"{name.upper()}_API_KEY"
            os.environ[env_key] = key
            logger.info(f"Injected {env_key}")
            log_event("provider_api_key_injected", {"env_var": env_key})
        if "base_url" in pdata:
            env_url = f"{name.upper()}_BASE_URL"
            os.environ[env_url] = pdata["base_url"]
            logger.info(f"Injected {env_url}")
            log_event("provider_base_url_injected", {"env_var": env_url})

def available_llm_options(config: Dict[str, Any]) -> Dict[str, List[str]]:
    """Returns available models per provider for UI selection."""
    return {
        k: v.get("models", [])
        for k, v in config.get("providers", {}).items()
    }

def get_node_model(node_name: str, config: Dict[str, Any], provider: str = None) -> Tuple[str, str]:
    """
    Dynamically select which <provider>_llm_models block to use for this node.
    """
    default_p = (provider or config.get("default_provider", "openai")).lower()
    providers = set(config.get("providers", {}).keys())
    if default_p not in providers:
        raise ValueError(f"Unknown provider '{default_p}', must be one of {sorted(providers)}")

    section = f"{default_p}_llm_models"
    mapping = config.get(section, {})
    if not isinstance(mapping, dict):
        raise ValueError(f"No models section named '{section}' in config")

    key = node_name.lower().removesuffix("_agent")
    model = mapping.get(key)
    if not model:
        raise ValueError(f"No model for agent '{key}' under '{section}'")
    logger.debug(f"Node '{node_name}' → provider='{default_p}', model='{model}'")
    return default_p, model

def get_provider_creds(provider: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Return API keys/base_url for a given provider, with together-fallback for other models."""
    p = provider
    if p in ("mixtral", "qwen", "deepseek", "meta"):
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
    elif api_type == "google": # Corrected syntax
        os.environ["GOOGLE_API_KEY"] = model_config.get("api_key", "")
        # Google Generative AI typically doesn't use a configurable base_url for direct API
        # Set to empty string if base_url is None in config to prevent errors
        os.environ["GOOGLE_BASE_URL"] = model_config.get("base_url", "") or "" 
    else: # Covers Together and other Together-hosted providers
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
    # For others, use Together creds, since that's how you're serving it.
    creds = config.get("providers", {}).get(provider, {})
    api_key  = creds.get("api_key")
    base_url = creds.get("base_url")
    temperature = float(config.get("temperature", 0.3))
    max_tokens = int(config.get("max_tokens", 4096))
    tool_timeout = int(config.get("tool_timeout", 30))

    if provider == "openai":
        # 1) Ensure the key is in the env (or pass it by its named arg)
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = base_url
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=api_key,
            openai_api_base=base_url,
            timeout=tool_timeout
        )
    elif provider == "google": # NEW: Google provider init
        # GOOGLE_API_KEY is picked from env var; no need for explicit param if env var is set
        os.environ["GOOGLE_API_KEY"] = api_key 
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=tool_timeout,
        )
    else:
        # 1) Push into the Together env vars
        os.environ["TOGETHER_API_KEY"] = api_key
        os.environ["TOGETHER_BASE_URL"] = base_url
        # 2) Call ChatTogether *without* an auth_token kwarg
        return ChatTogether(
            model=model_name,
            temperature=temperature,     
            max_tokens=max_tokens,      
            base_url=base_url,      
            request_timeout=tool_timeout, 
        )

class LangGraphLLMExecutor:
    """
    Wraps a LangChain chat model for use in LangGraph nodes,
    auto-selecting provider/model per node and capturing metrics.
    """
    def __init__(self, node_name: str):
        self.node_name = node_name
        self.config = get_app_config()
        self.provider, self.model_name = get_node_model(
            node_name,
            config=self.config
        )
        self.llm = _init_llm(
            provider=self.provider,
            model_name=self.model_name,
            config=self.config
        )

    def generate(self, messages: List[BaseMessage], agent_state: AgentState):
        model_start = datetime.now(timezone.utc)
        
        # Initialize metrics
        prompt_tokens = 0.0
        completion_tokens = 0.0
        cost = 0.0
        actual_retries = 0 # Track actual retries

        # 1) audit inputs
        for incoming in messages:
            agent_state.messages.append({
                "role":    incoming.type,
                "content": incoming.content,
                "ts":      datetime.now(timezone.utc).isoformat()
            })

        # 2) call LLM + measure usage
        if self.provider == "openai":
            with get_openai_callback() as cb:
                response = self.llm.invoke(messages)
            prompt_tokens     = cb.prompt_tokens
            completion_tokens = cb.completion_tokens
            cost              = cb.total_cost
        elif self.provider == "google": 
            response = self.llm.invoke(messages)
            try:
                prompt_tokens = self.llm.get_num_tokens_from_messages(messages)
                completion_tokens = self.llm.get_num_tokens(response.content)
            except NotImplementedError:
                # Fallback if get_num_tokens is not implemented for some reason
                logger.warning(f"get_num_tokens or get_num_tokens_from_messages not available for {self.provider}:{self.model_name}. Token counts may be inaccurate.")

            # Calculate cost using your pre-defined pricing
            price_in  = float(os.environ.get("GOOGLE_PRICE_INPUT", 0.0))
            price_out = float(os.environ.get("GOOGLE_PRICE_OUTPUT", 0.0))
            cost      = prompt_tokens * price_in + completion_tokens * price_out

        else: # Logic for Together-hosted models (mixtral, qwen, meta, deepseek)
            max_retries   = 3
            backoff_base  = 2

            for attempt in range(1, max_retries + 1):
                try:
                    response = self.llm.invoke(messages)
                    break
                except HTTPStatusError as e:
                    status = e.response.status_code
                    if 500 <= status < 600 and attempt < max_retries:
                        actual_retries += 1 # Increment actual_retries
                        wait = backoff_base ** (attempt - 1)
                        logger.warning(
                            f"{self.provider} 5xx ({status}), retry {attempt}/{max_retries}, backoff {wait}s"
                        )
                        time.sleep(wait)
                        continue
                    else:
                        raise # Re-raise for non-retriable errors or max retries reached

            # Get token usage from response (prefer usage_metadata, fallback to usage)
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                prompt_tokens     = response.usage_metadata.get("input_tokens", 0)
                completion_tokens = response.usage_metadata.get("output_tokens", 0)
            else:
                usage = getattr(response, "usage", {}) or {}
                prompt_tokens     = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

            # compute cost from injected pricing
            price_in  = float(os.environ.get(f"{self.provider.upper()}_PRICE_INPUT", 0.0))
            price_out = float(os.environ.get(f"{self.provider.upper()}_PRICE_OUTPUT", 0.0))
            cost = prompt_tokens * price_in + completion_tokens * price_out
            # retry_count is already handled by `actual_retries`

        # 3) update global agent_state totals
        agent_state.tokens_sent      += prompt_tokens
        agent_state.tokens_generated += completion_tokens
        agent_state.cost_llm         += cost

        # 4) record per-node metrics
        rec = getattr(agent_state, "_current_node_record", None)
        if rec is not None:
            rec["tokens_sent"]      += prompt_tokens
            rec["tokens_generated"] += completion_tokens
            rec["cost_llm"]         += cost
            rec.setdefault("tools_used", []).append(f"{self.provider}:{self.model_name}")

        # also store in memory.models_used
        model_end = datetime.now(timezone.utc)
        record = {
            "agent":             self.node_name,
            "provider":          self.provider,
            "model":             self.model_name,
            "tokens_sent":       prompt_tokens,
            "tokens_generated":  completion_tokens,
            "cost_llm":          cost,
            "model_start_time":  model_start.isoformat(),
            "model_end_time":    model_end.isoformat(),
            "model_duration":    (model_end - model_start).total_seconds(),
            "retry_count":       actual_retries 
        }
        agent_state.memory.setdefault("models_used", []).append(record)

        # 5) audit output (single, timestamped append)
        agent_state.messages.append({
            "role":    "assistant",
            "content": response.content,
            "ts":      datetime.now(timezone.utc).isoformat()
        })

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
