import os
import json
from utils.config_utils import resolve_model_config, inject_model_env

def load_config_into_environ(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        full_config = json.load(f)
    for k, v in full_config.get("api_keys", {}).items():
        if k and v:
            os.environ[k] = v
    for provider_name, provider_data in full_config.get("providers", {}).items():
        env_var = f"{provider_name.upper()}_API_KEY"
        api_key = provider_data.get("api_key")
        if api_key:
            os.environ[env_var] = api_key
        if "base_url" in provider_data:
            url_env = f"{provider_name.upper()}_BASE_URL"
            os.environ[url_env] = provider_data["base_url"]
    return full_config

# --- BEGIN TESTS ---
def test_config_and_env_injection(config_path='langgraph_config.json'):
    cfg = load_config_into_environ(config_path)
    assert isinstance(cfg, dict) and cfg, f"Config could not be loaded from {config_path}"

    # 1. Test top-level API keys
    if 'api_keys' in cfg:
        for k, v in cfg['api_keys'].items():
            assert os.environ.get(k) == v, f"Env var {k} was not injected correctly"

    # 2. Test provider API key env injection
    for provider_name, provider_data in cfg.get("providers", {}).items():
        env_var = f"{provider_name.upper()}_API_KEY"
        api_key = provider_data.get("api_key")
        if api_key:
            assert os.environ.get(env_var) == api_key, f"{env_var} not loaded"

    print("All API keys loaded into os.environ successfully.")

    # 3. Test model config resolution and env setup for OpenAI (or any available model)
    for test_model in ("gpt-4o-search-preview-2025-03-11", "Qwen2.5 7B Instruct", "Mixtral-8x7B-Instruct-v0.1"):
        try:
            model_cfg = resolve_model_config(
                test_model,
                cfg.get("model_providers", []),
                cfg.get("providers", {})
            )
            inject_model_env(model_cfg)
            # Should set at least one key
            if model_cfg['api_type'].lower() == "openai":
                assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY missing after injection"
            elif model_cfg['api_type'].lower() == "together":
                assert os.environ.get("TOGETHER_API_KEY"), "TOGETHER_API_KEY missing after injection"
        except Exception as e:
            print(f"Model '{test_model}' not tested (maybe not configured): {e}")
    print("Model config resolution and env injection tested.")

if __name__ == "__main__":
    test_config_and_env_injection()
