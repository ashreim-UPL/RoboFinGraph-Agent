import os, requests

API_KEY = "242490f7fd85226d09982fe8044b3271e5b53dd60cf538e55fa335a5e0599fd6"
headers = {"Authorization": f"Bearer {API_KEY}"}

resp = requests.get("https://api.together.ai/v1/models", headers=headers)
models = resp.json()

# e.g. lookup a single model:
def get_ctx(model_id):
    for m in models:
        if m["id"] == model_id:
            return m.get("context_length")
    return None

for model_id in [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    # …your other Together IDs…
]:
    print(model_id, "→", get_ctx(model_id))
