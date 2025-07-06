import os
import subprocess
import json
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import sys

python_exe = sys.executable  # <-- Use the correct venv/interpreter!

# === Configurations ===
companies = ["DMART", "MRF Tyres", "Apple", "Amazon", "Nvidea"]
years = ["2024", "2023"]
providers = ["openai"]  # add "qwen", "mixtral" as needed
config_file = "langgraph_config.json"
output_dir = "report"
main_py_path = "main.py"   # Adjust if needed

def run_pipeline(company, year, provider):
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    model_map = config.get(f"{provider}_llm_models", {})
    model_override = json.dumps(model_map)
    out_subdir = f"{output_dir}/{company}_{year}_{provider}"
    os.makedirs(out_subdir, exist_ok=True)

    cmd = [
        python_exe, main_py_path, company, year,   # <--- KEY FIX HERE!
        "--config", config_file,
        "--llm_models", model_override,
        "--report_type", "kpi_bullet_insights"
    ]
    print(f"\n=== Running: {company} | {year} | {provider} ===")
    print(" ".join([str(x) for x in cmd]))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    time.sleep(2)  # Avoid API rate limits

def find_metrics_files():
    files = []
    for root, dirs, filenames in os.walk(output_dir):
        for fn in filenames:
            if fn == "pipeline_metrics.json":
                files.append(os.path.join(root, fn))
    return files

def parse_metrics_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        kpis = metrics.get("final_evaluation", {}).get("kpis", {})
        icaif = metrics.get("final_evaluation", {}).get("icaif_scores", {})
        # Parse from directory name
        base = os.path.basename(os.path.dirname(path))
        splits = base.split("_")
        company = splits[0]
        year = splits[1] if len(splits) > 1 else ""
        provider = splits[2] if len(splits) > 2 else ""
        report_date = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d")
        row = {
            "Company": company,
            "Year": year,
            "Provider": provider,
            "Report Date": report_date,
            "LLM Provider": kpis.get("main_llm_provider", ""),
            "#Models Used": len(kpis.get("unique_llm_models", [])),
            "Total Tokens Sent": kpis.get("total_tokens_sent", 0),
            "Total Tokens Generated": kpis.get("total_tokens_generated", 0),
            "Total LLM Cost": kpis.get("total_pipeline_cost_usd", 0.0),
            "Start Time": kpis.get("pipeline_start_time", ""),
            "End Time": kpis.get("pipeline_end_time", ""),
            "Duration (sec)": kpis.get("pipeline_total_duration_sec", ""),
            "#Steps": len(kpis.get("pipeline_nodes_sequence", [])),
            "ICAIF Accuracy": kpis.get("icaif_accuracy", icaif.get("accuracy", "")),
            "ICAIF Logicality": kpis.get("icaif_logicality", icaif.get("logicality", "")),
            "ICAIF Storytelling": kpis.get("icaif_storytelling", icaif.get("storytelling", "")),
        }
        return row
    except Exception as e:
        print(f"Error parsing {path}: {e}")
        return None

def plot_metrics(df):
    # Example charts for research
    plt.figure(figsize=(10,6))
    df.groupby("Provider")["Total LLM Cost"].mean().plot(kind='bar')
    plt.title("Mean LLM Cost by Provider")
    plt.ylabel("Mean Cost (USD)")
    plt.savefig("llm_cost_by_provider.png")
    plt.close()

    plt.figure(figsize=(10,6))
    df.boxplot(column="Total Tokens Sent", by="Provider")
    plt.title("Total Tokens Sent by Provider")
    plt.suptitle("")
    plt.ylabel("Tokens Sent")
    plt.savefig("tokens_sent_by_provider.png")
    plt.close()

    plt.figure(figsize=(10,6))
    for provider in df["Provider"].unique():
        subset = df[df["Provider"] == provider]
        plt.plot(subset["Report Date"], subset["ICAIF Accuracy"], marker='o', label=provider)
    plt.title("ICAIF Accuracy Over Time by Provider")
    plt.xlabel("Report Date")
    plt.ylabel("ICAIF Accuracy")
    plt.legend()
    plt.savefig("icaif_accuracy_over_time.png")
    plt.close()
    print("Charts saved as PNGs.")

if __name__ == "__main__":
    plt.figure(figsize=(10,6))
    legend_labels = []
    # 1. Run orchestration for all company/year/provider combinations
    for company in companies:
        for year in years:
            for provider in providers:
                print("done")
                #run_pipeline(company, year, provider)

    # 2. Aggregate pipeline metrics
    metrics = []
    for path in find_metrics_files():
        row = parse_metrics_file(path)
        if row:
            metrics.append(row)
    df = pd.DataFrame(metrics)
    df.to_excel("finrobot_pipeline_metrics.xlsx", index=False)
    print("\nSaved all metrics to finrobot_pipeline_metrics.xlsx")

    # 3. (Optional) Generate research plots
    plot_metrics(df)
