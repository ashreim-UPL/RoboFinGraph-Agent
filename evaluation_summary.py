# Updated evaluation_summary.py

import os
import subprocess
import json
import time
from datetime import datetime
import pandas as pd
import sys

# Use a non-interactive backend for plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

python_exe = sys.executable  # Path to the active Python interpreter

# === Configurations ===
companies = ["Apple", "Amazon", "Nvidea", "DMART", "MRF Tyres"] #openai
years = ["2024", "2023"]
providers = ["qwen", "meta", "deepseek"]  # List of LLM providers/ don't use MIXTRAL too small conext window for summrizer
config_file = "langgraph_config.json"
main_py_path = "main.py"   # Entry-point script

def run_pipeline(company, year, provider):
    """Invoke the main orchestration for a given provider."""
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    model_map = config.get(f"{provider}_llm_models", {})
    model_override = json.dumps(model_map)

    cmd = [
        python_exe, main_py_path, company, year,
        "--config", config_file,
        "--llm_models", model_override,
        "--report_type", "kpi_bullet_insights"
    ]
    print(f"\n=== Running: {company} | {year} | {provider} ===")
    print(" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    time.sleep(2)  # Prevent API rate-limit issues

def find_metrics_files():
    """Recursively find all pipeline_metrics.json under the report directory."""
    files = []
    for root, _, filenames in os.walk("report"):
        for fn in filenames:
            if fn == "pipeline_metrics.json":
                files.append(os.path.join(root, fn))
    return files

def parse_metrics_file(path):
    """Extract key metrics from a pipeline_metrics.json file and annotate with metadata."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        kpis = metrics.get("final_evaluation", {}).get("kpis", {})
        icaif = metrics.get("final_evaluation", {}).get("icaif_scores", {})

        # Derive company, year, provider from the directory name
        parts = os.path.basename(os.path.dirname(path)).split("_")
        company = parts[0]
        year = parts[1] if len(parts) > 1 else ""
        provider = parts[2] if len(parts) > 2 else ""
        report_date = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d")

        return {
            "Company": company,
            "Year": year,
            "Provider": provider,
            "Report Date": report_date,
            "LLM Provider": kpis.get("main_llm_provider", ""),
            "#Models Used": len(kpis.get("unique_llm_models", [])),
            "Total Tokens Sent": kpis.get("total_tokens_sent", 0),
            "Total Tokens Generated": kpis.get("total_tokens_generated", 0),
            "Total LLM Cost": kpis.get("total_pipeline_cost_usd", 0.0),
            "Retry_Count":kpis.get("retry_count", 0),
            "Start Time": kpis.get("pipeline_start_time", ""),
            "End Time": kpis.get("pipeline_end_time", ""),
            "Duration (sec)": kpis.get("pipeline_total_duration_sec", ""),
            "#Steps": len(kpis.get("pipeline_nodes_sequence", [])),
            "ICAIF Accuracy": kpis.get("icaif_accuracy", icaif.get("accuracy", "")),
            "ICAIF Logicality": kpis.get("icaif_logicality", icaif.get("logicality", "")),
            "ICAIF Storytelling": kpis.get("icaif_storytelling", icaif.get("storytelling", "")),
        }
    except Exception as e:
        print(f"Error parsing {path}: {e}")
        return None

def plot_metrics(df):
    """Generate and save charts summarizing LLM cost, token usage, and accuracy."""
    if df.empty or "Provider" not in df.columns:
        print("No data available to plot.")
        return

    # Mean LLM cost by provider
    plt.figure(figsize=(10, 6))
    df.groupby("Provider")["Total LLM Cost"].mean().plot(kind="bar")
    plt.title("Mean LLM Cost by Provider")
    plt.ylabel("Mean Cost (USD)")
    plt.savefig(os.path.join("report", "llm_cost_by_provider.png"))
    plt.close()

    # Boxplot of total tokens sent by provider
    plt.figure(figsize=(10, 6))
    df.boxplot(column="Total Tokens Sent", by="Provider")
    plt.title("Total Tokens Sent by Provider")
    plt.suptitle("")  # Remove automatic subtitle
    plt.ylabel("Tokens Sent")
    plt.savefig(os.path.join("report", "tokens_sent_by_provider.png"))
    plt.close()

    # ICAIF accuracy over time
    df["Report Date"] = pd.to_datetime(df["Report Date"])
    plt.figure(figsize=(10, 6))
    for prov in df["Provider"].unique():
        subset = df[df["Provider"] == prov].sort_values("Report Date")
        plt.plot(subset["Report Date"], subset["ICAIF Accuracy"], marker="o", label=prov)
    plt.title("ICAIF Accuracy Over Time by Provider")
    plt.xlabel("Report Date")
    plt.ylabel("ICAIF Accuracy")
    plt.legend()
    plt.savefig(os.path.join("report", "icaif_accuracy_over_time.png"))
    plt.close()

    print("Charts saved in the 'report' directory.")

if __name__ == "__main__":
    # 1. Optionally invoke the pipeline for each combination (uncomment to enable)
    for company in companies:
        for year in years:
            for provider in providers:
                run_pipeline(company, year, provider)

    # 2. Collect metrics from all runs
    rows = []
    for path in find_metrics_files():
        row = parse_metrics_file(path)
        if row:
            rows.append(row)
    df = pd.DataFrame(rows)

    # 3. Save consolidated metrics to Excel under report/
    summary_path = os.path.join("report", "finrobot_pipeline_metrics.xlsx")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    df.to_excel(summary_path, index=False)
    print(f"Saved all metrics to {summary_path}")

    # 4. Create summary charts
    plot_metrics(df)
