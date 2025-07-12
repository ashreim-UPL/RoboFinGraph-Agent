Here’s your document, **now enhanced** with the “How to Run” section, polished for clarity and seamless integration.
You can copy-paste this directly into your README.md.

---

````markdown
![Logo](./docs/logo.png)

## How It Works

1. **Data Ingestion:**  
   - For **US equities**, RoboFinGraph collects structured financial metrics via APIs (such as Financial Modeling Prep) and parses annual report filings (PDFs) using SEC sources.
   - For **Indian equities**, **users must provide their own financial filings or RAG (retrieval-augmented generation) data**. Public APIs and open access to filings are limited in India, so the system is built to accept user-supplied company disclosures in either raw or vectorized format.
   - See [Data Sourcing for Indian Equities](#data-sourcing-for-indian-equities) for details.

2. **Entity Resolution:**  
   - Resolves company tickers, market, and peer/competitor relationships, adapting the pipeline for regional nuances.

3. **Summarization & Analysis:**  
   - Uses LLMs to chunk, embed, and retrieve the most relevant portions of company filings and data for downstream synthesis.

4. **Narrative Synthesis:**  
   - Orchestrates multi-agent workflows to generate structured, actionable equity reports, combining deterministic logic with generative reasoning.

5. **Auditing:**  
   - Optionally applies rubric-based scoring (inspired by ICAIF) for hallucination detection, completeness, and LLM output validation.

---

## How to Run RoboFinGraph

### 1. Prepare Your Configuration

- **Copy the template:**  
  Duplicate `langgraph_config_template.json` as `langgraph_config.json` in the project root.
- **Edit your config:**  
  Open `langgraph_config.json` and fill in your required API keys and endpoints (OpenAI, FMP, etc).
- **Important:** Never commit your real keys to git. Keep the real config local.

### 2. Set Up the Python Environment

- **Create and activate a virtual environment:**
  ```sh
  python -m venv venv
  # On Mac/Linux:
  source venv/bin/activate
  # On Windows:
  venv\Scripts\activate
````

* **Install dependencies:**

  ```sh
  pip install -r requirements.txt
  ```

### 3. Run the Pipeline

* Use the following command format:

  ```sh
  python main.py "<company_name>" <year> [--provider <provider_name>]
  ```

  * `<company_name>`: The target company (use quotes if the name has spaces)
  * `<year>`: The reporting year (e.g., 2024)
  * `--provider`: *(optional)* LLM provider to use (`qwen`, `openai`, etc.; OpenAI is default)
* **Example:**

  ```sh
  python main.py "Tata Consultancy Services" 2024 --provider qwen
  ```

### 4. Supported Providers

* **Default:** `openai` (robust, general-purpose)
* **Also available:** `qwen`, `deepseek`, `meta`, `google`
* **How to select:**
  Use `--provider <name>` in your command, or set defaults in `langgraph_config.json`.
* **Note:** Ensure the provider you select is enabled and properly configured in your config JSON.

### 5. LLM Node Assignment (Behind the Scenes)

The workflow assigns models strategically to balance performance and cost:

* **Company resolution/search:** Always uses OpenAI for best entity/ticker/market lookup.
* **Summarization node:**
  Uses a small/efficient model with large context window (e.g., Qwen2-72B, Deepseek) for handling large document chunks (up to 64k tokens).
* **Concept node:**
  Uses a high-end model (e.g., GPT-4, Gemini-1.5-Pro) for concept extraction and in-depth reasoning.
* **Validation node:**
  Uses your strongest available model for hallucination detection, scoring, and QA.

You may adjust assignments via `langgraph_config.json` as your needs evolve.

### 6. Tips & Troubleshooting

* If you see provider errors:
  Double-check your API keys and ensure your provider is supported in your config.
* For **Indian equities:**
  Prepare and supply your own financial filings or vectorized data. See the [Data Sourcing for Indian Equities](#data-sourcing-for-indian-equities) section for guidance.

---

## Data Sourcing for Indian Equities

> **Note:** Open, programmatic access to Indian financial filings is restricted.
> To use RoboFinGraph for Indian stocks, you **must supply your own filings** (PDFs, text, or pre-computed vector databases for RAG).
> The pipeline supports user-supplied ChromaDB/FAISS indexes or raw document ingestion.
> *See the sample data loader script for details (coming soon).*

---

## Project Summary

**RoboFinGraph** is a research-driven, multi-agent orchestration system that automates equity analysis for both US and Indian markets.
Key innovations include:

* Modular, extensible architecture (via LangGraph)
* Region-aware data sourcing and report logic
* Integrated retrieval-augmented generation (RAG)
* Support for plug-and-play LLM providers
* Emphasis on traceability, reproducibility, and agent transparency

**Key Results:**

* Outperforms baseline LLMs on standard rubric scoring and hallucination reduction (see final report).
* Achieves rapid, cost-aware report generation for both structured and unstructured financial data.

---

## Visual Summary

Include 1–2 impactful figures here, such as:

* **Pipeline Overview Graph:** Diagram the end-to-end agent flow (data, retrieval, summarization, synthesis, auditing).
* **Results Chart:** A bar chart or confusion matrix showing improved rubric scores vs. baseline, or hallucination/error rate reduction.

*Example placeholders:*

![Pipeline Overview](./docs/pipeline_overview.png)
![Rubric Scores vs Baseline](./docs/results_rubric_comparison.png)

---

> **For full methodology, detailed benchmarks, and design choices, see the project’s internal final report (contact the team for access).**

---

## References

[FinRobot (Zhou et al. 2024)](https://arxiv.org/abs/2411.08804)
[LangGraph](https://github.com/langchain-ai/langgraph)
[ICAIF Benchmark](https://acm-icaif.org/)
[IndianStockMarket API](https://indianapi.in/indian-stock-market)
[FMP API](https://site.financialmodelingprep.com/developer/docs)
[OpenAI GPT-4](https://platform.openai.com/docs/)

```

---

**Tips:**
- Place this after your project title/intro and before any team/contribution/license section.
- Add your real diagrams in `/docs/` and reference them as shown for visual impact.
- Adjust section numbers or details as your repo evolves.

Let me know if you want an even more detailed quickstart, contribution, or FAQ section!
```
