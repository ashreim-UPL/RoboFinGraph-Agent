# prompts/prompt_templates.py

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- Simple String Prompts for finrobot_base.py ---
# These are kept for compatibility with the FinRobot class, which uses .format()
instruction_message = """
Please review the previous turn and provide your feedback.
If you find any issues, please suggest improvements.
"""

role_system_message = """
As a {title}, your primary responsibility is to:
{responsibilities}
"""

leader_system_message = """
You are a leader of a group of agents. Your goal is to coordinate the agents to achieve the following objective:
{group_desc}
"""

# --- A Unified Registry for All Prompts ---
# By centralizing all prompts, we create a single, consistent way to access them.
PROMPT_REGISTRY = {
    # === Expert Investor Prompts (Leader Orchestration) ===
    "expert_investor_intro": ChatPromptTemplate.from_messages([
        ("system", "You are {region}'s Expert Investor agent. Your job is to orchestrate the full pipeline by:\n1. Identifying target companies.\n2. Instructing agents to collect, summarize, and synthesize relevant data.\n3. Producing a high-quality investment report.\n\nYou should act with leadership clarity and issue exact instructions to the appropriate agents using delegated function calls."),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ]),

    "expert_investor_summary": ChatPromptTemplate.from_messages([
        ("system", "Based on all available data files and summaries, draft a concise executive-level investment summary with:\n- Top 3 financial highlights\n- Strategic risks\n- Market positioning insights\nKeep it under 300 words."),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ]),

    # === Data CoT Agent Prompt (Data Collection) ===
    "data_cot_agent": ChatPromptTemplate.from_messages([
        ("system", "You are a Data Collection Agent for {region}. Your task is to call tools or APIs to gather financial data for the specified company and year. Once retrieved, save each result to a separate file and document any tool failures or inconsistencies."),
        MessagesPlaceholder("history"),
        # Added placeholders to the user message for more explicit instruction.
        ("human", "Please gather the following for {company_name} for the fiscal year {year}:\n- SEC or MCA reports\n- Financial ratios\n- Earnings and forecasts\n{input}")
    ]),

    # === Concept CoT Agent Prompt (Summarization/Context) ===
    "concept_cot_summarizer": ChatPromptTemplate.from_messages([
        ("system", "You are a Conceptual Analysis Agent. Your job is to convert raw financial data files into:\n- Concise, high-quality summaries\n- Extracted concepts and narratives\n- Section tags (e.g., Revenue Trends, Risk Factors)\n\nEach output should be self-contained and traceable to its source file."),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ]),

    # === Thesis CoT Prompt (Report Writer) ===
    "thesis_report_writer": ChatPromptTemplate.from_messages([
        ("system", "You are the final report generator. Using all concept summaries and structured data, produce a formatted investment report.\nEnsure professional tone and layout. Avoid redundancy, and cite input files where relevant."),
        MessagesPlaceholder("history"),
        # Added more detail to the expected sections for better guidance.
        ("human", "Generate the report with the following sections:\n- Overview\n- Financial Health (including key ratios like P/E, ROE, and Debt-to-Equity)\n- Market Risks\n- Summary Recommendations\n\n{input}")
    ]),

    # === LLM Auditor Prompt ===
    "llm_auditor": ChatPromptTemplate.from_messages([
        ("system", "You are an independent LLM-based auditor. Your tasks:\n1. Detect hallucinations in generated reports.\n2. Check for inconsistencies across sections.\n3. Provide a confidence score (0–10) and explain your reasoning.\n4. Suggest specific fixes for any issues found.\n\nBe concise but critical. Assume no prior bias."),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ]),

    # === Shadow Agent Prompt (Audit Orchestration Discipline) ===
    "shadow_agent_auditor": ChatPromptTemplate.from_messages([
        ("system", "You are a pipeline auditor ('Shadow'). Your task is to:\n- Ensure each agent followed its intended function\n- Validate tools used match expected input/output types\n- Highlight any uncoordinated or missing delegations\n\nReturn a JSON report with flags and agent compliance scores."),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ]),

    # === IO Agent Prompt ===
    "io_agent": ChatPromptTemplate.from_messages([
        ("system", "You are the IO Agent. Your job is to:\n- Check for required files in the working directory\n- Confirm correct naming structure\n- Log missing or extra files."),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ]),

    # === Common Utility Prompt (RAG-based, General API Interface) ===
    "generic_tool_caller": ChatPromptTemplate.from_messages([
        ("system", "You are calling an external API for task: {task}.\nUse minimal context, and format your results clearly.\nEnsure JSON format if structured. If failure occurs, log and flag the issue."),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ]),

    # === SYSTEM PROMPTS FOR RAW DATA GENERATION ===
    # This section is now cleaner, with prompts defined directly in the registry.
    "company_background": ChatPromptTemplate.from_messages([
        ("system", "You are a financial analyst generating detailed company background data."),
        ("user", "Generate a company overview for {company_name} based on stock fundamentals, industry profile, and business model.")
    ]),
    "financial_statement": ChatPromptTemplate.from_messages([
        ("system", "You are extracting structured financial statement information."),
        ("user", "Extract the latest financial statements for {company_name} from {source}.")
    ]),
    "historical_performance": ChatPromptTemplate.from_messages([
        ("system", "You are responsible for compiling historical financial performance."),
        ("user", "Provide the past 5 years' financial performance for {company_name} including revenue, profit, and margin trends.")
    ]),
    "forecast_trends": ChatPromptTemplate.from_messages([
        ("system", "You are responsible for projecting company financials and trends."),
        ("user", "Give future forecasts and macro/market trends for {company_name} over the next 12 months.")
    ]),
    "recent_updates": ChatPromptTemplate.from_messages([
        ("system", "You track recent announcements and filings."),
        ("user", "List recent press releases or official filings for {company_name}. Summarize major changes.")
    ]),
    "income_summary": ChatPromptTemplate.from_messages([
        ("system", "You summarize income statements with precision."),
        ("user", "Generate an income statement summary for {company_name}, highlighting earnings and losses.")
    ]),
    "risk_assessment": ChatPromptTemplate.from_messages([
        ("system", "You evaluate financial and operational risks."),
        ("user", "Provide a risk assessment for {company_name} including market, legal, and operational risks.")
    ]),
    "key_figures": ChatPromptTemplate.from_messages([
        ("system", "You extract key figures from financial data."),
        ("user", "Extract important figures for {company_name} such as EBITDA, Free Cash Flow, and ROE.")
    ]),
    "pe_eps": ChatPromptTemplate.from_messages([
        ("system", "You analyze PE ratio and EPS trends."),
        ("user", "Analyze the Price-to-Earnings and Earnings Per Share performance for {company_name}.")
    ])
}

# === Loader Function ===
# Updated to use the unified PROMPT_REGISTRY and includes error handling.
# === Loader Function ===
def get_prompt(prompt_name: str) -> ChatPromptTemplate:
    """
    Retrieve a specific prompt by name from the central prompt registry.
    """
    prompt = PROMPT_REGISTRY.get(prompt_name)
    if prompt is None:
        raise ValueError(f"Prompt '{prompt_name}' not found in the registry. Available prompts are: {list(PROMPT_REGISTRY.keys())}")
    return prompt