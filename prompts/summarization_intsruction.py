summarization_prompt_library = {
    "analyze_income_stmt": {
        "input_files": ["income_statement.json", "sec_filings/sec_10k_section_7.txt"],
        "raw_data_needed": [
            "Income statement table for fiscal year",
            "10-K Section 7 text"
        ],
        "prompt_template": """{table_str}

Resource: {section_text}

Instruction: 
Conduct a comprehensive analysis of the company's income statement for the current fiscal year. 
Start with an overall revenue record, including Year-over-Year or Quarter-over-Quarter comparisons, and break down revenue sources to identify primary contributors and trends. 
Examine the Cost of Goods Sold for potential cost control issues. 
Review profit margins such as gross, operating, and net profit margins to evaluate cost efficiency, operational effectiveness, and overall profitability. 
Analyze Earnings Per Share to understand investor perspectives. 
Compare these metrics with historical data and industry or competitor benchmarks to identify growth patterns, profitability trends, and operational challenges. 
The output should be a strategic overview of the company’s financial health in a single paragraph, less than 130 words, summarizing the previous analysis into 4-5 key points under respective subheadings with specific discussion and strong data support."""
    },

    "analyze_balance_sheet": {
        "input_files": ["balance_sheet.json", "sec_filings/sec_10k_section_7.txt"],
        "raw_data_needed": [
            "Balance sheet table for fiscal year",
            "10-K Section 7 text"
        ],
        "prompt_template": """{table_str}

Resource: {section_text}

Instruction: 
Delve into a detailed scrutiny of the company's balance sheet for the most recent fiscal year, pinpointing the structure of assets, liabilities, and shareholders' equity to decode the firm's financial stability and operational efficiency. 
Focus on evaluating the liquidity through current assets versus current liabilities, the solvency via long-term debt ratios, and the equity position to gauge long-term investment potential. 
Contrast these metrics with previous years' data to highlight financial trends, improvements, or deteriorations. 
Finalize with a strategic assessment of the company's financial leverage, asset management, and capital structure, providing insights into its fiscal health and future prospects in a single paragraph. Less than 130 words."""
    },

    "analyze_cash_flow": {
        "input_files": ["cash_flow.json", "sec_filings/sec_10k_section_7.txt"],
        "raw_data_needed": [
            "Cash flow statement/CFO for fiscal year",
            "10-K Section 7 text"
        ],
        "prompt_template": """{table_str}

Resource: {section_text}

Instruction: 
Dive into a comprehensive evaluation of the company's cash flow for the latest fiscal year, focusing on cash inflows and outflows across operating, investing, and financing activities. 
Examine the operational cash flow to assess the core business profitability, scrutinize investing activities for insights into capital expenditures and investments, and review financing activities to understand debt, equity movements, and dividend policies. 
Compare these cash movements to prior periods to discern trends, sustainability, and liquidity risks. 
Conclude with an informed analysis of the company's cash management effectiveness, liquidity position, and potential for future growth or financial challenges in a single paragraph. Less than 130 words."""
    },

    "analyze_segment_stmt": {
        "input_files": ["income_statement.json", "sec_filings/sec_10k_section_7.txt"],
        "raw_data_needed": [
            "Income statement table by segment (if available)",
            "10-K Section 7 text"
        ],
        "prompt_template": """{table_str}

Resource: {section_text}

Instruction: 
Identify the company's business segments and create a segment analysis using the Management's Discussion and Analysis and the income statement, subdivided by segment with clear headings. 
Address revenue and net profit with specific data, and calculate the changes. 
Detail strategic partnerships and their impacts, including details like the companies or organizations. 
Describe product innovations and their effects on income growth. 
Quantify market share and its changes, or state market position and its changes. 
Analyze market dynamics and profit challenges, noting any effects from national policy changes. 
Include the cost side, detailing operational costs, innovation investments, and expenses from channel expansion, etc. 
Support each statement with evidence, keeping each segment analysis concise and under 60 words, accurately sourcing information. 
For each segment, consolidate the most significant findings into one clear, concise paragraph, excluding less critical or vaguely described aspects to ensure clarity and reliance on evidence-backed information. For each segment, the output should be one single paragraph within 150 words."""
    },

    "get_risk_assessment": {
        "input_files": ["company_profile.json", "sec_filings/sec_10k_section_1a.txt"],
        "raw_data_needed": [
            "Company name/profile",
            "Risk factors from 10-K Section 1A"
        ],
        "prompt_template": """Resource: Company Name: {company_name}

Risk factors:
{risk_factors}

Instruction: 
According to the given information in the 10-k report, summarize the top 3 key risks of the company. 
Then, for each key risk, break down the risk assessment into the following aspects:
1. Industry Vertical Risk: How does this industry vertical compare with others in terms of risk? Consider factors such as regulation, market volatility, and competitive landscape.
2. Cyclicality: How cyclical is this industry? Discuss the impact of economic cycles on the company’s performance.
3. Risk Quantification: Enumerate the key risk factors with supporting data if the company or segment is deemed risky.
4. Downside Protections: If the company or segment is less risky, discuss the downside protections in place. Consider factors such as diversification, long-term contracts, and government regulation.
Finally, provide a detailed and nuanced assessment that reflects the true risk landscape of the company. And Avoid any bullet points in your response."""
    },

    "get_competitors_analysis": {
        "input_files": ["competitors.json", "key_data.json"],
        "raw_data_needed": [
            "Financial metrics for target and competitors (multi-year)",
            "Competitor names"
        ],
        "prompt_template": """{table_str}

Resource: Financial metrics for {company_name} and {competitor_names}

Instruction: 
Analyze the financial metrics for {company_name} and its competitors: {competitor_names} across multiple years (indicated as 0, 1, 2, 3, with 0 being the latest year and 3 the earliest year). 
Focus on the following metrics: EBITDA Margin, EV/EBITDA, FCF Conversion, Gross Margin, ROIC, Revenue, and Revenue Growth. 
For each year: Year-over-Year Trends: Identify and discuss the trends for each metric from the earliest year (3) to the latest year (0) for {company_name}. 
Highlight any significant improvements, declines, or stability in these metrics over time. 
Competitor Comparison: For each year, compare {company_name} against its {competitor_names} for each metric. Evaluate how {company_name} performs relative to its competitors, noting where it outperforms or lags behind.
Metric-Specific Insights:
EBITDA Margin: Discuss the profitability of {company_name} compared to its competitors, particularly in the most recent year.
EV/EBITDA: Provide insights on the valuation and whether {company_name} is over or undervalued compared to its competitors in each year.
FCF Conversion: Evaluate the cash flow efficiency of {company_name} relative to its competitors over time.
Gross Margin: Analyze the cost efficiency and profitability in each year.
ROIC: Discuss the return on invested capital and what it suggests about the company's efficiency in generating returns from its investments, especially focusing on recent trends.
Revenue and Revenue Growth: Provide a comprehensive view of {company_name}’s revenue performance and growth trajectory, noting any significant changes or patterns.
Conclusion: Summarize the overall financial health of {company_name} based on these metrics. Discuss how its performance over these years and across these metrics might justify or contradict its current market valuation (as reflected in the EV/EBITDA ratio).
Avoid using any bullet points."""
    },

    "analyze_company_description": {
        "input_files": ["company_profile.json", "sec_filings/sec_10k_section_1.txt", "sec_filings/sec_10k_section_7.txt"],
        "raw_data_needed": [
            "Company profile (founded year, sector, etc.)",
            "10-K Section 1 and Section 7 text"
        ],
        "prompt_template": """Resource: 
Company Name: {company_name}
Business summary: {business_summary}
Management's Discussion and Analysis: {section_7}

Instruction: 
According to the given information, 
1. Briefly describe the company overview and company’s industry, using the structure: "Founded in xxxx, 'company name' is a xxxx that provides ..... 
2. Highlight core strengths and competitive advantages key products or services,
3. Include topics about end market (geography), major customers (blue chip or not), market share for market position section,
4. Identify current industry trends, opportunities, and challenges that influence the company’s strategy,
5. Outline recent strategic initiatives such as product launches, acquisitions, or new partnerships, and describe the company's response to market conditions. 
Less than 300 words."""
    },

    "analyze_business_highlights": {
        "input_files": [
            "sec_filings/sec_10k_section_1.txt",
            "sec_filings/sec_10k_section_7.txt"
        ],
        "raw_data_needed": [
            "Business summary (10-K Section 1)",
            "Management's Discussion (Section 7)"
        ],
        "prompt_template": (
            "Business summary:\n"
            "{business_summary}\n\n"
            "Management's Discussion and Analysis of Financial Condition and Results of Operations:\n"
            "{section_7}\n\n"
            "Instruction:\n"
            "According to the given information above, describe the performance highlights for each company's business line. "
            "Each business description should contain one sentence of a summarization and one sentence of explanation."
        )
    },
}
